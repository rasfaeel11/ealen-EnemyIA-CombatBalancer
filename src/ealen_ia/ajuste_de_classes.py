"""Camada 1 (parte 3 de 3): auto-tuner via gradiente descendente numérico —
mas ajustando os ATRIBUTOS de cada classe (Dain/Eir/Nath/Il/Or) em vez dos
coeficientes globais de `ParametrosDeFormula` (isso é a parte 2, em
`auto_tuner.py`).

Por que separar em dois auto-tuners: descobrimos na prática que ajustar só
os 6 coeficientes globais cria um problema estrutural. Eles são
compartilhados por todas as 6 classes, então temos 6 graus de liberdade
tentando satisfazer 21 confrontos — um sistema subdeterminado onde a
derivada média sobre a matriz completa soma inclinações que às vezes se
cancelam (melhorar um confronto piora outro). O `AGENTS.md` já apontava pra
saída certa: "perturbar um parâmetro (ex: dano da classe X)" e "um conjunto
de parâmetros por classe" — ajustar os atributos de cada classe
independentemente dá 5 atributos x 6 classes = 30 graus de liberdade contra
21 confrontos, e cada perturbação só afeta os confrontos daquela classe
específica, não o sistema inteiro.

Este módulo pressupõe que `ParametrosDeFormula` já foi calibrado (ver
`auto_tuner.ajustar_parametros(..., apenas_espelhos=True)`) e mantém esses
coeficientes fixos enquanto ajusta os atributos — a mesma lógica de diferença
central e coordinate descent de `auto_tuner.py`, um atributo de uma classe
por vez.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ealen_ia.atributos import Atributos, ParametrosDeFormula
from ealen_ia.auto_tuner import erro_medio
from ealen_ia.classes import Classe, TODAS_AS_CLASSES
from ealen_ia.monte_carlo import matriz_de_confrontos_da_classe

# Atributos que o auto-tuner pode mexer por classe. `len_`/`ul` ficam de fora:
# nenhuma fórmula de combate em atributos.py os usa (ver a tabela do
# AGENTS.md — carisma e sabedoria não entram em combate).
ATRIBUTOS_AJUSTAVEIS: tuple[str, ...] = ("dain", "eir", "nath", "il", "or_")


@dataclass(frozen=True)
class PassoDeAjusteDeClasse:
    """Um registro de auditoria: o que mudou no atributo de uma classe, numa
    iteração, e por quê."""

    iteracao: int
    classe: str
    atributo: str
    valor_antes: float
    valor_depois: float
    derivada_estimada: float


def erro_medio_da_classe(
    nome_classe_alvo: str,
    atributos_por_classe: dict[str, Atributos],
    classes: tuple[Classe, ...],
    parametros: ParametrosDeFormula,
    n_simulacoes: int,
    seed: int | None,
) -> float:
    """Erro médio (|taxa_de_vitoria - 0.5|) só sobre os confrontos que
    envolvem `nome_classe_alvo` (o espelho dela + um cruzado contra cada
    outra classe) — ver `monte_carlo.matriz_de_confrontos_da_classe`.
    """
    matriz = matriz_de_confrontos_da_classe(
        nome_classe_alvo,
        classes,
        n=n_simulacoes,
        seed=seed,
        parametros=parametros,
        atributos_por_classe=atributos_por_classe,
    )
    erros = [resultado.erro(alvo=0.5) for resultado in matriz.values()]
    return sum(erros) / len(erros)


def _estimar_derivada_de_classe(
    nome_classe_alvo: str,
    nome_atributo: str,
    atributos_por_classe: dict[str, Atributos],
    classes: tuple[Classe, ...],
    parametros: ParametrosDeFormula,
    n_simulacoes: int,
    seed: int,
    delta: float,
) -> float:
    """Diferença central de `erro_medio_da_classe` em torno do valor atual do
    atributo — mesmo raciocínio de `auto_tuner._estimar_derivada`, só que
    perturbando um atributo dentro do dict de uma classe específica em vez de
    um campo de `ParametrosDeFormula`.

    Atributos não podem ser negativos, então o ponto baixo é limitado em 0
    (com o denominador ajustado pro espaçamento real) quando `delta` ficaria
    negativo.
    """
    valor_atual = getattr(atributos_por_classe[nome_classe_alvo], nome_atributo)
    valor_baixo = max(0.0, valor_atual - delta)
    valor_alto = valor_atual + delta

    def com_atributo(valor: float) -> dict[str, Atributos]:
        novo = dict(atributos_por_classe)
        novo[nome_classe_alvo] = replace(novo[nome_classe_alvo], **{nome_atributo: valor})
        return novo

    erro_baixo = erro_medio_da_classe(
        nome_classe_alvo, com_atributo(valor_baixo), classes, parametros, n_simulacoes, seed
    )
    erro_alto = erro_medio_da_classe(
        nome_classe_alvo, com_atributo(valor_alto), classes, parametros, n_simulacoes, seed
    )

    return (erro_alto - erro_baixo) / (valor_alto - valor_baixo)


def ajustar_atributos(
    classes: tuple[Classe, ...] = TODAS_AS_CLASSES,
    parametros: ParametrosDeFormula = ParametrosDeFormula(),
    atributos_iniciais: dict[str, Atributos] | None = None,
    n_simulacoes: int = 300,
    max_iteracoes: int = 10,
    delta: float = 2.0,
    taxa_de_aprendizado: float = 30.0,
    seed: int = 0,
    atributos_ajustaveis: tuple[str, ...] = ATRIBUTOS_AJUSTAVEIS,
) -> tuple[dict[str, Atributos], list[float], list[PassoDeAjusteDeClasse]]:
    """Roda o auto-tuner de atributos por classe e devolve os atributos finais
    (um dict nome_classe -> Atributos, nunca mutando `Classe.atributos_base`
    nem `TODAS_AS_CLASSES` — ambos `frozen` e constantes do módulo), a curva
    de convergência e o histórico detalhado de cada ajuste.

    A curva de convergência (`historico_de_erro`) mede o erro sobre a MATRIZ
    COMPLETA (todos os confrontos), não só os confrontos-estrela usados pra
    estimar cada derivada — é o desequilíbrio *entre* classes diferentes que
    queremos ver melhorar, então o gráfico final precisa refletir isso, mesmo
    que cada passo individual só olhe pros confrontos da classe sendo
    ajustada (mais barato e com menos ruído irrelevante).

    `taxa_de_aprendizado=30` (bem maior que o `0.3` do auto-tuner de
    `ParametrosDeFormula`) não é um capricho: as derivadas aqui vivem numa
    escala completamente diferente. Um coeficiente de fórmula varia em torno
    de 0-1, então uma pequena variação de erro por unidade de parâmetro já é
    "grande" relativa à escala do parâmetro. Um atributo como Nath varia em
    torno de 0-14, e empiricamente `d(erro)/d(atributo)` fica na casa de
    0.01-0.02 — uma derivada 15-30x menor em módulo. Sem compensar isso na
    taxa de aprendizado, o passo (`taxa * derivada`) fica pequeno demais pra
    mover o atributo de forma perceptível em poucas iterações (verificado na
    prática: com a taxa "padrão" de 0.3, os atributos mal se moviam depois de
    15 iterações). Não tem nada de errado matematicamente em uma derivada
    pequena — só significa que a escala do passo de gradiente descendente
    precisa ser recalibrada pra escala da variável, exatamente como ajustar
    o "tamanho do passo" ao qualquer variável de Cálculo 1 que você está
    otimizando.

    Vale registrar também outra causa de derivada zero encontrada na prática,
    sem relação com escala: se `ParametrosDeFormula` (calibrado antes, na
    Etapa A) zera um coeficiente (ex: `chance_acerto_por_il=0.0`) ou satura
    um clamp (ex: `reducao_de_dano` batendo no teto de
    `reducao_dano_maxima` pra qualquer `or_` acima de um limiar baixo), o
    atributo correspondente (`il`, `or_`) fica com derivada genuinamente
    zero — não é ruído, é a fórmula literalmente ignorando aquele atributo
    depois da Etapa A. Esse é um efeito colateral real de rodar as duas
    etapas em sequência: nem todo atributo continua "vivo" como alavanca
    depois que a Etapa A termina.

    Pra cada (classe, atributo), dentro de uma iteração, os dois pontos da
    diferença central usam o MESMO seed (common random numbers) — mesmo
    raciocínio de `auto_tuner.ajustar_parametros`.
    """
    atributos_por_classe = dict(atributos_iniciais) if atributos_iniciais else {
        classe.nome: classe.atributos_base for classe in classes
    }
    historico_de_erro: list[float] = []
    passos: list[PassoDeAjusteDeClasse] = []

    for iteracao in range(max_iteracoes):
        seed_da_iteracao = seed + iteracao

        for classe in classes:
            for nome_atributo in atributos_ajustaveis:
                valor_atual = getattr(atributos_por_classe[classe.nome], nome_atributo)

                derivada = _estimar_derivada_de_classe(
                    classe.nome,
                    nome_atributo,
                    atributos_por_classe,
                    classes,
                    parametros,
                    n_simulacoes,
                    seed_da_iteracao,
                    delta,
                )
                novo_valor = max(0.0, valor_atual - taxa_de_aprendizado * derivada)

                atributos_por_classe = dict(atributos_por_classe)
                atributos_por_classe[classe.nome] = replace(
                    atributos_por_classe[classe.nome], **{nome_atributo: novo_valor}
                )
                passos.append(
                    PassoDeAjusteDeClasse(iteracao, classe.nome, nome_atributo, valor_atual, novo_valor, derivada)
                )

        erro_da_iteracao = erro_medio(
            parametros, classes, n_simulacoes, seed_da_iteracao, atributos_por_classe=atributos_por_classe
        )
        historico_de_erro.append(erro_da_iteracao)

    return atributos_por_classe, historico_de_erro, passos

"""Camada 1 (parte 2 de 3): auto-tuner via gradiente descendente numérico —
ajusta os coeficientes GLOBAIS de `ParametrosDeFormula` (compartilhados por
todas as classes). O ajuste dos ATRIBUTOS de cada classe individualmente
(Dain/Eir/Nath/Il/Or) é a parte 3, em `ajuste_de_classes.py` — motivo da
separação no docstring daquele módulo.

Ideia central (é aqui que "derivada" vira código, não só teoria de Cálculo 1):
o erro de balanceamento não tem fórmula fechada — ele vem de uma simulação
estocástica (Monte Carlo). Então a única forma de saber "se eu aumentar esse
coeficiente, o erro sobe ou desce" é medir na prática: perturbar o parâmetro
e comparar o erro antes/depois. Isso é a reta secante da definição de
derivada — só que aqui ela nunca chega no limite (h -> 0) de verdade, porque
`erro(p)` vem de simulação estocástica: encolher demais o passo faz a
diferença medida virar ruído do Monte Carlo em vez de sinal real.

Por isso usamos a **diferença central** (em vez da "pra frente" ingênua),
que é um resultado padrão de Cálculo 1 numérico:

    pra frente:  d(erro)/d(p) ≈ [erro(p+h) - erro(p)]   / h
    central:     d(erro)/d(p) ≈ [erro(p+h) - erro(p-h)] / (2h)

A central cancela o termo de erro de 1ª ordem da expansão de Taylor de
`erro(p)` ao redor de `p` (o erro de truncamento cai de O(h) pra O(h²)), então
pra um mesmo `h` ela aproxima melhor a derivada real — e com o mesmo custo (2
simulações), porque a versão "pra frente" já precisava rodar `erro(p)` e
`erro(p+h)` separadamente.

Cada parâmetro de `ParametrosDeFormula` é ajustado nessa lógica, um de cada
vez — nunca uma derivada multivariável (ver non-goals do AGENTS.md) — e o
passo de atualização é o gradiente descendente clássico:

    parametro <- parametro - taxa_de_aprendizado * derivada
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ealen_ia.atributos import Atributos, ParametrosDeFormula
from ealen_ia.classes import Classe, TODAS_AS_CLASSES
from ealen_ia.monte_carlo import matriz_de_confrontos

# Coeficientes que o auto-tuner pode mexer. Bases fixas (hp_base, chance_*_base
# etc.) ficam de fora de propósito: elas afetam todo mundo igualmente e não
# mudam o equilíbrio *relativo* entre classes, que é o que queremos corrigir.
PARAMETROS_AJUSTAVEIS: tuple[str, ...] = (
    "hp_por_nath",
    "dano_fisico_por_dain",
    "dano_magico_por_eir",
    "chance_acerto_por_il",
    "chance_critico_por_il",
    "reducao_dano_por_or",
)


@dataclass(frozen=True)
class PassoDeAjuste:
    """Um registro de auditoria: o que mudou em um parâmetro, numa iteração, e por quê."""

    iteracao: int
    parametro: str
    valor_antes: float
    valor_depois: float
    derivada_estimada: float


def erro_medio(
    parametros: ParametrosDeFormula,
    classes: tuple[Classe, ...] = TODAS_AS_CLASSES,
    n_simulacoes: int = 200,
    seed: int | None = None,
    apenas_espelhos: bool = False,
    atributos_por_classe: dict[str, Atributos] | None = None,
) -> float:
    """Erro médio (|taxa_de_vitoria - 0.5|) sobre os confrontos da matriz.

    Simplificação assumida aqui: todo confronto usa alvo=0.5, mesmo os que por
    design talvez devessem ser assimétricos (ex: um sniper como o Rachador não
    precisa vencer igual contra todo mundo). Definir um alvo por confronto é
    uma extensão futura — fora do escopo desta etapa, que só quer provar o
    laço "medir -> ajustar -> medir de novo" funcionando.

    `apenas_espelhos=True` restringe a média aos 6 confrontos "classe contra
    si mesma" em vez dos 21 pares completos — ver o motivo no docstring de
    `matriz_de_confrontos`.

    `atributos_por_classe` (opcional) é repassado direto pra
    `matriz_de_confrontos` — usado pelo `ajuste_de_classes.ajustar_atributos`
    (Camada 1, parte 3) pra medir o erro da matriz completa com os atributos
    por classe que ele está ajustando, em vez do `atributos_base` fixo.
    """
    matriz = matriz_de_confrontos(
        classes,
        n=n_simulacoes,
        seed=seed,
        parametros=parametros,
        apenas_espelhos=apenas_espelhos,
        atributos_por_classe=atributos_por_classe,
    )
    erros = [resultado.erro(alvo=0.5) for resultado in matriz.values()]
    return sum(erros) / len(erros)


def _estimar_derivada(
    nome_parametro: str,
    parametros: ParametrosDeFormula,
    classes: tuple[Classe, ...],
    n_simulacoes: int,
    seed: int,
    delta: float,
    apenas_espelhos: bool,
) -> float:
    """Inclinação da reta secante de `erro_medio` em torno do valor atual do
    parâmetro — a aproximação de `d(erro)/d(parametro)` por diferença central.

    Se o valor menos `delta` ficaria negativo (alguns coeficientes não podem
    ser negativos, ex: `reducao_dano_por_or`), o ponto baixo é limitado em 0 e
    o denominador usa o espaçamento real entre os dois pontos — ainda é uma
    reta secante válida, só não fica perfeitamente centrada nesse caso de
    borda.
    """
    valor_atual = getattr(parametros, nome_parametro)
    valor_baixo = max(0.0, valor_atual - delta)
    valor_alto = valor_atual + delta

    parametros_baixo = replace(parametros, **{nome_parametro: valor_baixo})
    parametros_alto = replace(parametros, **{nome_parametro: valor_alto})

    erro_baixo = erro_medio(parametros_baixo, classes, n_simulacoes, seed, apenas_espelhos)
    erro_alto = erro_medio(parametros_alto, classes, n_simulacoes, seed, apenas_espelhos)

    return (erro_alto - erro_baixo) / (valor_alto - valor_baixo)


def ajustar_parametros(
    parametros_iniciais: ParametrosDeFormula,
    classes: tuple[Classe, ...] = TODAS_AS_CLASSES,
    n_simulacoes: int = 400,
    max_iteracoes: int = 20,
    delta: float = 0.3,
    taxa_de_aprendizado: float = 0.3,
    seed: int = 0,
    apenas_espelhos: bool = False,
) -> tuple[ParametrosDeFormula, list[float], list[PassoDeAjuste]]:
    """Roda o auto-tuner e devolve os parâmetros finais, a curva de convergência
    (erro médio ao fim de cada iteração — o que vira o gráfico de convergência)
    e o histórico detalhado de cada ajuste individual.

    `n_simulacoes` e `delta` maiores que o primeiro rascunho (200 e 0.05) de
    propósito: o ruído de uma proporção estimada por Monte Carlo cai com
    1/sqrt(n), e um `delta` maior faz o sinal real (a variação do erro)
    dominar sobre esse ruído na hora de estimar a derivada — ver a discussão
    de erro de truncamento vs. ruído no docstring do módulo.

    `apenas_espelhos=True` mede o erro só sobre os confrontos "classe contra
    si mesma" (6, em vez dos 21 pares completos). É uma correção pro seguinte
    problema: como os coeficientes são globais (compartilhados por todas as
    classes), o gradiente médio sobre os 21 pares soma derivadas que às vezes
    apontam em direções opostas (melhorar um confronto piora outro) e quase se
    cancelam — um platô na função objetivo, não ruído de medição. Restringir
    aos espelhos dá um sinal mais coerente, ainda que deixe o desequilíbrio
    *entre* classes diferentes fora do que o auto-tuner enxerga.

    Pra cada parâmetro, dentro de uma iteração, os dois pontos da diferença
    central usam o MESMO seed (common random numbers) — assim os dois Monte
    Carlo comparam os mesmos combates simulados, e a diferença entre eles
    reflete o parâmetro mudando, não sorte do RNG.
    """
    parametros = parametros_iniciais
    historico_de_erro: list[float] = []
    passos: list[PassoDeAjuste] = []

    for iteracao in range(max_iteracoes):
        seed_da_iteracao = seed + iteracao

        for nome_parametro in PARAMETROS_AJUSTAVEIS:
            valor_atual = getattr(parametros, nome_parametro)

            derivada = _estimar_derivada(
                nome_parametro, parametros, classes, n_simulacoes, seed_da_iteracao, delta, apenas_espelhos
            )
            novo_valor = max(0.0, valor_atual - taxa_de_aprendizado * derivada)

            parametros = replace(parametros, **{nome_parametro: novo_valor})
            passos.append(PassoDeAjuste(iteracao, nome_parametro, valor_atual, novo_valor, derivada))

        erro_da_iteracao = erro_medio(parametros, classes, n_simulacoes, seed_da_iteracao, apenas_espelhos)
        historico_de_erro.append(erro_da_iteracao)

    return parametros, historico_de_erro, passos

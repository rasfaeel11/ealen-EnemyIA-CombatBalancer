"""Camada 1 (parte 1 de 3): Monte Carlo — mede a taxa de vitória real de um confronto.

Este módulo só mede o desequilíbrio atual: roda N combates (reaproveitando a
Camada 0) e conta quem venceu. Ele ainda NÃO ajusta nada — os auto-tuners via
gradiente descendente que leem esse erro e corrigem os parâmetros/atributos
são os próximos módulos, separados, porque "medir o problema" e "corrigir o
problema" são responsabilidades diferentes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ealen_ia.atributos import Atributos, ParametrosDeFormula
from ealen_ia.classes import Classe, TODAS_AS_CLASSES
from ealen_ia.combate import simulate_combat
from ealen_ia.personagem import Personagem


@dataclass(frozen=True)
class ResultadoConfronto:
    classe_a: str
    classe_b: str
    n: int
    vitorias_a: int
    vitorias_b: int
    empates: int

    @property
    def taxa_vitoria_a(self) -> float:
        return self.vitorias_a / self.n

    def erro(self, alvo: float = 0.5) -> float:
        """|taxa_de_vitoria - alvo| — a métrica de desequilíbrio do AGENTS.md.

        `alvo=0.5` faz sentido pra confrontos que deveriam ser simétricos
        (ex: uma classe contra si mesma). Pra confrontos assimétricos por
        design, quem chama isso passa outro alvo.
        """
        return abs(self.taxa_vitoria_a - alvo)


def rodar_confronto(a: Personagem, b: Personagem, n: int, seed: int | None = None) -> ResultadoConfronto:
    """Roda `n` combates entre `a` e `b` e conta vitórias/empates.

    Cada uma das `n` repetições usa um seed diferente (derivado de `seed`,
    somando o índice da rodada) — senão as `n` repetições seriam todas
    idênticas, já que `simulate_combat` é determinístico dado o mesmo seed.
    Se `seed` for None, cada rodada é estocástica (comportamento normal do
    Monte Carlo).
    """
    vitorias_a = vitorias_b = empates = 0
    for i in range(n):
        seed_da_rodada = None if seed is None else seed + i
        resultado = simulate_combat(a, b, seed=seed_da_rodada)
        if resultado.vencedor == a.nome:
            vitorias_a += 1
        elif resultado.vencedor == b.nome:
            vitorias_b += 1
        else:
            empates += 1
    return ResultadoConfronto(a.nome, b.nome, n, vitorias_a, vitorias_b, empates)


def _rodar_confronto_entre_classes(
    classe_a: Classe,
    classe_b: Classe,
    n: int,
    seed: int | None,
    parametros: ParametrosDeFormula | None,
    atributos_por_classe: dict[str, Atributos] | None,
) -> ResultadoConfronto:
    """Monta os dois `Personagem` (com nomes distintos em caso de espelho) e
    roda o confronto — a parte comum de `matriz_de_confrontos` e
    `matriz_de_confrontos_da_classe`.

    `atributos_por_classe` permite testar atributos diferentes do
    `atributos_base` fixo de cada `Classe` (usado pelo ajuste de atributos por
    classe, Camada 1 parte 3) — quando `None` ou quando a classe não está no
    dict, cai no `atributos_base` de sempre.
    """
    # Em espelho (classe_a == classe_b) os dois nomes de Personagem não podem
    # ser iguais: rodar_confronto decide o vencedor comparando
    # `resultado.vencedor == a.nome`, e um nome duplicado faria a vitória
    # cair sempre no lado "a" mesmo quando "b" venceu.
    nome_a = f"{classe_a.nome} (1)" if classe_a is classe_b else classe_a.nome
    nome_b = f"{classe_b.nome} (2)" if classe_a is classe_b else classe_b.nome

    atributos_a = (atributos_por_classe or {}).get(classe_a.nome, classe_a.atributos_base)
    atributos_b = (atributos_por_classe or {}).get(classe_b.nome, classe_b.atributos_base)

    a = Personagem(nome_a, classe_a, atributos_a, parametros or ParametrosDeFormula())
    b = Personagem(nome_b, classe_b, atributos_b, parametros or ParametrosDeFormula())

    resultado = rodar_confronto(a, b, n, seed=seed)
    return ResultadoConfronto(
        classe_a=classe_a.nome,
        classe_b=classe_b.nome,
        n=resultado.n,
        vitorias_a=resultado.vitorias_a,
        vitorias_b=resultado.vitorias_b,
        empates=resultado.empates,
    )


def matriz_de_confrontos(
    classes: tuple[Classe, ...] = TODAS_AS_CLASSES,
    n: int = 1000,
    seed: int | None = None,
    parametros: ParametrosDeFormula | None = None,
    apenas_espelhos: bool = False,
    atributos_por_classe: dict[str, Atributos] | None = None,
) -> dict[tuple[str, str], ResultadoConfronto]:
    """Roda o confronto de cada par de classes (todas contra todas), incluindo
    cada classe contra si mesma (espelho).

    Cada par aparece uma única vez na matriz (ex: só "Guardião vs Rachador",
    não repetido como "Rachador vs Guardião") — como `a` sempre ataca primeiro
    em `simulate_combat`, simular os dois sentidos seria útil só se quiséssemos
    isolar a vantagem de iniciativa, o que não é o objetivo desta etapa.

    `parametros` é o mesmo `ParametrosDeFormula` pra todas as classes — é esse
    conjunto de coeficientes que o auto-tuner de parâmetros (Camada 1, parte
    2) ajusta e testa aqui.

    `atributos_por_classe` (opcional) sobrepõe o `atributos_base` de uma ou
    mais classes — é o que o ajuste de atributos por classe (Camada 1, parte
    3) usa pra testar Dain/Eir/Nath/Il/Or diferentes por classe, sem tocar em
    `ParametrosDeFormula` nem nas constantes de `classes.py`. Deixado de fora
    (`None`, o default), o comportamento é idêntico ao original: cada
    `Personagem` usa o `atributos_base` da sua própria classe.

    `apenas_espelhos=True` pula os confrontos entre classes diferentes e roda
    só cada classe contra si mesma — usado pelo auto-tuner de parâmetros pra
    medir o erro sobre um objetivo mais coerente (os 21 confrontos da matriz
    completa têm desequilíbrios que às vezes puxam os coeficientes globais em
    direções opostas; os 6 espelhos deveriam ser ~50/50 de forma menos
    ambígua).
    """
    resultados: dict[tuple[str, str], ResultadoConfronto] = {}
    for i, classe_a in enumerate(classes):
        for classe_b in classes[i:]:
            if apenas_espelhos and classe_a is not classe_b:
                continue
            resultados[(classe_a.nome, classe_b.nome)] = _rodar_confronto_entre_classes(
                classe_a, classe_b, n, seed, parametros, atributos_por_classe
            )
    return resultados


def matriz_de_confrontos_da_classe(
    nome_classe_alvo: str,
    classes: tuple[Classe, ...] = TODAS_AS_CLASSES,
    n: int = 1000,
    seed: int | None = None,
    parametros: ParametrosDeFormula | None = None,
    atributos_por_classe: dict[str, Atributos] | None = None,
) -> dict[tuple[str, str], ResultadoConfronto]:
    """Só os confrontos que envolvem `nome_classe_alvo`: o espelho dela + um
    confronto cruzado contra cada outra classe (`len(classes)` confrontos, não
    todos os `len(classes) * (len(classes) + 1) / 2` pares).

    Usada pelo ajuste de atributos por classe (Camada 1, parte 3): perturbar
    o Dain do Guardião só afeta os confrontos do Guardião, então medir a
    derivada rodando a matriz inteira seria desperdício (e ruído extra de
    confrontos irrelevantes entrando na média).

    A chave de cada par segue a mesma convenção de ordem que
    `matriz_de_confrontos` usaria pro mesmo par (índice na tupla `classes`
    decide quem é `classe_a`, ou seja, quem ataca primeiro) — assim os
    resultados daqui são diretamente comparáveis ao subconjunto correspondente
    da matriz completa.
    """
    classe_alvo = next(c for c in classes if c.nome == nome_classe_alvo)
    indice_alvo = classes.index(classe_alvo)

    resultados: dict[tuple[str, str], ResultadoConfronto] = {}
    for indice_outra, outra in enumerate(classes):
        classe_a, classe_b = (classe_alvo, outra) if indice_alvo <= indice_outra else (outra, classe_alvo)
        resultados[(classe_a.nome, classe_b.nome)] = _rodar_confronto_entre_classes(
            classe_a, classe_b, n, seed, parametros, atributos_por_classe
        )
    return resultados

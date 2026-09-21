"""Camada 2 (parte 2 de 2): agente de Q-learning tabular, treinado por
self-play sobre `combate_decisorio` (Camada 2, parte 1).

Ideia central (equação de Bellman): pra cada par (estado, ação), a Tabela Q
guarda uma estimativa de "quão bom é tomar essa ação nesse estado, contando
a recompensa imediata + o melhor que dá pra fazer dali em diante". A cada
transição observada (s, a, r, s'), a atualização é:

    Q(s,a) <- Q(s,a) + alfa * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

- alfa (taxa de aprendizado): o quanto confiar na observação nova vs. no que
  a tabela já "sabia".
- gamma (fator de desconto): o quanto recompensas futuras valem comparadas à
  recompensa imediata (gamma perto de 1 = agente "pensa a longo prazo").
- max_a' Q(s',a'): o valor do melhor próximo passo, assumindo que dali em
  diante o agente joga bem. É essa parte que propaga recompensas de
  vitória/derrota (que só existem no FIM do combate) pra trás, pros estados
  que levaram até lá — sem isso, cada estado só aprenderia com o dano daquele
  turno específico, nunca com o resultado final do combate.

Self-play: os dois lados de cada combate usam a MESMA tabela Q — o estado já
é definido em termos relativos (minha faixa de HP, a do oponente, quem tem
guarda), então a tabela não precisa saber "de que lado" está jogando, e
aprende uma política única que serve pra qualquer combate.

Exploração via epsilon-greedy: com probabilidade `epsilon`, escolhe uma ação
aleatória (explorar o que ainda não se sabe se é bom); senão, a de maior Q
conhecido (aproveitar o que já aprendeu). `epsilon` decai ao longo do
treino — explora muito no início (a tabela ainda não sabe nada), converge
pra quase-gulosa no fim.

Por que medir progresso contra uma política FIXA, não só self-play: em
self-play puro, a taxa de vitória de "quem age primeiro" tende a ficar perto
de 50% o tempo todo (é o mesmo agente jogando contra si mesmo, nos dois
lados) — não prova que ele está melhorando, só que ele empata consigo mesmo.
Pra ver progresso de verdade, `avaliar_taxa_vitoria` roda o agente (sem
exploração) contra uma política simples e fixa (`politica_sempre_ataca`) e
mede se ele aprendeu a ganhar mais que uma política ingênua.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace

from ealen_ia.combate_decisorio import (
    ACOES,
    Acao,
    Estado,
    Politica,
    TransicaoDecisoria,
    simulate_combat_decisorio,
)
from ealen_ia.personagem import Personagem

RECOMPENSA_VITORIA = 1.0
RECOMPENSA_DERROTA = -1.0
RECOMPENSA_EMPATE = -0.1

TabelaQ = defaultdict[tuple[Estado, Acao], float]


def criar_tabela_q() -> TabelaQ:
    """`defaultdict(float)`: um (estado, ação) nunca visto vale 0.0 — sem
    isso, toda leitura da tabela precisaria tratar o caso "ainda não vi esse
    estado" à parte."""
    return defaultdict(float)


def valor_maximo(tabela_q: TabelaQ, estado: Estado) -> float:
    return max(tabela_q[(estado, acao)] for acao in ACOES)


def melhor_acao(tabela_q: TabelaQ, estado: Estado) -> Acao:
    return max(ACOES, key=lambda acao: tabela_q[(estado, acao)])


def politica_epsilon_gulosa(tabela_q: TabelaQ, epsilon: float, rng: random.Random) -> Politica:
    def politica(estado: Estado) -> Acao:
        if rng.random() < epsilon:
            return rng.choice(ACOES)
        return melhor_acao(tabela_q, estado)

    return politica


def politica_gulosa(tabela_q: TabelaQ) -> Politica:
    """Sem exploração — a política "final", usada pra avaliar o agente já
    treinado (durante o treino, quem explora é `politica_epsilon_gulosa`)."""

    def politica(estado: Estado) -> Acao:
        return melhor_acao(tabela_q, estado)

    return politica


def politica_sempre_ataca(estado: Estado) -> Acao:
    """Baseline ingênua (a mesma coisa que a Camada 0 sempre fez) — serve de
    referência fixa pra medir se o agente aprendeu algo melhor que "só
    atacar"."""
    return Acao.ATACAR


def atualizar_q(tabela_q: TabelaQ, transicao: TransicaoDecisoria, alfa: float, gamma: float, terminal: bool) -> None:
    """Um passo da equação de Bellman. `terminal=True` quando essa é a
    última transição de um combate (vitória/derrota/empate): nesse caso não
    existe "próximo estado" de verdade pra olhar (o combate acabou), então o
    alvo é só a recompensa — sem o termo `gamma * max Q(s')`."""
    alvo = transicao.recompensa
    if not terminal:
        alvo += gamma * valor_maximo(tabela_q, transicao.proximo_estado)
    chave = (transicao.estado, transicao.acao)
    tabela_q[chave] += alfa * (alvo - tabela_q[chave])


def avaliar_taxa_vitoria(
    tabela_q: TabelaQ,
    a: Personagem,
    b: Personagem,
    politica_oponente: Politica = politica_sempre_ataca,
    n_combates: int = 50,
    seed: int = 0,
    max_turnos: int = 100,
) -> float:
    """Roda `n_combates` combates do agente (política gulosa da tabela Q, sem
    exploração) contra `politica_oponente`, e devolve a taxa de vitória do
    agente. Usada tanto pra acompanhar o progresso durante o treino quanto
    pra validar o agente treinado no fim ("Definição de pronto" da Camada 2
    no AGENTS.md: "taxa de vitória registrada")."""
    politica_agente = politica_gulosa(tabela_q)
    vitorias = 0
    for i in range(n_combates):
        resultado = simulate_combat_decisorio(a, b, politica_agente, politica_oponente, seed=seed + i, max_turnos=max_turnos)
        if resultado.vencedor == a.nome:
            vitorias += 1
    return vitorias / n_combates


def treinar(
    pares_de_treino: list[tuple[Personagem, Personagem]],
    n_episodios: int = 20000,
    alfa: float = 0.1,
    gamma: float = 0.95,
    epsilon_inicial: float = 1.0,
    epsilon_final: float = 0.05,
    max_turnos: int = 100,
    par_de_avaliacao: tuple[Personagem, Personagem] | None = None,
    n_combates_avaliacao: int = 30,
    n_checkpoints: int = 40,
    seed: int = 0,
) -> tuple[TabelaQ, list[float]]:
    """Roda `n_episodios` combates de self-play (par sorteado de
    `pares_de_treino` a cada episódio) e atualiza a mesma tabela Q a cada
    transição. Devolve a tabela final e a curva de progresso: a taxa de
    vitória do agente contra `politica_sempre_ataca`, medida a cada
    `n_episodios / n_checkpoints` episódios — o equivalente, pra RL, do
    gráfico de convergência da Camada 1 (aqui "convergir" significa "ficar
    bom contra uma política fixa", não "chegar a um erro-alvo").
    """
    tabela_q = criar_tabela_q()
    rng = random.Random(seed)
    par_de_avaliacao = par_de_avaliacao or pares_de_treino[0]
    registro_a_cada = max(1, n_episodios // n_checkpoints)
    historico_taxa_vitoria: list[float] = []

    for episodio in range(n_episodios):
        progresso = episodio / max(1, n_episodios - 1)
        epsilon = epsilon_inicial + (epsilon_final - epsilon_inicial) * progresso

        a, b = pares_de_treino[rng.randrange(len(pares_de_treino))]
        politica_a = politica_epsilon_gulosa(tabela_q, epsilon, rng)
        politica_b = politica_epsilon_gulosa(tabela_q, epsilon, rng)
        resultado = simulate_combat_decisorio(
            a, b, politica_a, politica_b, seed=rng.randrange(2**30), max_turnos=max_turnos
        )

        ultima_transicao_de: dict[str, TransicaoDecisoria] = {}
        for transicao in resultado.historico:
            ultima_transicao_de[transicao.quem_agiu] = transicao

        for transicao in resultado.historico:
            if ultima_transicao_de[transicao.quem_agiu] is transicao:
                continue  # a última transição de cada lado recebe o tratamento terminal abaixo, não este
            atualizar_q(tabela_q, transicao, alfa, gamma, terminal=False)

        for nome_personagem, transicao in ultima_transicao_de.items():
            if resultado.vencedor == nome_personagem:
                bonus = RECOMPENSA_VITORIA
            elif resultado.vencedor is not None:
                bonus = RECOMPENSA_DERROTA
            else:
                bonus = RECOMPENSA_EMPATE
            transicao_terminal = replace(transicao, recompensa=transicao.recompensa + bonus)
            atualizar_q(tabela_q, transicao_terminal, alfa, gamma, terminal=True)

        if (episodio + 1) % registro_a_cada == 0:
            taxa = avaliar_taxa_vitoria(
                tabela_q, *par_de_avaliacao, n_combates=n_combates_avaliacao, seed=seed + episodio, max_turnos=max_turnos
            )
            historico_taxa_vitoria.append(taxa)

    return tabela_q, historico_taxa_vitoria

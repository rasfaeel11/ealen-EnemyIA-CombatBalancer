"""Camada 2 (extensão fora do escopo inicial, implementada a pedido): Q-learning
com aproximação LINEAR de função de valor, em vez de tabela (`q_learning.py`).

Por que isso é "gradiente descendente de verdade" (é assim que o AGENTS.md
descreve essa extensão): na tabela Q, cada (estado, ação) guarda seu próprio
número independente — atualizar um não afeta os outros. Aqui, Q(s,a) é
aproximado por uma combinação linear de features do estado:

    Q(s,a) ≈ w_a · φ(s) = w_a[0]*φ(s)[0] + w_a[1]*φ(s)[1] + ...

`w_a` é o vetor de pesos DAQUELA ação (um vetor por ação, 3 no total), e
`φ(s)` é o vetor de features do estado. A atualização de cada peso é a
derivada da perda quadrática de TD-error em relação a ele:

    perda(w_a) = (alvo - w_a · φ(s))²
    d(perda)/d(w_a[i]) = -2 * (alvo - w_a·φ(s)) * φ(s)[i]

... e o passo de gradiente descendente (a constante -2 absorvida em `alfa`):

    w_a[i] <- w_a[i] + alfa * (alvo - w_a·φ(s)) * φ(s)[i]

É a MESMA lógica de "andar contra o gradiente" que a Camada 1 usa pros
coeficientes de fórmula — só que aqui a derivada é ANALÍTICA (é a própria
feature, `φ(s)[i]`, não precisa de diferença finita) porque o modelo é
linear: dQ/dw_a[i] = φ(s)[i] exatamente, sem aproximação nenhuma.

Trade-off em relação à tabela: um modelo linear generaliza entre estados
parecidos (o peso de "minha faixa de HP" é compartilhado por todos os
estados, não reaprendido do zero pra cada combinação), ao custo de perder a
precisão "um valor exato por estado" da tabela — se a relação real entre
estado e valor não for bem aproximada por um plano, o modelo linear erra
sistematicamente onde a tabela não erraria.

Achado real ao testar (por isso o AGENTS.md descreve essa extensão como
"deliberadamente adiada"): ao contrário da tabela — que tem garantia de
convergência sob as condições padrão de Q-learning tabular —, esse agente
linear é INSTÁVEL. Rodando o mesmo treino com seeds diferentes (mesmos
hiperparâmetros), a taxa de vitória final variou de ~0% a ~90% em testes
empíricos. Isso é a chamada "tríade mortal" do aprendizado por reforço:
bootstrap (o alvo usa a própria estimativa de Q) + o operador `max` (sempre
otimista sobre o próximo estado) + aproximação de função (os pesos
influenciam MUITOS estados de uma vez, então um ajuste pra corrigir um
estado pode desajustar outros) — juntas, essas três coisas não têm garantia
de convergência, podendo oscilar ou estacionar num ótimo local ruim (ex:
"sempre defender", que evita a atualização de erro furiosa de atacar errado
mas perde por inanição). Tentar mitigar com features de interação
(HP_próprio × HP_oponente) não resolveu — piorou, na verdade — confirmando
que o problema é estrutural do método, não falta de expressividade do
modelo. `treinar_linear` deve ser avaliado em MÉDIA sobre várias sementes,
nunca por uma única execução.
"""

from __future__ import annotations

import random
from dataclasses import replace

from ealen_ia.combate_decisorio import ACOES, Acao, Estado, Politica, TransicaoDecisoria, simulate_combat_decisorio
from ealen_ia.personagem import Personagem
from ealen_ia.q_learning import RECOMPENSA_DERROTA, RECOMPENSA_EMPATE, RECOMPENSA_VITORIA

# bias, minha faixa de HP (normalizada 0..1), faixa de HP do oponente, eu com
# guarda, oponente com guarda.
N_FEATURES = 5

PesosLineares = dict[Acao, list[float]]


def features(estado: Estado) -> list[float]:
    return [
        1.0,
        estado.hp_proprio_faixa / 4,
        estado.hp_oponente_faixa / 4,
        1.0 if estado.eu_com_guarda else 0.0,
        1.0 if estado.oponente_com_guarda else 0.0,
    ]


def criar_pesos() -> PesosLineares:
    return {acao: [0.0] * N_FEATURES for acao in ACOES}


def valor_aproximado(pesos: PesosLineares, estado: Estado, acao: Acao) -> float:
    return sum(w * f for w, f in zip(pesos[acao], features(estado)))


def valor_maximo_aproximado(pesos: PesosLineares, estado: Estado) -> float:
    return max(valor_aproximado(pesos, estado, acao) for acao in ACOES)


def melhor_acao_aproximada(pesos: PesosLineares, estado: Estado) -> Acao:
    return max(ACOES, key=lambda acao: valor_aproximado(pesos, estado, acao))


def politica_epsilon_gulosa_linear(pesos: PesosLineares, epsilon: float, rng: random.Random) -> Politica:
    def politica(estado: Estado) -> Acao:
        if rng.random() < epsilon:
            return rng.choice(ACOES)
        return melhor_acao_aproximada(pesos, estado)

    return politica


def politica_gulosa_linear(pesos: PesosLineares) -> Politica:
    def politica(estado: Estado) -> Acao:
        return melhor_acao_aproximada(pesos, estado)

    return politica


def atualizar_pesos(pesos: PesosLineares, transicao: TransicaoDecisoria, alfa: float, gamma: float, terminal: bool) -> None:
    """Um passo de gradiente descendente sobre os pesos da ação tomada nessa
    transição — ver a derivação completa no docstring do módulo."""
    alvo = transicao.recompensa
    if not terminal:
        alvo += gamma * valor_maximo_aproximado(pesos, transicao.proximo_estado)

    erro_td = alvo - valor_aproximado(pesos, transicao.estado, transicao.acao)
    phi = features(transicao.estado)

    pesos_da_acao = pesos[transicao.acao]
    for i in range(N_FEATURES):
        pesos_da_acao[i] += alfa * erro_td * phi[i]


def avaliar_taxa_vitoria_linear(
    pesos: PesosLineares,
    a: Personagem,
    b: Personagem,
    politica_oponente,
    n_combates: int = 50,
    seed: int = 0,
    max_turnos: int = 100,
) -> float:
    politica_agente = politica_gulosa_linear(pesos)
    vitorias = 0
    for i in range(n_combates):
        resultado = simulate_combat_decisorio(a, b, politica_agente, politica_oponente, seed=seed + i, max_turnos=max_turnos)
        if resultado.vencedor == a.nome:
            vitorias += 1
    return vitorias / n_combates


def treinar_linear(
    pares_de_treino: list[tuple[Personagem, Personagem]],
    n_episodios: int = 30000,
    alfa: float = 0.02,
    gamma: float = 0.95,
    epsilon_inicial: float = 1.0,
    epsilon_final: float = 0.05,
    max_turnos: int = 100,
    par_de_avaliacao: tuple[Personagem, Personagem] | None = None,
    n_combates_avaliacao: int = 30,
    n_checkpoints: int = 40,
    seed: int = 0,
) -> tuple[PesosLineares, list[float]]:
    """Mesmo laço de treino de `q_learning.treinar` (self-play, epsilon
    decaindo, avaliação periódica contra `politica_sempre_ataca`), trocando
    a tabela Q pelos pesos lineares. Ver `q_learning.treinar` pra a versão
    tabular comentada em detalhe — a estrutura do laço é a mesma de
    propósito, pra comparar as duas abordagens lado a lado."""
    from ealen_ia.q_learning import politica_sempre_ataca  # import local pra evitar ciclo de import

    pesos = criar_pesos()
    rng = random.Random(seed)
    par_de_avaliacao = par_de_avaliacao or pares_de_treino[0]
    registro_a_cada = max(1, n_episodios // n_checkpoints)
    historico_taxa_vitoria: list[float] = []

    for episodio in range(n_episodios):
        progresso = episodio / max(1, n_episodios - 1)
        epsilon = epsilon_inicial + (epsilon_final - epsilon_inicial) * progresso

        a, b = pares_de_treino[rng.randrange(len(pares_de_treino))]
        politica_a = politica_epsilon_gulosa_linear(pesos, epsilon, rng)
        politica_b = politica_epsilon_gulosa_linear(pesos, epsilon, rng)
        resultado = simulate_combat_decisorio(
            a, b, politica_a, politica_b, seed=rng.randrange(2**30), max_turnos=max_turnos
        )

        ultima_transicao_de: dict[str, TransicaoDecisoria] = {}
        for transicao in resultado.historico:
            ultima_transicao_de[transicao.quem_agiu] = transicao

        for transicao in resultado.historico:
            if ultima_transicao_de[transicao.quem_agiu] is transicao:
                continue
            atualizar_pesos(pesos, transicao, alfa, gamma, terminal=False)

        for nome_personagem, transicao in ultima_transicao_de.items():
            if resultado.vencedor == nome_personagem:
                bonus = RECOMPENSA_VITORIA
            elif resultado.vencedor is not None:
                bonus = RECOMPENSA_DERROTA
            else:
                bonus = RECOMPENSA_EMPATE
            transicao_terminal = replace(transicao, recompensa=transicao.recompensa + bonus)
            atualizar_pesos(pesos, transicao_terminal, alfa, gamma, terminal=True)

        if (episodio + 1) % registro_a_cada == 0:
            taxa = avaliar_taxa_vitoria_linear(
                pesos, *par_de_avaliacao, politica_sempre_ataca, n_combates=n_combates_avaliacao, seed=seed + episodio, max_turnos=max_turnos
            )
            historico_taxa_vitoria.append(taxa)

    return pesos, historico_taxa_vitoria

"""Camada 2 (parte 1 de 2): combate com decisão.

A mesma base de fórmulas da Camada 0 (`atributos.py`, `Personagem`), mas em
vez de "atacar automaticamente todo turno" (o que `combate.py` faz, e
continua fazendo — este módulo não o substitui, a Camada 1 depende dele como
está), cada turno pede uma ação a uma **política**: `Atacar`, `Defender` ou
usar `Habilidade`. Essa escolha real, com trade-offs, é o que dá ao
Q-learning (Camada 2, parte 2) algo pra aprender — sem pontos de decisão não
tem o que otimizar.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from ealen_ia.personagem import Personagem

MAX_TURNOS_PADRAO = 100

# Trade-off da Habilidade: mais dano, mas menos precisão (risco vs. recompensa).
BONUS_DANO_HABILIDADE = 1.8
PENALIDADE_ACERTO_HABILIDADE = 0.20

# Defender não causa dano, mas reduz o próximo golpe recebido pela metade.
REDUCAO_DANO_AO_DEFENDER = 0.5

N_FAIXAS_DE_HP = 5


class Acao(Enum):
    ATACAR = "atacar"
    DEFENDER = "defender"
    HABILIDADE = "habilidade"


ACOES: tuple[Acao, ...] = (Acao.ATACAR, Acao.DEFENDER, Acao.HABILIDADE)


@dataclass(frozen=True)
class Estado:
    """O que quem está decidindo observa: HP relativo (o próprio e o do
    oponente, discretizado em faixas) e se alguém está com a guarda
    levantada (efeito de "defender" pendente).

    Discretizado de propósito: Q-learning tabular precisa de um número finito
    de estados pra caber numa tabela; HP contínuo geraria infinitos estados.
    Não inclui o número do turno nem os atributos brutos das classes — o
    objetivo é uma política que generalize entre builds diferentes (ver
    "Definição de pronto" da Camada 2 no AGENTS.md), então o estado só
    guarda informação relativa (frações de HP), nunca valores absolutos de
    uma classe específica.
    """

    hp_proprio_faixa: int
    hp_oponente_faixa: int
    eu_com_guarda: bool
    oponente_com_guarda: bool


def _faixa_de_hp(hp_atual: float, hp_maximo: float, n_faixas: int = N_FAIXAS_DE_HP) -> int:
    fracao = max(0.0, min(1.0, hp_atual / hp_maximo))
    faixa = int(fracao * n_faixas)
    return min(faixa, n_faixas - 1)


def observar_estado(proprio: Personagem, oponente: Personagem, proprio_com_guarda: bool, oponente_com_guarda: bool) -> Estado:
    return Estado(
        hp_proprio_faixa=_faixa_de_hp(proprio.hp_atual, proprio.hp_maximo),
        hp_oponente_faixa=_faixa_de_hp(oponente.hp_atual, oponente.hp_maximo),
        eu_com_guarda=proprio_com_guarda,
        oponente_com_guarda=oponente_com_guarda,
    )


Politica = Callable[[Estado], Acao]


@dataclass(frozen=True)
class TransicaoDecisoria:
    """Um passo de (estado, ação, recompensa, próximo estado) — o que o
    Q-learning precisa pra aplicar a equação de Bellman (ver `q_learning.py`).
    `quem_agiu` é o nome do `Personagem` que tomou a decisão nesse turno.
    """

    quem_agiu: str
    estado: Estado
    acao: Acao
    recompensa: float
    proximo_estado: Estado


@dataclass(frozen=True)
class ResultadoCombateDecisorio:
    vencedor: str | None
    historico: list[TransicaoDecisoria] = field(default_factory=list)

    @property
    def turnos_totais(self) -> int:
        return len(self.historico)


def _dano_de_ataque(personagem: Personagem) -> float:
    return max(personagem.dano_fisico_base, personagem.dano_magico_base)


def _resolver_ataque(acao: Acao, atacante: Personagem, alvo: Personagem, alvo_com_guarda: bool, rng: random.Random) -> float:
    """Resolve `Acao.ATACAR` ou `Acao.HABILIDADE` (nunca `Acao.DEFENDER`,
    que não causa dano e é tratado direto no loop principal). Devolve o dano
    efetivamente causado (0 se errou)."""
    chance_de_acerto = atacante.chance_de_acerto
    dano_bruto = _dano_de_ataque(atacante)

    if acao is Acao.HABILIDADE:
        chance_de_acerto = max(0.05, chance_de_acerto - PENALIDADE_ACERTO_HABILIDADE)
        dano_bruto *= BONUS_DANO_HABILIDADE

    if rng.random() >= chance_de_acerto:
        return 0.0

    if rng.random() < atacante.chance_de_critico:
        dano_bruto *= atacante.parametros.multiplicador_critico

    if alvo_com_guarda:
        dano_bruto *= 1 - REDUCAO_DANO_AO_DEFENDER

    return alvo.receber_dano(dano_bruto)


def simulate_combat_decisorio(
    a: Personagem,
    b: Personagem,
    politica_a: Politica,
    politica_b: Politica,
    seed: int | None = None,
    max_turnos: int = MAX_TURNOS_PADRAO,
) -> ResultadoCombateDecisorio:
    """Roda um combate 1x1 completo em que cada turno pede uma ação à
    política do personagem que está agindo (`a` sempre age primeiro, mesma
    convenção da Camada 0).

    A guarda de "defender" dura até a próxima vez que esse personagem for
    alvo de uma ação ofensiva do oponente (acertando ou errando) — não
    expira por tempo, então se os dois ficarem só defendendo, ambas as
    guardas continuam ativas indefinidamente (sem dano trocado, o combate
    empata em `max_turnos`).
    """
    a.hp_atual = a.hp_maximo
    b.hp_atual = b.hp_maximo
    rng = random.Random(seed)

    guarda: dict[str, bool] = {a.nome: False, b.nome: False}
    historico: list[TransicaoDecisoria] = []
    ordem = (a, b)

    for numero_turno in range(1, max_turnos + 1):
        atacante, alvo = ordem[(numero_turno - 1) % 2], ordem[numero_turno % 2]

        estado_antes = observar_estado(atacante, alvo, guarda[atacante.nome], guarda[alvo.nome])
        acao = (politica_a if atacante is a else politica_b)(estado_antes)

        if acao is Acao.DEFENDER:
            guarda[atacante.nome] = True
            dano_causado = 0.0
        else:
            alvo_com_guarda = guarda[alvo.nome]
            guarda[alvo.nome] = False  # consumida por ser alvo de uma ação ofensiva, acertando ou não
            dano_causado = _resolver_ataque(acao, atacante, alvo, alvo_com_guarda, rng)

        recompensa = dano_causado / alvo.hp_maximo  # shaping normalizado (0..~) — o bônus de vitória/derrota é aplicado depois, pelo treino (ver q_learning.py)
        estado_depois = observar_estado(atacante, alvo, guarda[atacante.nome], guarda[alvo.nome])
        historico.append(TransicaoDecisoria(atacante.nome, estado_antes, acao, recompensa, estado_depois))

        if not alvo.esta_vivo:
            return ResultadoCombateDecisorio(vencedor=atacante.nome, historico=historico)

    return ResultadoCombateDecisorio(vencedor=None, historico=historico)

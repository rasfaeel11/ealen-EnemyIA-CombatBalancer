"""Camada 0: simulador de combate 1x1, turno a turno.

Regras simplificadas de propósito (ver AGENTS.md — "non-goals"):
    - Sem iniciativa/velocidade: os personagens se alternam em ordem fixa,
      `a` sempre ataca primeiro.
    - Sem escolha de tipo de dano: cada ataque usa o maior entre o dano
      físico e o mágico do atacante (o "forte" de cada classe já emerge
      dos próprios atributos, sem precisar de uma regra extra por classe).
    - Sem habilidades especiais, status ou defesa ativa nesta primeira
      versão — só ataque básico, acerto, crítico e redução por armadura.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ealen_ia.personagem import Personagem

MAX_TURNOS_PADRAO = 100


@dataclass(frozen=True)
class TurnoCombate:
    numero: int
    atacante: str
    alvo: str
    acertou: bool
    critico: bool
    dano_causado: float


@dataclass(frozen=True)
class ResultadoCombate:
    vencedor: str | None
    """Nome do vencedor, ou None em caso de empate (ninguém morreu em max_turnos)."""
    historico: list[TurnoCombate] = field(default_factory=list)

    @property
    def turnos_totais(self) -> int:
        return len(self.historico)


def _dano_de_ataque(atacante: Personagem) -> float:
    return max(atacante.dano_fisico_base, atacante.dano_magico_base)


def _resolver_turno(numero: int, atacante: Personagem, alvo: Personagem, rng: random.Random) -> TurnoCombate:
    acertou = rng.random() < atacante.chance_de_acerto
    if not acertou:
        return TurnoCombate(numero, atacante.nome, alvo.nome, acertou=False, critico=False, dano_causado=0.0)

    critico = rng.random() < atacante.chance_de_critico
    dano_bruto = _dano_de_ataque(atacante)
    if critico:
        dano_bruto *= atacante.parametros.multiplicador_critico

    dano_causado = alvo.receber_dano(dano_bruto)
    return TurnoCombate(numero, atacante.nome, alvo.nome, acertou=True, critico=critico, dano_causado=dano_causado)


def simulate_combat(
    a: Personagem,
    b: Personagem,
    seed: int | None = None,
    max_turnos: int = MAX_TURNOS_PADRAO,
) -> ResultadoCombate:
    """Roda um combate 1x1 completo entre `a` e `b`.

    Reinicia o HP de ambos pro máximo antes de começar (o combate é sempre
    "do zero"), depois alterna turnos até um dos dois morrer ou até
    `max_turnos` ser atingido (nesse caso o resultado é empate).

    Determinístico se `seed` for passado (mesmo `seed` -> mesmo resultado,
    útil pra debugar um combate específico), estocástico caso contrário.
    """
    a.hp_atual = a.hp_maximo
    b.hp_atual = b.hp_maximo
    rng = random.Random(seed)

    historico: list[TurnoCombate] = []
    ordem = (a, b)

    for numero_turno in range(1, max_turnos + 1):
        atacante, alvo = ordem[(numero_turno - 1) % 2], ordem[numero_turno % 2]
        turno = _resolver_turno(numero_turno, atacante, alvo, rng)
        historico.append(turno)

        if not alvo.esta_vivo:
            return ResultadoCombate(vencedor=atacante.nome, historico=historico)

    return ResultadoCombate(vencedor=None, historico=historico)

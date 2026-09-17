"""Atributos base dos personagens e as fórmulas que os convertem em números de combate.

Referência (AGENTS.md):
    Dain (Força)              -> dano corpo-a-corpo
    Eir  (Ressonância)        -> dano/cura mágica
    Nath (Vitalidade)         -> HP máximo
    Il   (Percepção)          -> precisão / chance de crítico
    Or   (Densidade)          -> defesa/armadura
    Len  (Som/Voz)            -> carisma, não usado em combate
    Ul   (Mistério)           -> sabedoria/intelecto, não usado em combate
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Atributos:
    """Os 7 atributos "de ficha" de um personagem. Len e Ul existem só por fidelidade
    ao jogo original — nenhuma fórmula de combate abaixo os utiliza."""

    dain: float
    eir: float
    nath: float
    il: float
    or_: float
    len_: float = 0.0
    ul: float = 0.0


def _clamp(valor: float, minimo: float, maximo: float) -> float:
    return max(minimo, min(maximo, valor))


@dataclass(frozen=True)
class ParametrosDeFormula:
    """Todos os coeficientes que a Camada 1 (auto-tuner) pode ajustar.

    Cada fórmula de combate abaixo é `base + coeficiente * atributo`, ou seja,
    uma reta. A derivada de uma reta em relação ao atributo é o próprio
    coeficiente — mas quem o auto-tuner ajusta não é o atributo do personagem,
    e sim esses coeficientes (o "quanto cada ponto de Dain vale em dano", por
    exemplo). É nesse coeficiente que a diferença finita da Camada 1 vai atuar.
    """

    hp_base: float = 50.0
    hp_por_nath: float = 8.0

    dano_fisico_base: float = 5.0
    dano_fisico_por_dain: float = 1.5

    dano_magico_base: float = 5.0
    dano_magico_por_eir: float = 1.5

    chance_acerto_base: float = 0.70
    chance_acerto_por_il: float = 0.02

    chance_critico_base: float = 0.05
    chance_critico_por_il: float = 0.015
    multiplicador_critico: float = 1.5

    reducao_dano_por_or: float = 0.01
    reducao_dano_maxima: float = 0.75


def hp_maximo(atributos: Atributos, p: ParametrosDeFormula) -> float:
    return p.hp_base + p.hp_por_nath * atributos.nath


def dano_fisico_base(atributos: Atributos, p: ParametrosDeFormula) -> float:
    return p.dano_fisico_base + p.dano_fisico_por_dain * atributos.dain


def dano_magico_base(atributos: Atributos, p: ParametrosDeFormula) -> float:
    return p.dano_magico_base + p.dano_magico_por_eir * atributos.eir


def chance_de_acerto(atributos: Atributos, p: ParametrosDeFormula) -> float:
    valor = p.chance_acerto_base + p.chance_acerto_por_il * atributos.il
    return _clamp(valor, 0.05, 0.99)


def chance_de_critico(atributos: Atributos, p: ParametrosDeFormula) -> float:
    valor = p.chance_critico_base + p.chance_critico_por_il * atributos.il
    return _clamp(valor, 0.0, 0.75)


def reducao_de_dano(atributos: Atributos, p: ParametrosDeFormula) -> float:
    """Fração do dano recebido que é absorvida pela armadura (Or). 0.30 = reduz 30%."""
    valor = p.reducao_dano_por_or * atributos.or_
    return _clamp(valor, 0.0, p.reducao_dano_maxima)

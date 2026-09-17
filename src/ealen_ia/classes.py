"""As 6 classes do jogo, cada uma como um preset de Atributos.

Os valores abaixo são um ponto de partida plausível, não um balanceamento
final — é exatamente esse "não final" que a Camada 1 (Monte Carlo + gradiente
descendente) existe pra corrigir, ajustando os coeficientes de
ParametrosDeFormula até as taxas de vitória convergirem pro alvo.
"""

from __future__ import annotations

from dataclasses import dataclass

from ealen_ia.atributos import Atributos


@dataclass(frozen=True)
class Classe:
    nome: str
    atributos_base: Atributos


LUMINAR = Classe(
    nome="Luminar",
    atributos_base=Atributos(dain=4, eir=10, nath=14, il=6, or_=12),
)

ENTROPISTA = Classe(
    nome="Entropista",
    atributos_base=Atributos(dain=6, eir=12, nath=8, il=10, or_=6),
)

CANTOR_DE_EALEN = Classe(
    nome="Cantor de Eälen",
    atributos_base=Atributos(dain=4, eir=14, nath=8, il=10, or_=6),
)

GUARDIAO = Classe(
    nome="Guardião",
    atributos_base=Atributos(dain=12, eir=4, nath=14, il=6, or_=10),
)

SOMBRILICO = Classe(
    nome="Sombrílico",
    atributos_base=Atributos(dain=8, eir=6, nath=9, il=13, or_=8),
)

RACHADOR = Classe(
    nome="Rachador",
    atributos_base=Atributos(dain=13, eir=3, nath=8, il=14, or_=4),
)

TODAS_AS_CLASSES: tuple[Classe, ...] = (
    LUMINAR,
    ENTROPISTA,
    CANTOR_DE_EALEN,
    GUARDIAO,
    SOMBRILICO,
    RACHADOR,
)

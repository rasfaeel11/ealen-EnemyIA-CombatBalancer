"""Personagem = uma classe + atributos + estado de combate (HP atual).

Este módulo ainda não é o simulador de combate (isso é a Camada 0, o próximo
passo do AGENTS.md) — aqui só existe o "personagem" como objeto: como ele
calcula seus próprios números de combate e como ele recebe dano.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ealen_ia.atributos import (
    Atributos,
    ParametrosDeFormula,
    chance_de_acerto,
    chance_de_critico,
    dano_fisico_base,
    dano_magico_base,
    hp_maximo,
    reducao_de_dano,
)
from ealen_ia.classes import Classe


@dataclass
class Personagem:
    nome: str
    classe: Classe
    atributos: Atributos
    parametros: ParametrosDeFormula = field(default_factory=ParametrosDeFormula)
    hp_atual: float = field(init=False)

    def __post_init__(self) -> None:
        self.hp_atual = self.hp_maximo

    @classmethod
    def da_classe(cls, nome: str, classe: Classe, parametros: ParametrosDeFormula | None = None) -> "Personagem":
        return cls(
            nome=nome,
            classe=classe,
            atributos=classe.atributos_base,
            parametros=parametros or ParametrosDeFormula(),
        )

    @property
    def hp_maximo(self) -> float:
        return hp_maximo(self.atributos, self.parametros)

    @property
    def dano_fisico_base(self) -> float:
        return dano_fisico_base(self.atributos, self.parametros)

    @property
    def dano_magico_base(self) -> float:
        return dano_magico_base(self.atributos, self.parametros)

    @property
    def chance_de_acerto(self) -> float:
        return chance_de_acerto(self.atributos, self.parametros)

    @property
    def chance_de_critico(self) -> float:
        return chance_de_critico(self.atributos, self.parametros)

    @property
    def reducao_de_dano(self) -> float:
        return reducao_de_dano(self.atributos, self.parametros)

    @property
    def esta_vivo(self) -> bool:
        return self.hp_atual > 0

    def receber_dano(self, dano_bruto: float) -> float:
        """Aplica a redução de dano (armadura) e desconta do HP atual.

        Retorna o dano efetivamente sofrido, já com a redução aplicada.
        """
        dano_efetivo = dano_bruto * (1 - self.reducao_de_dano)
        self.hp_atual = max(0.0, self.hp_atual - dano_efetivo)
        return dano_efetivo

    def curar(self, quantidade: float) -> None:
        self.hp_atual = min(self.hp_maximo, self.hp_atual + quantidade)

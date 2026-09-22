"""Camada 3 (opcional): modelagem da curva de progressão de XP/nível.

Script/módulo separado do resto do projeto — sem nenhuma relação com combate
ou balanceamento de classes (o AGENTS.md pede isso explicitamente: "trate
isso como um script separado, não misture com o auto-tuner de combate").

Dois conceitos de Cálculo 1 aplicados a uma curva de jogo:

    - DERIVADA = velocidade de crescimento do custo de XP por nível. Se
      f(n) é "quanto XP falta pra passar do nível n pro n+1", f'(n) diz o
      quão mais rápido esse custo cresce conforme n aumenta.
    - INTEGRAL = XP total acumulado até um nível N — a área sob a curva de
      f entre 0 e N. Como nível é discreto (1, 2, 3, ...), a soma de XP
      (sum(f(n) para n de 1 até N)) É um Riemann sum de passo 1: a versão
      discreta exata da integral, não uma aproximação grosseira dela.
      `integral_continua` calcula a versão contínua (fórmula fechada, nível
      tratado como variável real) pra comparar com a soma discreta — as duas
      divergem por um termo pequeno e previsível (ver `xp_acumulado_discreto`
      e os testes), o mesmo tipo de "erro de discretização" que aparece toda
      vez que se aproxima uma soma por uma integral (ou vice-versa).

Comparamos 3 formatos de curva — a mesma pergunta que qualquer RPG real
precisa responder: personagens de nível alto devem precisar de MUITO mais XP
proporcionalmente (exponencial), um pouco mais (quadrática), ou a mesma
quantidade extra a cada nível (linear)?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

FuncaoXP = Callable[[float], float]


@dataclass(frozen=True)
class CurvaXP:
    nome: str
    funcao: FuncaoXP
    derivada_analitica: FuncaoXP
    integral_continua: Callable[[float], float]  # integral de 0 até N (nível tratado como contínuo)


def curva_linear(a: float = 100.0, b: float = 50.0) -> CurvaXP:
    """f(n) = a + b*n — custo cresce sempre no mesmo ritmo (derivada
    constante = b, o mesmo em qualquer nível)."""
    return CurvaXP(
        nome="linear",
        funcao=lambda n: a + b * n,
        derivada_analitica=lambda n: b,
        integral_continua=lambda nivel_max: a * nivel_max + b * nivel_max**2 / 2,
    )


def curva_quadratica(a: float = 100.0, b: float = 20.0, c: float = 2.0) -> CurvaXP:
    """f(n) = a + b*n + c*n² — custo cresce cada vez mais rápido (derivada
    b + 2c*n, ela própria linear em n)."""
    return CurvaXP(
        nome="quadrática",
        funcao=lambda n: a + b * n + c * n**2,
        derivada_analitica=lambda n: b + 2 * c * n,
        integral_continua=lambda nivel_max: a * nivel_max + b * nivel_max**2 / 2 + c * nivel_max**3 / 3,
    )


def curva_exponencial(a: float = 100.0, r: float = 1.08) -> CurvaXP:
    """f(n) = a * r^n — custo cresce numa taxa PROPORCIONAL ao valor atual
    (derivada = f(n) * ln(r)), o padrão clássico de "cada nível fica
    proporcionalmente mais caro" em MMORPGs. `r` é o fator de crescimento
    por nível (ex: r=1.08 = 8% a mais de custo a cada nível)."""
    return CurvaXP(
        nome="exponencial",
        funcao=lambda n: a * r**n,
        derivada_analitica=lambda n: a * r**n * math.log(r),
        integral_continua=lambda nivel_max: a * (r**nivel_max - 1) / math.log(r),
    )


def derivada_numerica(funcao: FuncaoXP, n: float, h: float = 1e-3) -> float:
    """Diferença central — a mesma técnica da Camada 1 (`auto_tuner.py`),
    aqui usada só pra VALIDAR a derivada analítica de cada curva contra a
    definição de derivada como reta secante, não pra otimizar nada."""
    return (funcao(n + h) - funcao(n - h)) / (2 * h)


def xp_acumulado_discreto(funcao: FuncaoXP, nivel_maximo: int) -> float:
    """Soma de XP nível a nível, de 1 até `nivel_maximo` — um Riemann sum de
    passo 1 (a versão discreta exata da integral, já que nível é um número
    inteiro, não uma variável contínua)."""
    return sum(funcao(n) for n in range(1, nivel_maximo + 1))

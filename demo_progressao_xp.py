"""Demo da Camada 3 (opcional): modela 3 formatos de curva de XP/nível
(linear, quadrática, exponencial) e aplica dois conceitos de Cálculo 1 sobre
elas — a derivada (velocidade de crescimento do custo por nível) e a
integral/soma (XP total acumulado até um nível N).

Script separado do resto do projeto — não depende de combate nem de
balanceamento (AGENTS.md pede isso explicitamente).
"""

from ealen_ia.graficos import plot_series_multiplas
from ealen_ia.progressao_xp import (
    curva_exponencial,
    curva_linear,
    curva_quadratica,
    derivada_numerica,
    xp_acumulado_discreto,
)

NIVEL_MAXIMO = 50

curvas = [curva_linear(), curva_quadratica(), curva_exponencial()]
niveis = list(range(1, NIVEL_MAXIMO + 1))

print("--- Validando derivada analítica vs. numérica (diferença central, mesma técnica da Camada 1) ---")
for curva in curvas:
    maior_diferenca = max(abs(curva.derivada_analitica(n) - derivada_numerica(curva.funcao, n)) for n in niveis)
    print(f"{curva.nome:<12} maior diferença analítica vs. numérica: {maior_diferenca:.6f}")

xp_por_nivel = {curva.nome: [curva.funcao(n) for n in niveis] for curva in curvas}
plot_series_multiplas(xp_por_nivel, niveis, "xp_por_nivel.png", "XP necessário por nível", "XP", "nível")

velocidade = {curva.nome: [curva.derivada_analitica(n) for n in niveis] for curva in curvas}
plot_series_multiplas(
    velocidade, niveis, "velocidade_xp.png", "Velocidade de crescimento do custo (derivada)", "d(XP)/d(nível)", "nível"
)

xp_acumulado = {curva.nome: [xp_acumulado_discreto(curva.funcao, n) for n in niveis] for curva in curvas}
plot_series_multiplas(
    xp_acumulado, niveis, "xp_acumulado.png", "XP total acumulado até o nível N (soma discreta)", "XP acumulado", "nível N"
)

print(f"\n--- Soma discreta vs. integral contínua fechada, no nível {NIVEL_MAXIMO} ---")
for curva in curvas:
    soma_discreta = xp_acumulado_discreto(curva.funcao, NIVEL_MAXIMO)
    integral_continua = curva.integral_continua(NIVEL_MAXIMO)
    diferenca_percentual = abs(soma_discreta - integral_continua) / soma_discreta
    print(
        f"{curva.nome:<12} soma discreta={soma_discreta:>12.1f}  "
        f"integral contínua={integral_continua:>12.1f}  diferença={diferenca_percentual:.1%}"
    )

print("\nGráficos salvos: xp_por_nivel.png, velocidade_xp.png, xp_acumulado.png")

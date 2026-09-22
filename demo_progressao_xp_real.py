"""Análise da curva de XP REAL do jogo Eälen (repositório separado, em
../ealen/server/src/combat/leveling.ts): `xpToNextLevel(level) = level * 100`.
Usa as ferramentas de Cálculo 1 da Camada 3 (derivada = velocidade de
crescimento, integral/soma = XP acumulado) sobre essa fórmula real, e
compara com duas alternativas hipotéticas.

Só leitura — não depende do código do jogo nem escreve nada lá (AGENTS.md é
explícito sobre isso). Ver INTEGRACAO_COM_O_JOGO.md pra a análise completa
de por que essa é a ÚNICA integração de verdade viável hoje entre os dois
projetos: o motor de combate real usa d20 e 6 ações (incompatível com o
simulador desta Camada 0), as classes não têm atributos-base fixos (o
jogador escolhe na criação), e os inimigos do bestiário são autorais — a
curva de XP é a única estrutura numérica que bate 1:1 com o que a Camada 3
já modela.
"""

from ealen_ia.graficos import plot_series_multiplas
from ealen_ia.progressao_xp import curva_exponencial, curva_linear, curva_quadratica, derivada_numerica, xp_acumulado_discreto

NIVEL_MAXIMO = 50
niveis = list(range(1, NIVEL_MAXIMO + 1))

# A curva REAL do jogo (xpToNextLevel em leveling.ts): level * 100, ou seja
# a=0, b=100 na nossa parametrização de curva_linear.
curva_real = curva_linear(a=0.0, b=100.0)

# Duas alternativas hipotéticas, só pra contraste visual — não são propostas
# de mudança, é só "como ficaria diferente se o jogo tivesse escolhido outro
# formato de curva".
alternativa_quadratica = curva_quadratica(a=50.0, b=30.0, c=3.0)
alternativa_exponencial = curva_exponencial(a=90.0, r=1.06)

print("--- Curva real do jogo: xpToNextLevel(level) = level * 100 ---")
print(f"derivada (velocidade de crescimento): {curva_real.derivada_analitica(1)} XP por nível — CONSTANTE (é linear, a derivada não depende do nível)")
diferenca_numerica = abs(curva_real.derivada_analitica(25) - derivada_numerica(curva_real.funcao, 25))
print(f"validação numérica (diferença central) no nível 25: diferença de {diferenca_numerica:.6f} em relação à derivada analítica")

for nivel_de_referencia in (10, 25, 50):
    xp_total = xp_acumulado_discreto(curva_real.funcao, nivel_de_referencia)
    print(f"XP total acumulado até o nível {nivel_de_referencia}: {xp_total:,.0f}")

xp_por_nivel = {
    "real do jogo (linear)": [curva_real.funcao(n) for n in niveis],
    "alternativa quadrática": [alternativa_quadratica.funcao(n) for n in niveis],
    "alternativa exponencial": [alternativa_exponencial.funcao(n) for n in niveis],
}
plot_series_multiplas(
    xp_por_nivel, niveis, "xp_por_nivel_real_vs_alternativas.png",
    "Curva de XP real do jogo vs. alternativas hipotéticas", "XP necessário", "nível"
)

xp_acumulado = {
    "real do jogo (linear)": [xp_acumulado_discreto(curva_real.funcao, n) for n in niveis],
    "alternativa quadrática": [xp_acumulado_discreto(alternativa_quadratica.funcao, n) for n in niveis],
    "alternativa exponencial": [xp_acumulado_discreto(alternativa_exponencial.funcao, n) for n in niveis],
}
plot_series_multiplas(
    xp_acumulado, niveis, "xp_acumulado_real_vs_alternativas.png",
    "XP total acumulado: real do jogo vs. alternativas", "XP acumulado", "nível N"
)

print("\nGráficos salvos: xp_por_nivel_real_vs_alternativas.png, xp_acumulado_real_vs_alternativas.png")

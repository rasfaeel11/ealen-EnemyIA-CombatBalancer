"""Demo da Camada 1 completa: mede o desequilíbrio atual, roda as duas etapas
do auto-tuner (Etapa A: coeficientes globais de fórmula, só nos espelhos;
Etapa B: atributos de cada classe, na matriz completa), e mostra o antes/
depois de cada etapa — em texto (matriz no terminal) e em imagem (PNGs
salvos na pasta atual).
"""

from ealen_ia.ajuste_de_classes import ajustar_atributos
from ealen_ia.atributos import ParametrosDeFormula
from ealen_ia.auto_tuner import ajustar_parametros
from ealen_ia.balanceamento_io import salvar_balanceamento
from ealen_ia.classes import TODAS_AS_CLASSES
from ealen_ia.graficos import plot_convergencia, plot_matriz_de_confrontos
from ealen_ia.monte_carlo import matriz_de_confrontos


def imprimir_matriz(titulo: str, matriz) -> None:
    print(f"\n--- {titulo} ---")
    for (nome_a, nome_b), resultado in matriz.items():
        print(
            f"{nome_a:<16} vs {nome_b:<16} "
            f"taxa_a={resultado.taxa_vitoria_a:>5.0%}  erro={resultado.erro():.2f}"
        )
    erro_medio_da_matriz = sum(r.erro() for r in matriz.values()) / len(matriz)
    print(f"erro médio: {erro_medio_da_matriz:.3f}")


N_SIMULACOES = 500
SEED = 1

parametros_iniciais = ParametrosDeFormula()
matriz_antes = matriz_de_confrontos(TODAS_AS_CLASSES, n=N_SIMULACOES, seed=SEED, parametros=parametros_iniciais)
imprimir_matriz("ANTES (parâmetros e atributos padrão)", matriz_antes)
plot_matriz_de_confrontos(matriz_antes, TODAS_AS_CLASSES, "matriz_antes.png")

# --- Etapa A: calibra os coeficientes globais de fórmula, só nos espelhos ---
print("\n=== Etapa A: ajustando ParametrosDeFormula (gradiente descendente, só espelhos) ===")
parametros_finais, historico_parametros, _passos_a = ajustar_parametros(
    parametros_iniciais,
    classes=TODAS_AS_CLASSES,
    n_simulacoes=400,
    max_iteracoes=25,
    seed=SEED,
    apenas_espelhos=True,
)
plot_convergencia(historico_parametros, "convergencia_parametros.png")
print(f"erro médio (espelhos) por iteração: {[round(e, 3) for e in historico_parametros]}")
print(f"parâmetros finais: {parametros_finais}")

matriz_apos_etapa_a = matriz_de_confrontos(TODAS_AS_CLASSES, n=N_SIMULACOES, seed=SEED, parametros=parametros_finais)
imprimir_matriz("DEPOIS da Etapa A (só parâmetros globais)", matriz_apos_etapa_a)
plot_matriz_de_confrontos(matriz_apos_etapa_a, TODAS_AS_CLASSES, "matriz_apos_etapa_a.png")

# --- Etapa B: calibra os atributos de cada classe, na matriz completa ---
print("\n=== Etapa B: ajustando atributos por classe (gradiente descendente, matriz completa) ===")
atributos_finais, historico_atributos, _passos_b = ajustar_atributos(
    classes=TODAS_AS_CLASSES,
    parametros=parametros_finais,
    n_simulacoes=150,
    max_iteracoes=10,
    seed=SEED,
)
plot_convergencia(historico_atributos, "convergencia_atributos.png")
print(f"erro médio (matriz completa) por iteração: {[round(e, 3) for e in historico_atributos]}")
for nome_classe, atributos in atributos_finais.items():
    print(f"  {nome_classe:<16} {atributos}")

matriz_final = matriz_de_confrontos(
    TODAS_AS_CLASSES,
    n=N_SIMULACOES,
    seed=SEED,
    parametros=parametros_finais,
    atributos_por_classe=atributos_finais,
)
imprimir_matriz("DEPOIS da Etapa B (parâmetros + atributos ajustados)", matriz_final)
plot_matriz_de_confrontos(matriz_final, TODAS_AS_CLASSES, "matriz_final.png")

salvar_balanceamento(parametros_finais, atributos_finais)

print(
    "\nGráficos salvos: matriz_antes.png, convergencia_parametros.png, "
    "matriz_apos_etapa_a.png, convergencia_atributos.png, matriz_final.png"
)
print("Balanceamento salvo: balanceamento.json (usado por demo_q_learning.py)")

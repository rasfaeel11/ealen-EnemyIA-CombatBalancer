"""Demo da extensão de Q-learning com aproximação linear (`q_linear.py`),
comparada lado a lado com a tabela (`q_learning.py`) sobre o mesmo par de
treino — pra ver na prática a diferença de estabilidade entre as duas.

Usa o resultado balanceado da Camada 1 quando `balanceamento.json` existe,
mesma lógica de `demo_q_learning.py`.
"""

from pathlib import Path

from ealen_ia.atributos import Atributos, ParametrosDeFormula
from ealen_ia.balanceamento_io import CAMINHO_PADRAO, carregar_balanceamento
from ealen_ia.classes import GUARDIAO, RACHADOR
from ealen_ia.graficos import plot_series_multiplas
from ealen_ia.personagem import Personagem
from ealen_ia.q_learning import avaliar_taxa_vitoria, politica_sempre_ataca, treinar
from ealen_ia.q_linear import avaliar_taxa_vitoria_linear, treinar_linear

SEED = 1
N_EPISODIOS = 40000
N_CHECKPOINTS = 40

if Path(CAMINHO_PADRAO).exists():
    parametros, atributos_por_classe = carregar_balanceamento()
    print(f"Usando o balanceamento da Camada 1 ({CAMINHO_PADRAO}).")
else:
    parametros, atributos_por_classe = None, None
    print(f"{CAMINHO_PADRAO} não encontrado — usando atributos padrão.")


def _personagem(nome: str, classe) -> Personagem:
    atributos = (atributos_por_classe or {}).get(classe.nome, classe.atributos_base)
    return Personagem(nome, classe, atributos, parametros or ParametrosDeFormula())


a = _personagem("Guardião", GUARDIAO)
b = _personagem("Rachador", RACHADOR)

print("\n=== Treinando a versão TABULAR (garantia de convergência) ===")
tabela_q, progresso_tabular = treinar(
    pares_de_treino=[(a, b)], n_episodios=N_EPISODIOS, n_checkpoints=N_CHECKPOINTS, seed=SEED
)
taxa_final_tabular = avaliar_taxa_vitoria(tabela_q, a, b, politica_sempre_ataca, n_combates=200, seed=99)
print(f"taxa de vitória final (tabular): {taxa_final_tabular:.0%}")

print("\n=== Treinando a versão LINEAR (instável — ver docstring de q_linear.py) ===")
pesos, progresso_linear = treinar_linear(
    pares_de_treino=[(a, b)], n_episodios=N_EPISODIOS, n_checkpoints=N_CHECKPOINTS, alfa=0.02, seed=SEED
)
taxa_final_linear = avaliar_taxa_vitoria_linear(pesos, a, b, politica_sempre_ataca, n_combates=200, seed=99)
print(f"taxa de vitória final (linear, esta semente): {taxa_final_linear:.0%}")

checkpoints = list(range(1, N_CHECKPOINTS + 1))
plot_series_multiplas(
    {"tabular": progresso_tabular, "linear": progresso_linear},
    checkpoints,
    "comparacao_tabular_vs_linear.png",
    "Tabular vs. aproximação linear (Guardião vs Rachador)",
    "taxa de vitória vs. 'sempre ataca'",
    "checkpoint do treino",
)

print("\nGráfico salvo: comparacao_tabular_vs_linear.png")
print(
    "\nNote a diferença de comportamento das duas curvas: a tabular sobe e "
    "estabiliza; a linear tende a ser mais ruidosa/instável (pode até "
    "colapsar pra uma política ruim dependendo da semente — ver "
    "test_q_linear.py, que testa isso em várias sementes de propósito)."
)

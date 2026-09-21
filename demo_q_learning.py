"""Demo da Camada 2: treina o agente de Q-learning por self-play sobre todos
os confrontos entre classes, plota o progresso do treino, e reporta a taxa
de vitória final do agente contra a política ingênua ("sempre ataca") em
cada confronto — a "Definição de pronto" da Camada 2 no AGENTS.md.

Usa o resultado balanceado da Camada 1 quando `balanceamento.json` existe
(gerado por `demo_balanceamento.py`) — treinar sobre classes já balanceadas
dá um sinal de treino mais realista (AGENTS.md). Sem esse arquivo, cai de
volta pros atributos padrão de cada classe.
"""

from pathlib import Path

from ealen_ia.atributos import Atributos, ParametrosDeFormula
from ealen_ia.balanceamento_io import CAMINHO_PADRAO, carregar_balanceamento
from ealen_ia.classes import Classe, TODAS_AS_CLASSES
from ealen_ia.graficos import plot_progresso_treino
from ealen_ia.personagem import Personagem
from ealen_ia.q_learning import avaliar_taxa_vitoria, politica_sempre_ataca, treinar

SEED = 1


def construir_pares_de_treino(
    classes: tuple[Classe, ...],
    parametros: ParametrosDeFormula | None = None,
    atributos_por_classe: dict[str, Atributos] | None = None,
) -> list[tuple[Personagem, Personagem]]:
    """Todos os confrontos (i <= j, incluindo espelho) — mesma convenção de
    `monte_carlo.matriz_de_confrontos`, com nomes distintos no espelho pelo
    mesmo motivo (o vencedor é identificado pelo nome). `parametros` e
    `atributos_por_classe` (opcionais) permitem plugar o resultado
    balanceado da Camada 1 em vez do `atributos_base`/`ParametrosDeFormula()`
    padrão de cada classe.
    """
    pares = []
    for i, classe_a in enumerate(classes):
        for classe_b in classes[i:]:
            nome_a = f"{classe_a.nome} (1)" if classe_a is classe_b else classe_a.nome
            nome_b = f"{classe_b.nome} (2)" if classe_a is classe_b else classe_b.nome
            atributos_a = (atributos_por_classe or {}).get(classe_a.nome, classe_a.atributos_base)
            atributos_b = (atributos_por_classe or {}).get(classe_b.nome, classe_b.atributos_base)
            a = Personagem(nome_a, classe_a, atributos_a, parametros or ParametrosDeFormula())
            b = Personagem(nome_b, classe_b, atributos_b, parametros or ParametrosDeFormula())
            pares.append((a, b))
    return pares


if Path(CAMINHO_PADRAO).exists():
    parametros, atributos_por_classe = carregar_balanceamento()
    print(f"Usando o balanceamento da Camada 1 ({CAMINHO_PADRAO}).")
else:
    parametros, atributos_por_classe = None, None
    print(f"{CAMINHO_PADRAO} não encontrado — usando atributos padrão (rode demo_balanceamento.py primeiro pra treinar sobre classes balanceadas).")

pares_de_treino = construir_pares_de_treino(TODAS_AS_CLASSES, parametros, atributos_por_classe)
par_de_avaliacao = next(p for p in pares_de_treino if p[0].classe.nome == "Guardião" and p[1].classe.nome == "Rachador")

print(f"Treinando por self-play sobre {len(pares_de_treino)} confrontos diferentes...")
tabela_q, historico_taxa_vitoria = treinar(
    pares_de_treino=pares_de_treino,
    n_episodios=60000,
    n_checkpoints=40,
    par_de_avaliacao=par_de_avaliacao,
    seed=SEED,
)
plot_progresso_treino(historico_taxa_vitoria, "progresso_q_learning.png")
print(f"progresso (Guardião vs Rachador): {[round(t, 2) for t in historico_taxa_vitoria]}")
print(f"tamanho da tabela Q (estados x ações visitados): {len(tabela_q)}")

print("\n--- Taxa de vitória do agente treinado vs. política 'sempre ataca' ---")
for a, b in pares_de_treino:
    taxa = avaliar_taxa_vitoria(tabela_q, a, b, politica_sempre_ataca, n_combates=100, seed=SEED)
    print(f"{a.classe.nome:<16} vs {b.classe.nome:<16} taxa_vitoria_agente={taxa:.0%}")

print("\nGráfico salvo: progresso_q_learning.png")

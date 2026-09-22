"""Visualizações da Camada 1 (convergência do auto-tuner e mapa de calor da
matriz de confrontos) e da Camada 2 (progresso do treino de Q-learning).
Salva PNG em vez de abrir janela — mais confiável rodando por script/CI, e o
arquivo fica fácil de olhar ou compartilhar depois.

Duas regras de cor (do form/color guide de dataviz), nunca um "rainbow":
    - convergência: uma série só, um hue sequencial (azul, claro->escuro
      não se aplica a uma única linha, mas o princípio "uma cor por série" sim).
    - matriz de confrontos: a pergunta é "o quanto esse confronto se desvia de
      50/50", ou seja, tem polaridade (favorece a ou b) -> cor divergente,
      dois tons opostos com um meio neutro em vez de uma escala sequencial.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ealen_ia.classes import Classe, TODAS_AS_CLASSES
from ealen_ia.monte_carlo import ResultadoConfronto

_SUPERFICIE = "#fcfcfb"
_TINTA_PRIMARIA = "#0b0b0b"
_TINTA_SECUNDARIA = "#52514e"
_TINTA_MUTED = "#898781"
_GRADE = "#e1e0d9"
_AZUL = "#2a78d6"
_VERMELHO = "#e34948"
_CINZA_NEUTRO = "#f0efec"

_DIVERGENTE_AZUL_VERMELHO = LinearSegmentedColormap.from_list(
    "azul_vermelho", [_AZUL, _CINZA_NEUTRO, _VERMELHO]
)

# Paleta categórica (slots 1-3 do palette.md da skill de dataviz) — ordem
# fixa, nunca ciclada, pra até 3 séries num mesmo gráfico (ex: as 3 curvas de
# XP da Camada 3).
_CATEGORICO = ("#2a78d6", "#eb6834", "#1baf7a")


def plot_series_multiplas(
    series: dict[str, list[float]],
    eixo_x: list[float],
    caminho: str,
    titulo: str,
    ylabel: str,
    xlabel: str = "nível",
) -> None:
    """Gráfico de linhas com até 3 séries nomeadas, cada uma numa cor fixa da
    paleta categórica, com legenda — usado pela Camada 3 (progressão de XP)
    pra comparar curvas diferentes sobre o mesmo eixo."""
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=_SUPERFICIE)
    ax.set_facecolor(_SUPERFICIE)

    for (nome, valores), cor in zip(series.items(), _CATEGORICO):
        ax.plot(eixo_x, valores, color=cor, linewidth=2, label=nome)

    ax.set_title(titulo, color=_TINTA_PRIMARIA, fontsize=13, loc="left")
    ax.set_xlabel(xlabel, color=_TINTA_SECUNDARIA)
    ax.set_ylabel(ylabel, color=_TINTA_SECUNDARIA)
    ax.tick_params(colors=_TINTA_MUTED)
    ax.grid(True, color=_GRADE, linewidth=0.8)
    legenda = ax.legend(frameon=False, labelcolor=_TINTA_SECUNDARIA)
    for texto in legenda.get_texts():
        texto.set_color(_TINTA_SECUNDARIA)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def plot_convergencia(historico_de_erro: list[float], caminho: str = "convergencia.png") -> None:
    """Gráfico de convergência do erro ao longo das iterações do auto-tuner —
    a "definição de pronto" da Camada 1 no AGENTS.md."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_SUPERFICIE)
    ax.set_facecolor(_SUPERFICIE)

    iteracoes = list(range(1, len(historico_de_erro) + 1))
    ax.plot(iteracoes, historico_de_erro, color=_AZUL, linewidth=2, solid_capstyle="round")
    ax.scatter(iteracoes, historico_de_erro, color=_AZUL, s=18, zorder=3)

    ax.set_title("Convergência do auto-tuner", color=_TINTA_PRIMARIA, fontsize=13, loc="left")
    ax.set_xlabel("iteração", color=_TINTA_SECUNDARIA)
    ax.set_ylabel("erro médio (|taxa de vitória - 0.5|)", color=_TINTA_SECUNDARIA)
    ax.set_ylim(bottom=0)
    ax.tick_params(colors=_TINTA_MUTED)
    ax.grid(True, color=_GRADE, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def plot_progresso_treino(historico_taxa_vitoria: list[float], caminho: str = "progresso_q_learning.png") -> None:
    """Curva de progresso do treino de Q-learning: taxa de vitória do agente
    (política gulosa, sem exploração) contra a política fixa "sempre ataca",
    medida a cada checkpoint do treino.

    Não é a mesma coisa que o gráfico de convergência da Camada 1 — lá o
    "erro" tinha um alvo fixo (0.5) porque o objetivo era equilíbrio; aqui o
    objetivo é o agente ficar bom, então o que queremos ver é a taxa subir
    e estabilizar perto de um valor alto (não perto de 0).
    """
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_SUPERFICIE)
    ax.set_facecolor(_SUPERFICIE)

    checkpoints = list(range(1, len(historico_taxa_vitoria) + 1))
    ax.plot(checkpoints, historico_taxa_vitoria, color=_AZUL, linewidth=2, solid_capstyle="round")
    ax.scatter(checkpoints, historico_taxa_vitoria, color=_AZUL, s=18, zorder=3)
    ax.axhline(0.5, color=_TINTA_MUTED, linewidth=1, linestyle="--")

    ax.set_title("Progresso do treino (Q-learning vs. política fixa)", color=_TINTA_PRIMARIA, fontsize=13, loc="left")
    ax.set_xlabel("checkpoint do treino", color=_TINTA_SECUNDARIA)
    ax.set_ylabel("taxa de vitória vs. 'sempre ataca'", color=_TINTA_SECUNDARIA)
    ax.set_ylim(0, 1)
    ax.tick_params(colors=_TINTA_MUTED)
    ax.grid(True, color=_GRADE, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def plot_matriz_de_confrontos(
    matriz: dict[tuple[str, str], ResultadoConfronto],
    classes: tuple[Classe, ...] = TODAS_AS_CLASSES,
    caminho: str = "matriz_de_confrontos.png",
) -> None:
    """Mapa de calor da taxa de vitória de cada par de classes.

    Só a metade "de cima" da matriz é preenchida — cada par só foi simulado
    uma vez (ver `monte_carlo.matriz_de_confrontos`) — a metade de baixo fica
    em branco em vez de mostrar um valor inventado.
    """
    nomes = [c.nome for c in classes]
    n = len(nomes)
    valores = np.full((n, n), np.nan)
    for i, nome_a in enumerate(nomes):
        for j, nome_b in enumerate(nomes):
            if j < i:
                continue
            chave = (nome_a, nome_b) if (nome_a, nome_b) in matriz else (nome_b, nome_a)
            valores[i, j] = matriz[chave].taxa_vitoria_a

    fig, ax = plt.subplots(figsize=(1.1 * n + 2, 1.1 * n + 1), facecolor=_SUPERFICIE)
    ax.set_facecolor(_SUPERFICIE)

    mapa = np.ma.masked_invalid(valores)
    imagem = ax.imshow(mapa, cmap=_DIVERGENTE_AZUL_VERMELHO, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_xticklabels(nomes, rotation=45, ha="right", color=_TINTA_SECUNDARIA)
    ax.set_yticks(range(n))
    ax.set_yticklabels(nomes, color=_TINTA_SECUNDARIA)
    ax.set_title(
        "Taxa de vitória por confronto (linha vs coluna)", color=_TINTA_PRIMARIA, fontsize=12, loc="left"
    )

    for i in range(n):
        for j in range(n):
            if np.isnan(valores[i, j]):
                continue
            cor_texto = _TINTA_PRIMARIA if 0.3 < valores[i, j] < 0.7 else "#ffffff"
            ax.text(j, i, f"{valores[i, j]:.0%}", ha="center", va="center", color=cor_texto, fontsize=9)

    barra = fig.colorbar(imagem, ax=ax, fraction=0.046, pad=0.04)
    barra.set_label("taxa de vitória", color=_TINTA_SECUNDARIA)
    barra.ax.tick_params(colors=_TINTA_MUTED)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)

"""Export/import do resultado do balanceamento (Camada 1) como JSON — pra
`demo_q_learning.py` (Camada 2) poder treinar sobre classes já balanceadas
em vez dos atributos padrão, sem precisar rodar o auto-tuner de novo.

Não é o export pro jogo web mencionado no AGENTS.md ("se algum dia os
parâmetros balanceados forem incorporados ao jogo") — isso aqui é só um
artefato interno entre os dois módulos deste projeto, reproduzível a
qualquer momento rodando `demo_balanceamento.py` de novo com o mesmo seed.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ealen_ia.atributos import Atributos, ParametrosDeFormula

CAMINHO_PADRAO = "balanceamento.json"


def salvar_balanceamento(
    parametros: ParametrosDeFormula,
    atributos_por_classe: dict[str, Atributos],
    caminho: str = CAMINHO_PADRAO,
) -> None:
    dados = {
        "parametros": asdict(parametros),
        "atributos_por_classe": {nome: asdict(atributos) for nome, atributos in atributos_por_classe.items()},
    }
    Path(caminho).write_text(json.dumps(dados, indent=2, ensure_ascii=False), encoding="utf-8")


def carregar_balanceamento(caminho: str = CAMINHO_PADRAO) -> tuple[ParametrosDeFormula, dict[str, Atributos]]:
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    parametros = ParametrosDeFormula(**dados["parametros"])
    atributos_por_classe = {
        nome: Atributos(**atributos) for nome, atributos in dados["atributos_por_classe"].items()
    }
    return parametros, atributos_por_classe

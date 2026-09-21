from ealen_ia.atributos import Atributos, ParametrosDeFormula
from ealen_ia.balanceamento_io import carregar_balanceamento, salvar_balanceamento


def test_salvar_e_carregar_reproduz_os_mesmos_valores(tmp_path):
    caminho = str(tmp_path / "balanceamento.json")
    parametros = ParametrosDeFormula(hp_por_nath=7.5, dano_fisico_por_dain=1.3)
    atributos_por_classe = {
        "Luminar": Atributos(dain=4, eir=10, nath=14, il=6, or_=12),
        "Rachador": Atributos(dain=15, eir=3, nath=8, il=14, or_=4),
    }

    salvar_balanceamento(parametros, atributos_por_classe, caminho)
    parametros_carregados, atributos_carregados = carregar_balanceamento(caminho)

    assert parametros_carregados == parametros
    assert atributos_carregados == atributos_por_classe


def test_arquivo_salvo_e_json_legivel(tmp_path):
    import json

    caminho = str(tmp_path / "balanceamento.json")
    salvar_balanceamento(ParametrosDeFormula(), {"Guardião": Atributos(dain=12, eir=4, nath=14, il=6, or_=10)}, caminho)

    dados = json.loads((tmp_path / "balanceamento.json").read_text(encoding="utf-8"))

    assert "parametros" in dados
    assert "atributos_por_classe" in dados
    assert dados["atributos_por_classe"]["Guardião"]["dain"] == 12

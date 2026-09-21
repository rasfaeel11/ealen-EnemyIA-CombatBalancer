from ealen_ia.ajuste_de_classes import ATRIBUTOS_AJUSTAVEIS, ajustar_atributos, erro_medio_da_classe
from ealen_ia.atributos import ParametrosDeFormula
from ealen_ia.classes import ENTROPISTA, GUARDIAO, LUMINAR, RACHADOR


def test_erro_medio_da_classe_fica_entre_0_e_0_5():
    classes = (GUARDIAO, RACHADOR)
    atributos_por_classe = {c.nome: c.atributos_base for c in classes}

    erro = erro_medio_da_classe("Guardião", atributos_por_classe, classes, ParametrosDeFormula(), 30, 1)

    assert 0.0 <= erro <= 0.5


def test_ajustar_atributos_produz_um_historico_por_iteracao():
    classes = (GUARDIAO, RACHADOR)

    atributos_finais, historico_de_erro, passos = ajustar_atributos(
        classes=classes,
        n_simulacoes=20,
        max_iteracoes=2,
        seed=1,
    )

    assert len(historico_de_erro) == 2
    assert len(passos) == 2 * len(classes) * len(ATRIBUTOS_AJUSTAVEIS)
    assert set(atributos_finais.keys()) == {"Guardião", "Rachador"}


def test_ajustar_atributos_e_deterministico_com_mesmo_seed():
    classes = (GUARDIAO, RACHADOR)

    resultado1 = ajustar_atributos(classes=classes, n_simulacoes=20, max_iteracoes=2, seed=5)
    resultado2 = ajustar_atributos(classes=classes, n_simulacoes=20, max_iteracoes=2, seed=5)

    assert resultado1[0] == resultado2[0]
    assert resultado1[1] == resultado2[1]


def test_ajustar_atributos_nunca_deixa_atributo_negativo():
    classes = (GUARDIAO, RACHADOR, LUMINAR)

    atributos_finais, _, _ = ajustar_atributos(classes=classes, n_simulacoes=20, max_iteracoes=2, seed=1)

    for atributos in atributos_finais.values():
        for nome_atributo in ATRIBUTOS_AJUSTAVEIS:
            assert getattr(atributos, nome_atributo) >= 0.0


def test_ajustar_atributos_nunca_mexe_em_len_ou_ul():
    classes = (GUARDIAO, RACHADOR)

    _, _, passos = ajustar_atributos(classes=classes, n_simulacoes=20, max_iteracoes=2, seed=1)

    assert {p.atributo for p in passos} == set(ATRIBUTOS_AJUSTAVEIS)
    assert "len_" not in {p.atributo for p in passos}
    assert "ul" not in {p.atributo for p in passos}


def test_ajustar_atributos_nao_muta_as_classes_originais():
    atributos_base_guardiao_antes = GUARDIAO.atributos_base
    atributos_base_entropista_antes = ENTROPISTA.atributos_base

    ajustar_atributos(classes=(GUARDIAO, ENTROPISTA), n_simulacoes=20, max_iteracoes=2, seed=1)

    assert GUARDIAO.atributos_base == atributos_base_guardiao_antes
    assert ENTROPISTA.atributos_base == atributos_base_entropista_antes

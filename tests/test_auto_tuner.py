from ealen_ia.atributos import ParametrosDeFormula
from ealen_ia.auto_tuner import PARAMETROS_AJUSTAVEIS, ajustar_parametros, erro_medio
from ealen_ia.classes import GUARDIAO, RACHADOR


def test_erro_medio_fica_entre_0_e_0_5():
    erro = erro_medio(ParametrosDeFormula(), classes=(GUARDIAO, RACHADOR), n_simulacoes=50, seed=1)

    assert 0.0 <= erro <= 0.5


def test_ajustar_parametros_produz_um_historico_por_iteracao():
    classes = (GUARDIAO, RACHADOR)

    _, historico_de_erro, passos = ajustar_parametros(
        ParametrosDeFormula(),
        classes=classes,
        n_simulacoes=30,
        max_iteracoes=3,
        seed=1,
    )

    assert len(historico_de_erro) == 3
    assert len(passos) == 3 * len(PARAMETROS_AJUSTAVEIS)
    assert all(nome in PARAMETROS_AJUSTAVEIS for nome in {p.parametro for p in passos})


def test_ajustar_parametros_e_deterministico_com_mesmo_seed():
    classes = (GUARDIAO, RACHADOR)

    resultado1 = ajustar_parametros(ParametrosDeFormula(), classes=classes, n_simulacoes=30, max_iteracoes=2, seed=5)
    resultado2 = ajustar_parametros(ParametrosDeFormula(), classes=classes, n_simulacoes=30, max_iteracoes=2, seed=5)

    assert resultado1[0] == resultado2[0]
    assert resultado1[1] == resultado2[1]


def test_ajustar_parametros_nunca_deixa_coeficiente_negativo():
    classes = (GUARDIAO, RACHADOR)

    parametros_finais, _, _ = ajustar_parametros(
        ParametrosDeFormula(), classes=classes, n_simulacoes=30, max_iteracoes=3, seed=1
    )

    for nome_parametro in PARAMETROS_AJUSTAVEIS:
        assert getattr(parametros_finais, nome_parametro) >= 0.0

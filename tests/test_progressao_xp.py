from ealen_ia.progressao_xp import (
    curva_exponencial,
    curva_linear,
    curva_quadratica,
    derivada_numerica,
    xp_acumulado_discreto,
)


def test_curva_linear_no_nivel_0_e_o_termo_independente():
    curva = curva_linear(a=100.0, b=50.0)
    assert curva.funcao(0) == 100.0


def test_curva_linear_derivada_e_constante():
    curva = curva_linear(a=100.0, b=50.0)
    assert curva.derivada_analitica(1) == 50.0
    assert curva.derivada_analitica(40) == 50.0


def test_curva_quadratica_derivada_cresce_com_o_nivel():
    curva = curva_quadratica(a=100.0, b=20.0, c=2.0)
    assert curva.derivada_analitica(10) > curva.derivada_analitica(1)
    assert curva.derivada_analitica(5) == 20.0 + 2 * 2.0 * 5


def test_curva_exponencial_derivada_e_proporcional_ao_valor():
    import math

    curva = curva_exponencial(a=100.0, r=1.08)
    n = 10
    assert abs(curva.derivada_analitica(n) - curva.funcao(n) * math.log(1.08)) < 1e-9


def test_derivada_numerica_bate_com_analitica_pras_3_curvas():
    for curva in (curva_linear(), curva_quadratica(), curva_exponencial()):
        for n in (1, 10, 30):
            diferenca = abs(curva.derivada_analitica(n) - derivada_numerica(curva.funcao, n))
            assert diferenca < 1e-2


def test_xp_acumulado_discreto_e_a_soma_termo_a_termo():
    curva = curva_linear(a=10.0, b=5.0)
    esperado = sum(curva.funcao(n) for n in range(1, 6))
    assert xp_acumulado_discreto(curva.funcao, 5) == esperado


def test_soma_discreta_diverge_da_integral_continua_por_um_termo_previsivel():
    # Pra f(n) = a + b*n: soma_{1..N} = a*N + b*N(N+1)/2, integral_0^N = a*N + b*N^2/2.
    # A diferença exata entre as duas é b*N/2 — um jeito de verificar que a
    # soma discreta (Riemann sum de passo 1) e a integral contínua não são
    # coincidência, tem uma relação algébrica exata entre elas.
    a, b, N = 100.0, 50.0, 20
    curva = curva_linear(a=a, b=b)

    soma_discreta = xp_acumulado_discreto(curva.funcao, N)
    integral_continua = curva.integral_continua(N)

    assert abs(soma_discreta - (integral_continua + b * N / 2)) < 1e-9


def test_integral_continua_fica_proxima_da_soma_discreta_em_termos_relativos():
    for curva in (curva_linear(), curva_quadratica(), curva_exponencial()):
        soma_discreta = xp_acumulado_discreto(curva.funcao, 50)
        integral_continua = curva.integral_continua(50)
        diferenca_relativa = abs(soma_discreta - integral_continua) / soma_discreta
        assert diferenca_relativa < 0.1

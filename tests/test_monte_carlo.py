from ealen_ia.atributos import Atributos
from ealen_ia.classes import GUARDIAO, LUMINAR, RACHADOR
from ealen_ia.monte_carlo import matriz_de_confrontos, matriz_de_confrontos_da_classe, rodar_confronto
from ealen_ia.personagem import Personagem


def test_rodar_confronto_conta_n_resultados():
    a = Personagem.da_classe("A", GUARDIAO)
    b = Personagem.da_classe("B", RACHADOR)

    resultado = rodar_confronto(a, b, n=50, seed=1)

    assert resultado.n == 50
    assert resultado.vitorias_a + resultado.vitorias_b + resultado.empates == 50


def test_rodar_confronto_mesmo_seed_e_deterministico():
    a1 = Personagem.da_classe("A", GUARDIAO)
    b1 = Personagem.da_classe("B", RACHADOR)
    a2 = Personagem.da_classe("A", GUARDIAO)
    b2 = Personagem.da_classe("B", RACHADOR)

    resultado1 = rodar_confronto(a1, b1, n=30, seed=7)
    resultado2 = rodar_confronto(a2, b2, n=30, seed=7)

    assert resultado1 == resultado2


def test_taxa_vitoria_a_e_erro():
    a = Personagem.da_classe("A", GUARDIAO)
    b = Personagem.da_classe("B", RACHADOR)

    resultado = rodar_confronto(a, b, n=100, seed=1)

    assert 0.0 <= resultado.taxa_vitoria_a <= 1.0
    assert resultado.erro(alvo=0.5) == abs(resultado.taxa_vitoria_a - 0.5)
    assert resultado.erro(alvo=resultado.taxa_vitoria_a) == 0.0


def test_matriz_de_confrontos_tem_cada_par_uma_unica_vez():
    classes = (LUMINAR, GUARDIAO, RACHADOR)

    matriz = matriz_de_confrontos(classes, n=10, seed=1)

    assert set(matriz.keys()) == {
        ("Luminar", "Luminar"),
        ("Luminar", "Guardião"),
        ("Luminar", "Rachador"),
        ("Guardião", "Guardião"),
        ("Guardião", "Rachador"),
        ("Rachador", "Rachador"),
    }
    for resultado in matriz.values():
        assert resultado.n == 10


def test_matriz_de_confrontos_apenas_espelhos_pula_pares_diferentes():
    classes = (LUMINAR, GUARDIAO, RACHADOR)

    matriz = matriz_de_confrontos(classes, n=10, seed=1, apenas_espelhos=True)

    assert set(matriz.keys()) == {
        ("Luminar", "Luminar"),
        ("Guardião", "Guardião"),
        ("Rachador", "Rachador"),
    }


def test_matriz_de_confrontos_espelho_e_proximo_de_50_por_cento():
    classes = (GUARDIAO,)

    matriz = matriz_de_confrontos(classes, n=500, seed=1)
    resultado = matriz[("Guardião", "Guardião")]

    assert resultado.erro(alvo=0.5) < 0.15


def test_matriz_de_confrontos_atributos_por_classe_none_reproduz_comportamento_padrao():
    classes = (LUMINAR, GUARDIAO, RACHADOR)

    matriz_padrao = matriz_de_confrontos(classes, n=20, seed=1)
    matriz_explicita = matriz_de_confrontos(classes, n=20, seed=1, atributos_por_classe=None)

    assert matriz_padrao == matriz_explicita


def test_matriz_de_confrontos_usa_atributos_por_classe_customizados():
    classes = (LUMINAR, GUARDIAO)
    # Luminar sem Dain/Eir/Nath (dano fraco, HP mínimo) contra o Guardião
    # padrão (bem mais forte e tanque): deve perder virtualmente sempre.
    atributos_por_classe = {"Luminar": Atributos(dain=0, eir=0, nath=0, il=6, or_=12)}

    matriz = matriz_de_confrontos(classes, n=30, seed=1, atributos_por_classe=atributos_por_classe)
    resultado = matriz[("Luminar", "Guardião")]

    assert resultado.vitorias_a == 0


def test_matriz_de_confrontos_da_classe_so_traz_confrontos_da_classe_alvo():
    classes = (LUMINAR, GUARDIAO, RACHADOR)

    matriz = matriz_de_confrontos_da_classe("Guardião", classes, n=10, seed=1)

    assert set(matriz.keys()) == {
        ("Luminar", "Guardião"),
        ("Guardião", "Guardião"),
        ("Guardião", "Rachador"),
    }
    assert all(resultado.n == 10 for resultado in matriz.values())


def test_matriz_de_confrontos_da_classe_e_consistente_com_matriz_completa():
    classes = (LUMINAR, GUARDIAO, RACHADOR)

    matriz_completa = matriz_de_confrontos(classes, n=15, seed=3)
    matriz_da_classe = matriz_de_confrontos_da_classe("Guardião", classes, n=15, seed=3)

    for chave in matriz_da_classe:
        assert matriz_da_classe[chave] == matriz_completa[chave]

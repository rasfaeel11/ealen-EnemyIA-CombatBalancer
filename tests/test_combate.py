from ealen_ia.classes import GUARDIAO, RACHADOR
from ealen_ia.combate import simulate_combat
from ealen_ia.personagem import Personagem


def _duelo():
    a = Personagem.da_classe("A", GUARDIAO)
    b = Personagem.da_classe("B", RACHADOR)
    return a, b


def test_mesmo_seed_produz_resultado_identico():
    a1, b1 = _duelo()
    a2, b2 = _duelo()

    resultado1 = simulate_combat(a1, b1, seed=42)
    resultado2 = simulate_combat(a2, b2, seed=42)

    assert resultado1.vencedor == resultado2.vencedor
    assert resultado1.historico == resultado2.historico


def test_seeds_diferentes_podem_gerar_resultados_diferentes():
    resultados = set()
    for seed in range(20):
        a, b = _duelo()
        resultado = simulate_combat(a, b, seed=seed)
        resultados.add(resultado.vencedor)

    assert len(resultados) > 1


def test_combate_tem_um_vencedor_com_hp_e_um_perdedor_sem_hp():
    a, b = _duelo()

    resultado = simulate_combat(a, b, seed=1)

    vencedor = a if a.nome == resultado.vencedor else b
    perdedor = b if vencedor is a else a
    assert vencedor.hp_atual > 0
    assert perdedor.hp_atual == 0


def test_turnos_alternam_entre_os_dois_lados():
    a, b = _duelo()

    resultado = simulate_combat(a, b, seed=7)

    atacantes = [turno.atacante for turno in resultado.historico]
    esperado = [a.nome if i % 2 == 0 else b.nome for i in range(len(atacantes))]
    assert atacantes == esperado


def test_max_turnos_zero_da_empate_sem_historico():
    a, b = _duelo()

    resultado = simulate_combat(a, b, seed=1, max_turnos=0)

    assert resultado.vencedor is None
    assert resultado.turnos_totais == 0

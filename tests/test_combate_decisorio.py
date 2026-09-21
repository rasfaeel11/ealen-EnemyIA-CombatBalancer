from ealen_ia.classes import GUARDIAO, RACHADOR
from ealen_ia.combate_decisorio import (
    ACOES,
    Acao,
    Estado,
    simulate_combat_decisorio,
)
from ealen_ia.personagem import Personagem


def _duelo():
    a = Personagem.da_classe("A", GUARDIAO)
    b = Personagem.da_classe("B", RACHADOR)
    return a, b


def _sempre(acao: Acao):
    def politica(estado: Estado) -> Acao:
        return acao

    return politica


def test_acoes_tem_exatamente_tres_opcoes():
    assert set(ACOES) == {Acao.ATACAR, Acao.DEFENDER, Acao.HABILIDADE}


def test_mesmo_seed_produz_resultado_identico():
    a1, b1 = _duelo()
    a2, b2 = _duelo()

    r1 = simulate_combat_decisorio(a1, b1, _sempre(Acao.ATACAR), _sempre(Acao.ATACAR), seed=42)
    r2 = simulate_combat_decisorio(a2, b2, _sempre(Acao.ATACAR), _sempre(Acao.ATACAR), seed=42)

    assert r1.vencedor == r2.vencedor
    assert r1.historico == r2.historico


def test_defender_nunca_causa_dano():
    a, b = _duelo()

    resultado = simulate_combat_decisorio(a, b, _sempre(Acao.DEFENDER), _sempre(Acao.ATACAR), seed=1)

    transicoes_de_a = [t for t in resultado.historico if t.quem_agiu == "A"]
    assert all(t.recompensa == 0.0 for t in transicoes_de_a)


def test_dois_lados_defendendo_sempre_empata():
    a, b = _duelo()

    resultado = simulate_combat_decisorio(a, b, _sempre(Acao.DEFENDER), _sempre(Acao.DEFENDER), seed=1, max_turnos=50)

    assert resultado.vencedor is None
    assert resultado.turnos_totais == 50


def test_defender_reduz_dano_esperado_recebido():
    # Rachador sempre ataca; Guardião ora sempre defende, ora sempre ataca de volta.
    # Em média, ao longo de várias seeds, o Guardião que se defende deve
    # tomar menos dano acumulado (ele nunca causa dano, então o que importa é
    # quantos turnos ele sobrevive / quanto dano leva antes de morrer).
    turnos_defendendo = []
    turnos_atacando = []
    for seed in range(30):
        a, b = _duelo()
        r_defende = simulate_combat_decisorio(a, b, _sempre(Acao.DEFENDER), _sempre(Acao.ATACAR), seed=seed)
        turnos_defendendo.append(r_defende.turnos_totais)

        a2, b2 = _duelo()
        r_ataca = simulate_combat_decisorio(a2, b2, _sempre(Acao.ATACAR), _sempre(Acao.ATACAR), seed=seed)
        turnos_atacando.append(r_ataca.turnos_totais)

    # Defendendo, o Guardião aguenta mais turnos em média (dano reduzido) do
    # que atacando de volta (que nem sempre reduz o número de turnos, já que
    # ele mesmo não estava perto de vencer de qualquer forma no cenário Guardião-vs-Rachador).
    assert sum(turnos_defendendo) / len(turnos_defendendo) > sum(turnos_atacando) / len(turnos_atacando)


def test_habilidade_causa_mais_dano_que_ataque_quando_acerta():
    a, b = _duelo()
    # Fixamos chance de crítico em 0 pra isolar o efeito do bônus de dano da habilidade.
    from dataclasses import replace

    a.parametros = replace(a.parametros, chance_critico_base=0.0, chance_critico_por_il=0.0)

    resultado = simulate_combat_decisorio(a, b, _sempre(Acao.HABILIDADE), _sempre(Acao.ATACAR), seed=5, max_turnos=1)
    transicao = resultado.historico[0]

    if transicao.recompensa > 0:  # só valida quando a habilidade realmente acertou nesse seed
        dano_normal = max(a.dano_fisico_base, a.dano_magico_base) / b.hp_maximo
        assert transicao.recompensa > dano_normal


def test_estado_discretiza_hp_em_faixas_validas():
    a, b = _duelo()
    a.hp_atual = a.hp_maximo  # hp cheio
    from ealen_ia.combate_decisorio import N_FAIXAS_DE_HP, observar_estado

    estado = observar_estado(a, b, False, False)
    assert 0 <= estado.hp_proprio_faixa < N_FAIXAS_DE_HP
    assert 0 <= estado.hp_oponente_faixa < N_FAIXAS_DE_HP
    assert estado.hp_proprio_faixa == N_FAIXAS_DE_HP - 1  # hp cheio cai na faixa mais alta

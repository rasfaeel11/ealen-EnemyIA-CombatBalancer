from ealen_ia.classes import GUARDIAO, RACHADOR
from ealen_ia.combate_decisorio import ACOES, Acao, Estado, TransicaoDecisoria
from ealen_ia.personagem import Personagem
from ealen_ia.q_learning import (
    atualizar_q,
    avaliar_taxa_vitoria,
    criar_tabela_q,
    melhor_acao,
    politica_sempre_ataca,
    treinar,
    valor_maximo,
)


def _duelo():
    a = Personagem.da_classe("A", GUARDIAO)
    b = Personagem.da_classe("B", RACHADOR)
    return a, b


def _estado(hp_proprio=2, hp_oponente=2, eu_guarda=False, oponente_guarda=False):
    return Estado(hp_proprio, hp_oponente, eu_guarda, oponente_guarda)


def test_tabela_q_comeca_zerada_pra_qualquer_estado_e_acao():
    tabela = criar_tabela_q()
    estado = _estado()
    assert tabela[(estado, Acao.ATACAR)] == 0.0


def test_melhor_acao_desempata_de_forma_deterministica_quando_tudo_zero():
    tabela = criar_tabela_q()
    estado = _estado()
    assert melhor_acao(tabela, estado) == ACOES[0]


def test_atualizar_q_nao_terminal_bellman_correto():
    tabela = criar_tabela_q()
    estado = _estado(2, 2)
    proximo_estado = _estado(2, 1)
    tabela[(proximo_estado, Acao.ATACAR)] = 0.5
    tabela[(proximo_estado, Acao.DEFENDER)] = 0.2

    transicao = TransicaoDecisoria("A", estado, Acao.ATACAR, recompensa=0.3, proximo_estado=proximo_estado)
    atualizar_q(tabela, transicao, alfa=0.5, gamma=0.9, terminal=False)

    alvo_esperado = 0.3 + 0.9 * 0.5  # max Q do próximo estado é 0.5 (ATACAR)
    valor_esperado = 0.0 + 0.5 * (alvo_esperado - 0.0)
    assert tabela[(estado, Acao.ATACAR)] == valor_esperado


def test_atualizar_q_terminal_ignora_proximo_estado():
    tabela = criar_tabela_q()
    estado = _estado(0, 2)
    proximo_estado = _estado(1, 2)  # precisa ser um estado DIFERENTE de `estado`, senão é a mesma chave no dict
    tabela[(proximo_estado, Acao.ATACAR)] = 999.0  # não deveria influenciar o alvo terminal

    transicao = TransicaoDecisoria("A", estado, Acao.ATACAR, recompensa=-1.0, proximo_estado=proximo_estado)
    atualizar_q(tabela, transicao, alfa=0.5, gamma=0.9, terminal=True)

    valor_esperado = 0.0 + 0.5 * (-1.0 - 0.0)
    assert tabela[(estado, Acao.ATACAR)] == valor_esperado


def test_valor_maximo_pega_o_maior_entre_as_acoes():
    tabela = criar_tabela_q()
    estado = _estado()
    tabela[(estado, Acao.ATACAR)] = 0.1
    tabela[(estado, Acao.DEFENDER)] = 0.9
    tabela[(estado, Acao.HABILIDADE)] = 0.3

    assert valor_maximo(tabela, estado) == 0.9


def test_avaliar_taxa_vitoria_fica_entre_0_e_1():
    a, b = _duelo()
    tabela = criar_tabela_q()

    taxa = avaliar_taxa_vitoria(tabela, a, b, politica_sempre_ataca, n_combates=10, seed=1)

    assert 0.0 <= taxa <= 1.0


def test_treinar_roda_sem_erro_e_produz_historico():
    a, b = _duelo()

    tabela_q, historico = treinar(
        pares_de_treino=[(a, b)],
        n_episodios=200,
        n_checkpoints=4,
        seed=1,
    )

    assert len(historico) == 4
    assert len(tabela_q) > 0
    assert all(0.0 <= taxa <= 1.0 for taxa in historico)


def test_treinar_e_deterministico_com_mesmo_seed():
    a1, b1 = _duelo()
    a2, b2 = _duelo()

    tabela1, historico1 = treinar(pares_de_treino=[(a1, b1)], n_episodios=100, n_checkpoints=2, seed=7)
    tabela2, historico2 = treinar(pares_de_treino=[(a2, b2)], n_episodios=100, n_checkpoints=2, seed=7)

    assert dict(tabela1) == dict(tabela2)
    assert historico1 == historico2


def test_agente_treinado_aprende_a_vencer_mais_que_sempre_atacar():
    a, b = _duelo()

    tabela_q, _ = treinar(pares_de_treino=[(a, b)], n_episodios=6000, n_checkpoints=1, seed=1)

    taxa_final = avaliar_taxa_vitoria(tabela_q, a, b, politica_sempre_ataca, n_combates=200, seed=99)

    # o agente joga como "A" com política gulosa (guarda + habilidade
    # disponíveis) contra um oponente que só ataca — deve ganhar bem mais
    # que os ~50% que "A sempre atacar" já teria contra "B sempre atacar"
    # nesse confronto específico (Guardião tem vantagem natural sobre
    # Rachador nesse combate automático, ver Camada 1).
    assert taxa_final > 0.6

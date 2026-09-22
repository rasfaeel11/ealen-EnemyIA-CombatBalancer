from ealen_ia.classes import GUARDIAO, RACHADOR
from ealen_ia.combate_decisorio import Acao, Estado, TransicaoDecisoria
from ealen_ia.personagem import Personagem
from ealen_ia.q_learning import politica_sempre_ataca
from ealen_ia.q_linear import (
    N_FEATURES,
    atualizar_pesos,
    avaliar_taxa_vitoria_linear,
    criar_pesos,
    features,
    melhor_acao_aproximada,
    treinar_linear,
    valor_aproximado,
    valor_maximo_aproximado,
)


def _duelo():
    a = Personagem.da_classe("A", GUARDIAO)
    b = Personagem.da_classe("B", RACHADOR)
    return a, b


def _estado(hp_proprio=2, hp_oponente=2, eu_guarda=False, oponente_guarda=False):
    return Estado(hp_proprio, hp_oponente, eu_guarda, oponente_guarda)


def test_pesos_comecam_zerados():
    pesos = criar_pesos()
    assert all(w == 0.0 for vetor in pesos.values() for w in vetor)


def test_valor_aproximado_e_zero_com_pesos_zerados():
    pesos = criar_pesos()
    assert valor_aproximado(pesos, _estado(), Acao.ATACAR) == 0.0


def test_features_tem_o_tamanho_esperado_e_bias_fixo():
    vetor = features(_estado(hp_proprio=4, hp_oponente=0, eu_guarda=True, oponente_guarda=False))
    assert len(vetor) == N_FEATURES
    assert vetor[0] == 1.0  # bias
    assert vetor[1] == 1.0  # hp_proprio_faixa=4 normalizado -> 4/4
    assert vetor[2] == 0.0  # hp_oponente_faixa=0 normalizado -> 0/4
    assert vetor[3] == 1.0  # eu_com_guarda
    assert vetor[4] == 0.0  # oponente_com_guarda


def test_atualizar_pesos_move_na_direcao_que_reduz_o_erro_td():
    pesos = criar_pesos()
    estado = _estado(2, 2)
    proximo_estado = _estado(2, 1)

    transicao = TransicaoDecisoria("A", estado, Acao.ATACAR, recompensa=1.0, proximo_estado=proximo_estado)
    valor_antes = valor_aproximado(pesos, estado, Acao.ATACAR)
    atualizar_pesos(pesos, transicao, alfa=0.1, gamma=0.9, terminal=True)
    valor_depois = valor_aproximado(pesos, estado, Acao.ATACAR)

    # alvo era 1.0 (terminal), valor antes era 0.0 -> o valor aproximado
    # depois do passo deve ter se movido em direção a 1.0.
    assert valor_depois > valor_antes


def test_valor_maximo_aproximado_pega_o_maior_entre_as_acoes():
    pesos = criar_pesos()
    estado = _estado()
    pesos[Acao.DEFENDER][0] = 5.0  # via bias, Q(estado, DEFENDER) = 5.0 pra qualquer estado

    assert valor_maximo_aproximado(pesos, estado) == 5.0
    assert melhor_acao_aproximada(pesos, estado) == Acao.DEFENDER


def test_avaliar_taxa_vitoria_linear_fica_entre_0_e_1():
    a, b = _duelo()
    pesos = criar_pesos()

    taxa = avaliar_taxa_vitoria_linear(pesos, a, b, politica_sempre_ataca, n_combates=10, seed=1)

    assert 0.0 <= taxa <= 1.0


def test_treinar_linear_roda_sem_erro_e_produz_historico():
    a, b = _duelo()

    pesos, historico = treinar_linear(pares_de_treino=[(a, b)], n_episodios=200, n_checkpoints=4, seed=1)

    assert len(historico) == 4
    assert all(0.0 <= taxa <= 1.0 for taxa in historico)


def test_treinar_linear_e_deterministico_com_mesmo_seed():
    a1, b1 = _duelo()
    a2, b2 = _duelo()

    pesos1, historico1 = treinar_linear(pares_de_treino=[(a1, b1)], n_episodios=100, n_checkpoints=2, seed=7)
    pesos2, historico2 = treinar_linear(pares_de_treino=[(a2, b2)], n_episodios=100, n_checkpoints=2, seed=7)

    assert pesos1 == pesos2
    assert historico1 == historico2


def test_agente_linear_aprende_em_media_mas_e_instavel_entre_seeds():
    """Achado real ao testar: ao contrário da tabela (`test_q_learning.py`),
    o agente linear NÃO converge de forma confiável — rodando o mesmo treino
    com seeds diferentes, a taxa de vitória final varia de ~0% a ~90%. Isso é
    a "tríade mortal" do RL (bootstrap + operador max + aproximação de
    função): sem as garantias de convergência da tabela, o gradiente
    descendente sobre os pesos pode oscilar ou convergir pra um ótimo local
    ruim dependendo da trajetória de treino. Por isso o teste mede a MÉDIA
    entre várias sementes (ainda melhor que o acaso), em vez de exigir que
    uma única semente específica funcione — um teste de semente única seria
    instável (flaky) por natureza, não por erro de implementação.
    """
    a, b = _duelo()

    taxas = []
    for seed in range(4):
        pesos, _ = treinar_linear(pares_de_treino=[(a, b)], n_episodios=8000, n_checkpoints=1, alfa=0.02, seed=seed)
        taxas.append(avaliar_taxa_vitoria_linear(pesos, a, b, politica_sempre_ataca, n_combates=100, seed=99))

    media = sum(taxas) / len(taxas)
    assert media > 0.4

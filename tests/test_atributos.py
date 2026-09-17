from ealen_ia.atributos import ParametrosDeFormula, hp_maximo
from ealen_ia.classes import TODAS_AS_CLASSES
from ealen_ia.personagem import Personagem


def test_cada_classe_tem_hp_dano_e_chances_coerentes():
    p = ParametrosDeFormula()
    for classe in TODAS_AS_CLASSES:
        personagem = Personagem.da_classe(classe.nome, classe)

        assert personagem.hp_maximo == hp_maximo(classe.atributos_base, p)
        assert personagem.hp_atual == personagem.hp_maximo
        assert personagem.dano_fisico_base > 0
        assert 0.0 <= personagem.chance_de_acerto <= 1.0
        assert 0.0 <= personagem.chance_de_critico <= 1.0
        assert 0.0 <= personagem.reducao_de_dano <= p.reducao_dano_maxima


def test_receber_dano_aplica_reducao_de_armadura():
    p = ParametrosDeFormula()
    tanque = Personagem.da_classe("Tanque de teste", TODAS_AS_CLASSES[0])

    dano_efetivo = tanque.receber_dano(100.0)

    assert dano_efetivo == 100.0 * (1 - tanque.reducao_de_dano)
    assert tanque.hp_atual == tanque.hp_maximo - dano_efetivo


def test_hp_nao_fica_negativo():
    personagem = Personagem.da_classe("x", TODAS_AS_CLASSES[0])

    personagem.receber_dano(personagem.hp_maximo * 10)

    assert personagem.hp_atual == 0.0
    assert not personagem.esta_vivo

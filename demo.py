from ealen_ia.classes import TODAS_AS_CLASSES
from ealen_ia.personagem import Personagem

print(f"{'Classe':<18}{'HP':>7}{'DanoFis':>9}{'DanoMag':>9}{'Acerto':>9}{'Crit':>7}{'Reducao':>9}")
for classe in TODAS_AS_CLASSES:
    p = Personagem.da_classe(classe.nome, classe)
    print(
        f"{classe.nome:<18}{p.hp_maximo:>7.0f}{p.dano_fisico_base:>9.1f}{p.dano_magico_base:>9.1f}"
        f"{p.chance_de_acerto:>9.0%}{p.chance_de_critico:>7.0%}{p.reducao_de_dano:>9.0%}"
    )

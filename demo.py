from ealen_ia.classes import GUARDIAO, RACHADOR, TODAS_AS_CLASSES
from ealen_ia.combate import simulate_combat
from ealen_ia.personagem import Personagem

print(f"{'Classe':<18}{'HP':>7}{'DanoFis':>9}{'DanoMag':>9}{'Acerto':>9}{'Crit':>7}{'Reducao':>9}")
for classe in TODAS_AS_CLASSES:
    p = Personagem.da_classe(classe.nome, classe)
    print(
        f"{classe.nome:<18}{p.hp_maximo:>7.0f}{p.dano_fisico_base:>9.1f}{p.dano_magico_base:>9.1f}"
        f"{p.chance_de_acerto:>9.0%}{p.chance_de_critico:>7.0%}{p.reducao_de_dano:>9.0%}"
    )

print("\n--- Guardião vs Rachador (seed=1) ---")
guardiao = Personagem.da_classe("Guardião", GUARDIAO)
rachador = Personagem.da_classe("Rachador", RACHADOR)
resultado = simulate_combat(guardiao, rachador, seed=1)

for turno in resultado.historico:
    if not turno.acertou:
        print(f"turno {turno.numero:>2}: {turno.atacante} errou o ataque")
    else:
        marca_critico = " (CRÍTICO)" if turno.critico else ""
        print(f"turno {turno.numero:>2}: {turno.atacante} acertou {turno.alvo} causando {turno.dano_causado:.1f}{marca_critico}")

print(f"\nVencedor: {resultado.vencedor} em {resultado.turnos_totais} turnos")

# AGENTS.md — Balanceador de Combate (Monte Carlo + Gradiente Descendente) + IA de Inimigo (Q-Learning)

## Projeto relacionado (repo separado, sem dependência de código)

Existe um segundo projeto, "Eälen: O Canto das Primeiras Luzes" (RPG de navegador, React + Node + Supabase), em outro repositório. Este projeto aqui é **standalone em Python** e reaproveita apenas conceitualmente as regras de combate daquele jogo (classes, atributos, fórmulas de dano/acerto) — **não depende do código dele**, nada de frontend, banco de dados ou infraestrutura web aqui. Se algum dia os parâmetros balanceados por este projeto forem incorporados ao jogo, isso vira um export simples (ex: JSON com os valores finais) consumido pelo outro repo — não assuma essa integração agora.

**O que este projeto NÃO é:** não é o jogo. Não existe objetivo de shippar uma feature jogável. O produto final são duas ferramentas de análise/IA e os artefatos que elas geram (parâmetros balanceados, gráficos de convergência, um agente treinado).

**Motivação pessoal (importante pro agente entender o nível de profundidade esperado):** o autor está cursando Cálculo 1 e quer um projeto que reforce derivada como taxa de variação e otimização por gradiente de forma aplicada — **não** é pra escalar pra cálculo multivariável, autograd ou deep learning. Se a implementação natural de algo pedir uma dessas coisas, prefira a versão mais simples e explícita (ex: diferença finita em vez de autograd), mesmo que menos "correta" academicamente. Simplicidade e capacidade de explicar o próprio código pesam mais que sofisticação aqui.

## Referência: atributos do jogo (usados como base do simulador)

| Atributo | Runa | Governa |
|---|---|---|
| Dain | Força | Dano corpo-a-corpo |
| Eir | Ressonância/Espírito | Dano/cura mágica |
| Nath | Vitalidade | HP máximo |
| Il | Percepção | Precisão / chance de crítico |
| Or | Densidade | Defesa/armadura |
| Len | Som/Voz | Carisma (não usado no combate) |
| Ul | Mistério | Sabedoria/Intelecto (não usado no combate) |

**Classes:** Luminar (Tank/Suporte), Entropista (Debuffer), Cantor de Eälen (Controle), Guardião (Tank Ofensivo), Sombrílico (Anti-Mago), Rachador (Sniper Físico)

## Non-goals (fora do escopo, não sugira expandir pra isso sem eu pedir)

- Sem frontend/UI visual — scripts e/ou notebook bastam.
- Sem redes neurais multi-camada / deep learning / autograd (PyTorch, TensorFlow).
- Sem integração com o código do jogo web — o simulador de combate aqui é uma reimplementação isolada em Python das regras, não um import do backend TS.
- Sem cálculo multivariável — toda derivada aqui é de uma variável por vez (diferença finita 1D).

## Stack

- Python 3
- NumPy (simulação, cálculo numérico)
- Matplotlib (gráficos de convergência e curvas de XP)
- Scripts ou Jupyter notebook — sem framework de ML pesado

## Arquitetura em camadas

```
Camada 0: Simulador de combate (base, compartilhada pelos dois módulos)
Camada 1: Monte Carlo + auto-tuner via gradiente descendente (balanceamento)
Camada 2: Agente de IA via Q-learning (inimigo competente)
Camada 3 (opcional): Modelagem da curva de XP/progressão (derivada/integral)
```

Camada 2 depende da Camada 0, não da Camada 1 — mas usar classes já balanceadas pela Camada 1 pra treinar o agente da Camada 2 dá um sinal de treino mais realista. Construa a Camada 0 uma vez só e reaproveite nas outras duas.

## Camada 0 — Simulador de combate

**Objetivo:** modelar em código as classes/atributos do jogo (dano, HP, chance de acerto, etc. — ver tabela acima) e implementar um simulador de combate 1x1 simplificado que serve de base pros dois módulos.

**Definição de pronto:** dado dois builds de personagem (classe + atributos), a função `simulate_combat(a, b, seed=None)` roda um combate completo turno a turno e retorna o vencedor + histórico de dano trocado. Determinístico se `seed` for passado (pra debug), estocástico se não.

## Camada 1 — Balanceamento: Monte Carlo + Gradiente Descendente

**Objetivo:** ajustar automaticamente os parâmetros numéricos das classes até que os confrontos fiquem equilibrados (taxa de vitória ~50/50 em confrontos que deveriam ser simétricos, ou dentro de uma faixa-alvo pra confrontos assimétricos por design).

**Como funciona:**
1. **Monte Carlo:** rodar N combates simulados (Camada 0) entre pares de classes/builds, registrando vencedor de cada um.
2. **Métrica de desequilíbrio:** com as N simulações, calcular taxa de vitória por confronto. Erro = `|taxa_de_vitoria - alvo|` (alvo = 0.5 pra confrontos simétricos).
3. **Auto-tuner via gradiente descendente numérico:** como não há fórmula fechada pro erro (é estocástico), estimar a derivada por diferença finita — perturbar um parâmetro (ex: dano da classe X) por um delta pequeno, rodar novas simulações, medir a variação do erro. Isso *é* a derivada na prática. Ajustar o parâmetro na direção que reduz o erro, repetir até convergir (ou até um número máximo de iterações).

**Definição de pronto:** um conjunto de parâmetros por classe que resulta em confrontos equilibrados, mais um gráfico (Matplotlib) da convergência do erro ao longo das iterações.

**Extensão opcional desta camada:** modelar a curva de progressão de XP/nível — decidir se o crescimento é linear, quadrático ou exponencial (derivada = velocidade de crescimento), e calcular XP total acumulado até o nível N como integral (área sob a curva) da função de XP por nível. Trate isso como um script separado, não misture com o auto-tuner de combate.

## Camada 2 — IA de inimigo via Q-Learning

**Objetivo:** treinar um agente que joga o combate por turnos de forma competente, servindo como IA de inimigo (ou "playtester automático" pra validar o balanceamento da Camada 1).

**Como funciona:**
1. Modelar o combate como problema de decisão: **estado** (HP atual próprio e do oponente, turno, status ativos), **ações** (atacar, defender, usar habilidade), **recompensa** (dano causado, vitória = recompensa grande, derrota = penalidade).
2. Implementar **Q-learning tabular**: tabela Q(estado, ação) atualizada após cada combate simulado via equação de Bellman (recompensa imediata + recompensa futura estimada, ponderada).
3. Treinar jogando muitos combates simulados (reaproveitando o simulador da Camada 0) até a política convergir.

**Definição de pronto:** um agente capaz de jogar contra builds diferentes, com taxa de vitória registrada — se o agente vence fácil demais contra uma classe específica, isso é sinal de desbalanceamento pra realimentar a Camada 1.

**Extensão fora do escopo inicial (só mencionar como próximo passo possível, não implementar sem eu pedir):** evoluir a tabela Q pra uma função de valor aproximada linearmente — isso introduz gradiente descendente "de verdade" sobre pesos lineares, mas é deliberadamente adiado.

## Ordem de implementação sugerida

1. Modelar classes/atributos do jogo em Python (dano, HP, chance de acerto etc.)
2. Implementar o simulador de combate (Camada 0)
3. Implementar o runner de Monte Carlo + métrica de desequilíbrio
4. Implementar o auto-tuner via gradiente descendente numérico
5. Implementar o agente de Q-learning reaproveitando o simulador
6. (Opcional) Módulo de curva de XP/progressão com derivada/integral

Cada etapa deve ser testável isoladamente antes de seguir pra próxima — não avance pra Camada 2 sem a Camada 0 e a Camada 1 rodando de ponta a ponta.

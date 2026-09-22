# Integração com o jogo Eälen — o que dá e o que não dá pra fazer hoje

Este documento existe porque o `AGENTS.md` deste projeto trata a integração
com o jogo ("Eälen: O Canto das Primeiras Luzes", repo separado, React +
Node + Supabase) como algo hipotético — "se algum dia os parâmetros
balanceados por este projeto forem incorporados ao jogo, isso vira um
export simples... não assuma essa integração agora". Explorei o código do
jogo (repo em `../ealen`, só leitura, nada foi escrito lá) pra descobrir
concretamente o que essa integração exigiria. O resumo: menos do que se
imaginava — os dois sistemas de combate são estruturalmente diferentes, não
a mesma fórmula com coeficientes trocados.

## O que foi explorado

- `server/src/combat/engine.ts` — o motor de combate real.
- `shared/types/characterClass.ts` — definição das 6 classes (`CLASS_INFO`).
- `shared/types/attributes.ts` — os 7 atributos (mesmos nomes, `or`/`len` em
  vez de `or_`/`len_` do Python, mas o mesmo sistema Tirán).
- `server/src/combat/leveling.ts` — a curva de XP/nível real.
- `server/src/routes/characters.ts` — criação de personagem.
- `shared/mock/bestiary.ts` — os inimigos do jogo.

## Por que o combate/balanceamento não integra direto

**O motor de combate real é outro sistema, não uma variação do nosso.**
`resolveCombatTurn` (engine.ts) rola d20 pra acerto (`d20 + atributo` vs.
`10 + Or do alvo`), tem 6 ações (`attack`, `quick_attack`, `heavy_attack`,
`defend`, `heal`, `use_item`), iniciativa por Il, crítico em 20 natural,
falha crítica em 1 natural (com penalidade de Or até o próximo turno),
bloqueio dinâmico e buffs de item. O simulador da Camada 0 usa fórmulas de
probabilidade direta (`chance_acerto_base + chance_acerto_por_il * il`, sem
d20 nenhum). Não existe um "slot" no motor real onde um valor de
`ParametrosDeFormula` (ex: `hp_por_nath`, `reducao_dano_por_or`) encaixe —
são parametrizações de fórmulas diferentes, não a mesma fórmula com números
diferentes.

**As classes não têm atributos-base fixos no jogo real.** `ClassInfo` (em
`characterClass.ts`) só define `primaryAttributes` — quais atributos sobem
mais rápido ao subir de nível (ver `leveling.ts`: +2 por nível nos atributos
primários, +1 nos demais). Os valores iniciais são escolhidos livremente
pelo jogador na criação do personagem (`POST /characters` recebe
`attributes` no corpo da requisição, validado só quanto ao formato). Não
existe "o Guardião base" contra o qual comparar o resultado do nosso
auto-tuner — a Camada 1 balanceia um conceito (atributos-base fixos por
classe) que o jogo não usa.

**Os inimigos do bestiário são autorais, não sistêmicos.** Cada entrada de
`shared/mock/bestiary.ts` tem nome, lore e atributos escolhidos à mão junto
com "Arts" nomeadas (ex: o Lobo-de-Bruma tem "Bote Silencioso", "Dentada",
"Investida Cega"). Sobrescrever esses números com a saída de um auto-tuner
numérico destruiria decisões de design e narrativa que não são só
"estatísticas de combate equilibradas" — descartaria justamente aquilo que
o próprio jogo trata como seu diferencial (ver `LORE.md` do outro repo, ou o
comentário em `bestiary.ts`: "nada aqui é monstro por natureza... cada uma
tem nomes próprios de ação").

## O que precisaria mudar pra uma integração de verdade ser possível

1. Reimplementar o simulador da Camada 0 usando o mesmo modelo de combate
   do jogo (d20, 6 ações, iniciativa, bloqueio) — bem mais trabalho que o
   simulador atual, e sairia dos non-goals do `AGENTS.md` (sem UI/frontend
   aqui, mas o MODELO de combate em si precisaria mudar).
2. Decidir se o jogo passaria a ter atributos-base fixos por classe (mudança
   de design do jogo, não deste projeto) — ou adaptar a Camada 1 pra
   balancear outra coisa (ex: os multiplicadores de `TO_HIT_MODIFIER`, o
   dado de dano, os bônus de `heavy_attack`).
3. Tratar os inimigos do bestiário como um caso à parte — talvez o
   auto-tuner sugira uma FAIXA de atributos "equilibrada" pro nível de um
   inimigo, e um humano escolhe os valores finais dentro dela preservando a
   caracterização narrativa, em vez de sobrescrever automaticamente.

Nenhum desses pontos foi implementado — são só o mapeamento do que a
integração exigiria, pra não ficar vago da próxima vez que a pergunta surgir.

## O que integra de verdade, hoje: a curva de XP

`xpToNextLevel(level) = level * 100` (`leveling.ts`) é uma curva **linear**
— exatamente o que `progressao_xp.curva_linear` já modela, com `a=0, b=100`.
É a única estrutura numérica do jogo que bate 1:1 com algo que a Camada 3
já sabe analisar, então é o único ponto onde rodei uma análise de verdade:
`demo_progressao_xp_real.py` calcula a derivada dessa curva real (constante,
100 XP por nível — faz sentido, é linear) e o XP total acumulado até os
níveis 10/25/50 (5.500 / 32.500 / 127.500), e compara visualmente com duas
alternativas hipotéticas (quadrática, exponencial) — não como uma proposta
de mudança, só como contraste didático de como a progressão ficaria
diferente com outro formato de curva.

Esse script só lê a fórmula (copiada aqui como constante, não importada do
outro repo) — nenhum código foi escrito ou alterado em `../ealen`.

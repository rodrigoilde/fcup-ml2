# Checkpoint — 5 nov 2026 (notas internas)

Não há apresentação formal: o professor pergunta o que fizemos e nós explicamos. Levar o `main.ipynb` aberto
(já executado) e este ficheiro. São ~2 min por membro e depois perguntas. **Cada um deve conseguir responder a
qualquer pergunta da lista, não só às da sua parte.**

---

## Guião

### Membro A: dados e protocolo (§1–3 do main)

- 942 pacientes: 74% *Absent*, 19% *Present*, 7% *Unknown*. Sempre-*Present* já dá WA 0,499: é a nossa linha de base.
- **Achado 1:** 70 crianças aparecem com dois `Patient ID` (uma visita em cada campanha, ligadas pelo
  `Additional ID`). Dividir por `Patient ID` deixaria passar informação entre treino e teste → agrupamos pelo sujeito.
- Protocolo: teste de 20% fechado (189 pacientes) + CV de 5 folds no resto, estratificados por sopro × campanha.
  A exploração do áudio foi feita **só** no desenvolvimento.
- Métrica implementada e testada contra a fórmula do enunciado (8 testes). As linhas da matriz são as previsões, como
  na Tabela 1 (ao contrário do sklearn).
- **Achado 2:** em 54 de 143 *Present*, o sopro não se ouve em todos os locais → rótulos por gravação conforme o local.
- **Achado 3:** na gravação inteira, *Present* e *Absent* quase não se distinguem no espectro; dentro da sístole,
  *Present* tem 2,5–5× mais potência entre 25 e 800 Hz → bandas até 800 Hz e modelação da fase cardíaca.
- **Achado 4:** os anotadores não conseguiram segmentar 69% do tempo das gravações *Unknown* (contra ~43% nas
  outras) → "*Unknown*" é sobretudo uma questão de qualidade do sinal.

### Membro B: CNN 1D (§4.1, §5)

- Entrada: 2 kHz, 5 bandas (25–45, 45–80, 80–200, 200–400, 400–800 Hz) como 5 canais, normalização robusta,
  janelas de 4 s. As bandas vêm de Potes et al. (vencedor da PhysioNet 2016) e foram alargadas até 800 Hz.
- 5 blocos Conv1D-BN-ReLU-MaxPool, campo recetivo ≈ 0,92 s (≈ um ciclo; a mediana é 0,58 s), GAP, dropout.
- Early stopping no **WA por paciente** do fold de validação (callback próprio), não na `val_loss`.
- Duas configurações para o desequilíbrio de classes:
  - E1: pesos de classe + argmax → **0,779 ± 0,025**;
  - E2: CE sem pesos + máximo sobre os locais + regra de Bayes `argmax(5·pP, 3·pU, 1·pA)` → **0,821 ± 0,025**,
    melhor nos 5 folds.
- O que interpretámos: a regra de decisão é o que mais pesa (E2 com argmax: 0,693). Pesos na perda + Bayes conta os
  custos duas vezes (0,54–0,57). Os pesos de classe deixam o treino instável.
- Fraqueza: *Unknown* (6/54 em E2). Os números são **otimistas** porque a época é escolhida no mesmo fold.

### Membro C: BiGRU multitarefa (§4.2, §6)

- Entrada: log-mel com 32 bandas de 25 a 800 Hz, frames de 20 ms. 2×BiGRU(64) com duas cabeças:
  - **segmentação** (estado de cada frame: não anotado, S1, sístole, S2, diástole), treinada com os .tsv;
  - **sopro**, com *attention pooling*.
- Porquê: os sopros estão na sístole (achado 3) e o *Unknown* tem a ver com o sinal ser ou não segmentável
  (achado 4). Os .tsv são só alvos de treino, por isso não fazem falta na inferência.
- *Smoke run* (fold 0, 10 épocas): a cabeça de segmentação chega a 0,69 de exatidão por frame (a classe maioritária
  dá 0,47) e reproduz o ciclo; o WA do fold 0 é 0,806. É só um fold, não é uma comparação com a CNN.
- A seguir: CV completa e **ablação λ = 0 vs λ > 0**, para saber se a cabeça auxiliar ajuda de facto.

### Fecho (qualquer membro)

Próximas semanas: *Unknown* (limiares, sinal de qualidade), afinação da CNN, CV da RNN, ensemble dos folds,
avaliação única no teste, script para os dados ocultos. O plano está em `docs/PLANO.md`.

---

## Perguntas prováveis

**Porque não dividiram só por `Patient ID`?** Porque 70 crianças têm dois IDs (uma visita em cada campanha). Com a
divisão por ID, o mesmo coração podia ficar no treino e no teste. Ligámos os IDs com union-find (`src/data.py`).

**Usaram o conjunto de teste para alguma coisa?** Não. Não está carregado em nenhuma secção de modelos, e o `main` tem
um `assert`. Até a exploração do áudio que justificou o pré-processamento foi feita só no desenvolvimento.

**Como obtêm uma previsão por paciente?** Cada gravação recebe a média das probabilidades das suas janelas. Para o
paciente, P(Present) = máximo sobre as gravações (basta um local com sopro) e P(Unknown) e P(Absent) = médias; depois
renormalizamos. A decisão é `argmax(w_c · p_c)`, com w = (5, 3, 1).

**Porque é que essa regra é a certa?** O denominador do WA não depende das previsões. Acertar um paciente da classe c
vale w_c, logo o ganho esperado de prever c é w_c · p_c. Com probabilidades calibradas, maximizar isso é ótimo.

**Então porque não usar também pesos de classe?** Porque os custos seriam contados duas vezes. Testámos: E1 + Bayes cai
para ~0,55, porque quase tudo passa a ser *Present*.

**0,82 não é bom demais?** É um valor de validação: a época de paragem é escolhida no mesmo fold, e as curvas são
ruidosas, por isso é otimista. A estimativa honesta vai ser o ensemble dos folds avaliado uma vez no teste. Não
comparamos com o Challenge, cujo teste era outro.

**Porque CNN 1D e não 2D sobre espectrograma?** As bandas cobrem exatamente a zona onde o espectro da sístole difere, e
a convolução 1D mantém a resolução temporal fina de S1/S2. Há precedente (Potes et al. 2016). A representação
tempo-frequência fica para a RNN (log-mel).

**Porque GRU e não LSTM? E porque bidirecional?** Temos ~2000 gravações de treino por fold: a GRU tem menos parâmetros.
É bidirecional porque classificamos gravações completas (offline): saber o que vem a seguir ajuda a localizar S1/S2.

**E se os dados ocultos não tiverem .tsv?** Nenhum modelo precisa deles na inferência. Na RNN, os .tsv são apenas alvos
da cabeça auxiliar durante o treino.

**Como lidam com durações diferentes?** Cortes aleatórios de 4 s (CNN) ou 5 s (RNN) no treino; janelas com 50% de
sobreposição na inferência, com média por gravação. Todas as gravações têm mais de 5 s, por isso nunca há *padding*.

**Porque early stopping no WA e não na `val_loss`?** A `val_loss` é calculada por corte e com rótulos por gravação; o
que nos avaliam é o WA por paciente. Implementámos um callback que calcula o WA no fim de cada época.

**Qual é a maior fraqueza agora?** A classe *Unknown*: a E2 só reconhece 6 de 54. É a prioridade das próximas semanas.

**Usaram IA generativa?** Responder com verdade e de forma concreta: sim, como apoio, e cada decisão está justificada
com evidência tirada dos próprios dados (secção 3 e registo de decisões no PLANO). Cada um de nós sabe explicar o código.

---

## Perguntas a fazer ao professor

1. Os dados ocultos vão trazer os ficheiros `<paciente>.txt` e/ou os `.tsv`, ou só os `.wav`?
2. Quando são libertados?
3. Reportar a CV com early stopping no próprio fold, assinalando que é otimista, é aceitável? Ou preferem que
   reportemos só o teste?

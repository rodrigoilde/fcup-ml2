# Checkpoint de 5 nov 2026 (notas internas)

Não há apresentação formal: o professor pergunta o que fizemos e nós explicamos. Levamos o `main.ipynb` aberto (já
executado) e este ficheiro. São uns 2 min por pessoa e depois perguntas. Cada um deve conseguir responder a qualquer
pergunta da lista, não só às da sua parte.

## Guião

### Membro A: dados e protocolo (secções 1 a 3 do main)

- 942 pacientes: 74% Absent, 19% Present, 7% Unknown. Prever sempre Present já dá WA 0,499, por isso é essa a nossa
  referência.
- 70 crianças aparecem com dois `Patient ID` (uma visita em cada campanha, ligadas pelo `Additional ID`). Se
  dividíssemos por `Patient ID` passava informação entre treino e teste, por isso agrupamos por criança.
- Protocolo: teste de 20% fechado (189 pacientes) + CV de 5 folds no resto, estratificados por sopro x campanha. A
  exploração do áudio foi feita só no desenvolvimento.
- A métrica está implementada e testada contra a fórmula do enunciado (8 testes). As linhas da matriz são as
  previsões, como na Tabela 1 (o sklearn faz ao contrário).
- Em 54 de 143 Present o sopro não se ouve em todos os locais, por isso os rótulos das gravações dependem do local.
- Na gravação inteira, Present e Absent quase não se distinguem no espectro. Dentro da sístole, as Present têm 2,5 a
  5 vezes mais potência entre 25 e 800 Hz. Daí as bandas até 800 Hz e a ideia de modelar a fase do ciclo.
- Os anotadores não conseguiram segmentar 69% do tempo das gravações Unknown (contra ~43% nas outras). Ou seja, o
  Unknown é sobretudo uma questão de qualidade do sinal.

### Membro B: CNN 1D (secções 4.1 e 5)

- Entrada: 2 kHz, 5 bandas (25-45, 45-80, 80-200, 200-400 e 400-800 Hz) como 5 canais, normalização robusta, janelas
  de 4 s. As bandas vêm de Potes et al. (vencedores da PhysioNet 2016), alargadas até 800 Hz.
- 5 blocos Conv1D-BN-ReLU-MaxPool, receptive field de ~0,92 s (mais ou menos um ciclo; a mediana é 0,58 s), GAP,
  dropout.
- Early stopping no WA por paciente do fold de validação (callback nosso), não na `val_loss`.
- Duas configurações para o desequilíbrio das classes:
  - E1: class weights + argmax, WA de 0,779 ± 0,025;
  - E2: CE sem pesos + máximo sobre os locais + regra de Bayes (`argmax(5 pP, 3 pU, 1 pA)`), WA de 0,821 ± 0,025,
    melhor nos 5 folds.
- O que tirámos daqui: a regra de decisão é o que mais pesa (a E2 com argmax dá 0,693). Pesos na loss + Bayes conta
  os custos duas vezes (0,54-0,57). Os class weights deixam o treino instável.
- Ponto fraco: Unknown (6 de 54 na E2). Os números são otimistas porque a época é escolhida no mesmo fold.

### Membro C: BiGRU multitarefa (secções 4.2 e 6)

- Entrada: log-mel com 32 bandas de 25 a 800 Hz, frames de 20 ms. Duas camadas BiGRU(64) com duas saídas:
  - segmentação (estado de cada frame: não anotado, S1, sístole, S2, diástole), treinada com os .tsv;
  - sopro, com attention pooling.
- Porquê: os sopros estão na sístole e o Unknown tem a ver com o sinal ser ou não segmentável (ver os dois últimos
  pontos do membro A). Os .tsv são só alvos de treino, por isso não fazem falta na inferência.
- Smoke run (fold 0, 10 épocas): a saída de segmentação chega a 0,69 de accuracy por frame (a classe maioritária dá
  0,47) e reproduz o ciclo; o WA do fold 0 é 0,806. É só um fold, não é uma comparação com a CNN.
- A seguir: CV completa e ablação λ = 0 vs λ > 0, para saber se a saída auxiliar ajuda mesmo.

### Para fechar (qualquer um)

Nas próximas semanas: Unknown (limiares, sinal de qualidade), afinação da CNN, CV da RNN, ensemble dos folds,
avaliação única no teste e script para os dados ocultos. O plano está em `docs/PLANO.md`.

## Perguntas prováveis

P: Porque não dividiram só por `Patient ID`?

R: Porque 70 crianças têm dois IDs (uma visita em cada campanha). Dividindo por ID, o mesmo coração podia ficar no
treino e no teste. Ligámos os IDs com um union-find (`src/data.py`).

P: Usaram o conjunto de teste para alguma coisa?

R: Não. Não está carregado em nenhuma secção dos modelos, e o `main` tem um `assert`. Até a exploração do áudio que
justificou o pré-processamento foi feita só no desenvolvimento.

P: Como obtêm uma previsão por paciente?

R: Cada gravação fica com a média das probabilidades das suas janelas. Para o paciente, P(Present) é o máximo sobre as
gravações (basta um local com sopro) e P(Unknown) e P(Absent) são médias; depois renormalizamos. A decisão é
`argmax(w_c * p_c)`, com w = (5, 3, 1).

P: Porque é que essa regra é a certa?

R: O denominador do WA não depende das previsões. Acertar um paciente da classe c vale w_c, logo o ganho esperado de
prever c é w_c * p_c. Com probabilidades calibradas, maximizar isso é o melhor que se pode fazer.

P: Então porque não usar também class weights?

R: Porque os custos ficavam contados duas vezes. Testámos: E1 + Bayes cai para ~0,55, porque quase tudo passa a ser
Present.

P: 0,82 não é bom demais?

R: É um valor de validação: a época de paragem é escolhida no mesmo fold e as curvas são ruidosas, por isso é
otimista. A estimativa honesta vai ser a do ensemble dos folds avaliado uma vez no teste. Não comparamos com o
Challenge, que tinha outro conjunto de teste.

P: Porque CNN 1D e não 2D sobre um espectrograma?

R: As bandas cobrem exatamente a zona onde o espectro da sístole é diferente, e a convolução 1D mantém a resolução
temporal fina do S1/S2. Há precedente (Potes et al. 2016). A representação tempo-frequência ficou para a RNN
(log-mel).

P: Porque GRU e não LSTM? E porque bidirecional?

R: Temos ~2000 gravações de treino por fold e a GRU tem menos parâmetros. É bidirecional porque classificamos
gravações completas (offline): saber o que vem a seguir ajuda a localizar o S1 e o S2.

P: E se os dados ocultos não tiverem .tsv?

R: Nenhum modelo precisa deles na inferência. Na RNN, os .tsv são só alvos da saída auxiliar durante o treino.

P: Como lidam com durações diferentes?

R: Crops aleatórios de 4 s (CNN) ou 5 s (RNN) no treino, e janelas com 50% de sobreposição na inferência, com a média
por gravação. Todas as gravações têm mais de 5 s, por isso nunca é preciso padding.

P: Porque early stopping no WA e não na `val_loss`?

R: A `val_loss` é calculada por crop e com os rótulos das gravações, e o que nos avaliam é o WA por paciente. Fizemos
um callback que calcula o WA no fim de cada época.

P: Qual é a maior fraqueza agora?

R: A classe Unknown: a E2 só reconhece 6 de 54. É a prioridade das próximas semanas.

P: Usaram IA generativa?

R: Responder com verdade e em concreto: sim, como apoio, e cada decisão está justificada com evidência tirada dos
nossos dados (secção 3 do notebook e registo de decisões no PLANO). Cada um de nós sabe explicar o código.

## Perguntas a fazer ao professor

1. Os dados ocultos vão trazer os ficheiros `<paciente>.txt` e/ou os `.tsv`, ou só os `.wav`?
2. Quando são libertados?
3. É aceitável reportar a CV com early stopping no próprio fold, dizendo que é otimista? Ou preferem que reportemos
   só o teste?

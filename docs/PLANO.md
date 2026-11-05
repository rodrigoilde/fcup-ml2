# Plano do projeto: deteção de sopros cardíacos (CirCor)

Documento interno do grupo, atualizado a 3 nov 2026, antes do checkpoint de 5 nov.
Entrega: 30 nov 2026, 23:59. Apresentações: 3 e 10 dez.

O entregável é o `main.ipynb` (secções 1 a 8). Quando abaixo falamos em "secção X", é a secção desse notebook.

## 1. Divisão de trabalho

Quase todo o código está no `main.ipynb`. Fora dele só está `src/training.py` (a maquinaria de treino do Keras).

| Membro | Responsável por | Secções do main |
|---|---|---|
| A | Regras do problema, métrica e regras de decisão, dados, divisão, exploração do áudio, consequências práticas | 1, 2, 3 (exploração), 6, 7 |
| B | CNN 1D (pré-processamento por bandas, arquitetura, treino) e `src/training.py` | 3 (entrada da CNN), 4, 5, 6 |
| C | BiGRU multitarefa (log-mel, estados por frame, saída auxiliar) | 3 (entrada da RNN), 4, 5 |

Cada um tem de conseguir explicar o projeto todo, porque o enunciado penaliza quem não souber. Antes de cada
reunião, cada um lê as secções dos outros.

## 2. Estado por etapa (secções 3.1 a 3.4 do enunciado)

### Comum

| Tarefa | Estado |
|---|---|
| Visão geral dos dados (classes, campanhas, IDs ligados) | feito (secção 2) |
| Divisão: teste fechado de 20% + CV de 5 folds, agrupada por criança e estratificada por sopro x campanha | feito (secção 2, `splits/splits.csv`) |
| Métrica WA + matriz de confusão (linhas = previsto) + verificações (asserts) | feito (secção 1) |
| Exploração do áudio só no conjunto de desenvolvimento | feito (secções 2 e 3) |
| Agregação das gravações por paciente (média, máximo para Present) e regras de decisão (argmax, Bayes) | feito (secção 1) |
| Consequências práticas (o que os números querem dizer num rastreio) | feito (secção 7) |
| Melhorar a deteção de Unknown (limiares nos OOF, sinal de qualidade) | por fazer, é a prioridade |
| Ensemble dos modelos dos 5 folds | por fazer |
| Avaliação no teste (no fim, uma única vez por modelo) | por fazer |
| Script de previsão para os dados ocultos (`patient_id,murmur`) | por fazer |
| Slides (10 min + apêndice) | por fazer |
| Autoavaliação | por fazer |

### CNN 1D (membro B)

| Etapa | Estado |
|---|---|
| 3.1 Pré-processamento: 2 kHz, 5 bandas (25-800 Hz), normalização robusta, janelas de 4 s | feito |
| 3.2 Arquitetura: 5 blocos Conv1D-BN-ReLU-MaxPool, receptive field de ~0,92 s, GAP, dropout 0,5 (247k parâmetros) | feito |
| 3.3 Treino: Adam 1e-3, batch 32, crops aleatórios + ganho + polaridade, early stopping no WA por paciente | feito (versão base) |
| 3.4 CV de 5 folds, duas configurações (E1 vs E2) | feito |
| Afinação a partir da E2: janela (3/4/6 s), GAP vs GAP+GMP, bandas, com e sem corte em 20 | por fazer |

### BiGRU multitarefa (membro C)

| Etapa | Estado |
|---|---|
| 3.1 Log-mel (32 bandas, 25-800 Hz, 50/20 ms) + estados por frame a partir dos .tsv | feito |
| 3.2 Duas camadas BiGRU(64) + saída de segmentação + saída do sopro com attention pooling (117k parâmetros) | feito |
| 3.3 Treino multitarefa (λ = 0,5): smoke run no fold 0, 10 épocas | em curso |
| 3.4 CV completa de 5 folds, com mais épocas | por fazer |
| Ablação principal: λ = 0 (sem saída auxiliar) vs λ > 0 | por fazer |

## 3. Registo de decisões

Para cada decisão pusemos a evidência que a justifica (com a secção do `main.ipynb`) e a alternativa "por defeito"
que não escolhemos. Esta tabela vai servir de base à parte da implementação nos slides.

| # | Decisão | Evidência | Alternativa que não escolhemos |
|---|---|---|---|
| 1 | Agrupar os splits por criança (juntando `Patient ID` e `Additional ID`) | 70 crianças aparecem com dois IDs, uma visita em cada campanha (secção 2) | Dividir só por `Patient ID`, o que punha a mesma criança no treino e no teste |
| 2 | Estratificar por sopro x campanha | CC2014 tem 16% Present e CC2015 tem 21% (secção 2) | Estratificar só pelo sopro |
| 3 | Teste de 20% fechado + CV de 5 folds no resto | Só há 68 Unknown no total, por isso um único split de validação seria muito ruidoso | Um único train/val/test |
| 4 | Explorar o áudio só no desenvolvimento | O enunciado proíbe usar o teste em decisões de pré-processamento | Explorar os 942 pacientes (a primeira versão fazia isto; corrigimos a 3 nov) |
| 5 | Rótulos por gravação conforme o local | Em 54 de 143 Present (dev) o sopro não se ouve em todos os locais (secção 2) | Copiar o rótulo do paciente para todas as gravações |
| 6 | 2 kHz + banco de 5 filtros entre 25 e 800 Hz | Na sístole, as Present têm 2,5 a 5 vezes mais potência entre 25 e 800 Hz e praticamente a mesma acima de 800 Hz; na gravação inteira quase não há diferença (secção 3) | Banda completa a 4 kHz |
| 7 | CNN 1D sobre as bandas (e não CNN 2D sobre mel) | Bandas de Potes et al. (vencedores da PhysioNet 2016) e a análise do espectro; mantém a resolução temporal fina do S1/S2 | CNN 2D sobre mel-espectrograma (a escolha mais comum) |
| 8 | Normalização robusta (mediana/MAD) + corte em 20 | Picos de fricção; o corte apanha 0,72% das amostras (secção 3). Ainda não testámos se ajuda | z-score simples |
| 9 | Janelas de 4 s, receptive field de ~0,92 s | Ciclo cardíaco: mediana 0,58 s, p95 0,81 s; todas as gravações têm mais de 5 s (secções 2 e 3) | Gravação inteira com padding |
| 10 | Saída auxiliar de segmentação na RNN (os .tsv só servem de alvo no treino) | Os sopros são quase sempre sistólicos; as Unknown têm 69% do tempo sem anotação, contra 41-45% (secção 3) | LSTM sobre MFCC; ou usar os .tsv como entrada (podem não vir nos dados ocultos) |
| 11 | GRU em vez de LSTM | ~2000 gravações de treino por fold, por isso menos parâmetros | LSTM |
| 12 | Early stopping no WA por paciente (callback nosso) | É a métrica com que somos avaliados; a `val_loss` é calculada por crop | `val_loss` |
| 13 | CE sem pesos + P(Present) = máximo sobre os locais + regra de Bayes (`argmax w_c p_c`) | O denominador do WA é constante, logo a decisão ótima é `argmax w_c p_c`. Confirmado: a E2 ganha à E1 nos 5 folds (secção 6) | CE com class weights + argmax |
| 14 | Modelo final = ensemble dos 5 modelos dos folds | Cada um tem um early stopping honesto, e assim não é preciso treinar outra vez sem validação | Treinar de novo em todo o dev com um número fixo de épocas |

## 4. Resultados intermédios (CV, conjunto de desenvolvimento, por paciente)

Fonte: `results/cnn_cv_summary.csv`, `results/cnn_posthoc_rules.csv` e a secção 6 do `main.ipynb`.

| Modelo | WA por fold | WA média ± dp | Recall P / U / A (OOF, folds juntos) |
|---|---|---|---|
| Sempre Present (referência) | - | 0,499 | 1 / 0 / 0 |
| CNN E1 (CE com pesos, média, argmax) | 0,768; 0,768; 0,751; 0,817; 0,790 | 0,779 ± 0,025 | 0,88 / 0,30 / 0,79 |
| CNN E2 (CE sem pesos, máximo para Present, Bayes) | 0,796; 0,825; 0,799; 0,827; 0,857 | 0,821 ± 0,025 | 0,91 / 0,11 / 0,91 |
| BiGRU, smoke run (fold 0, 10 épocas) | 0,806 (CNN E2 no mesmo fold: 0,796) | - | Unknown: 0/11 |

O que aprendemos até agora:

1. A regra de decisão é o que mais conta. O mesmo modelo E2 com argmax fica em 0,693 e com a regra de Bayes sobe
   para 0,821. Usar o máximo sobre os locais dá mais cerca de 0,045 em relação à média.
2. Pôr os custos na loss e na decisão ao mesmo tempo estraga tudo. E1 (loss com pesos) + regra de Bayes cai para
   0,54-0,57, porque quase toda a gente passa a ser Present.
3. Os class weights deixam o treino instável. Uma gravação Unknown pesa ~16 vezes mais do que uma Absent e as curvas
   de validação da E1 oscilam muito.
4. O ponto fraco é o Unknown. A E2 só reconhece 6 de 54 (37 vão para Absent). É a prioridade das próximas semanas.
5. Os números são otimistas, porque a época é escolhida no mesmo fold que reportamos. A estimativa honesta vai ser a
   do ensemble no teste. Nos slides não podemos dizer que "igualamos o vencedor do Challenge".

## 5. Calendário

| Período | Tarefas | Quem |
|---|---|---|
| até 5 nov | Tudo o que está marcado como feito; preparar o checkpoint (`docs/checkpoint.md`) | todos |
| 6 a 12 nov | CNN: afinação a partir da E2. RNN: CV completa + ablação de λ | B, C |
| 6 a 12 nov | Unknown: limiares nos OOF, sinal de qualidade; análise de erros por campanha, idade e grau do sopro | A |
| 13 a 19 nov | Escolher a configuração final de cada modelo só com a CV; ensemble dos folds; avaliação única no teste | todos |
| 20 a 26 nov | Previsões nos dados ocultos (quando forem libertados); slides | A (script), todos (slides) |
| 27 a 30 nov | Correr o `main.ipynb` de raiz com `RETRAIN_*=True` (reprodutibilidade), autoavaliação, submissão | todos |
| 3 e 10 dez | Apresentação (todos têm de estar presentes) | todos |

## 6. Riscos e dúvidas

- Formato dos dados ocultos. Os .tsv podem não vir, mas nenhum dos modelos precisa deles na inferência. Se não vierem
  os `<paciente>.txt`, temos de agrupar os .wav pelo prefixo do nome do ficheiro. Perguntar ao professor no
  checkpoint.
- A classe Unknown é pequena (54 pacientes no dev, ~11 por fold), por isso o recall de Unknown por fold é muito
  ruidoso. Convém olhar para a matriz dos OOF com os folds todos juntos e não só para a média.
- Ruído nos rótulos: em 15 dos 70 pares ligados as duas visitas têm rótulos diferentes.
- Otimismo da CV (ver o ponto 5 da parte 4 deste documento). Uma alternativa é reportar também a CV com um número
  fixo de épocas.
- Disciplina com o teste: ninguém usa pacientes com `split == "test"` antes da semana de 13 a 19 nov. O `main.ipynb`
  tem um `assert` para isso.
- Tempo de CPU: a CV da CNN demora ~35 min. Correr as afinações à noite ou dividir os folds entre nós.

## 7. Regras do grupo

1. Nenhum número entra nos slides ou neste documento sem vir de um output executado (`main.ipynb` ou CSV em
   `results/`).
2. Cada experiência nova tem um nome (`cnn_E3`, ...), o OOF guardado em `results/` e uma linha no registo de decisões.
3. Quem mexer no código corre o `main.ipynb` de raiz antes de partilhar: os asserts confirmam que a métrica, a
   divisão e as regras de decisão continuam certas.

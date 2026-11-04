# Plano do projeto — deteção de sopros cardíacos (CirCor)

Documento interno do grupo. Atualizado a **3 nov 2026**, antes do checkpoint de **5 nov**.
Entrega: **30 nov 2026, 23:59**. Apresentações: **3 e 10 dez**.

Legenda: ✅ feito · 🟡 em curso · ⬜ por fazer

O entregável é o **`main.ipynb`** (secções 0–7). As referências "§" abaixo são secções desse notebook.

---

## 1. Divisão de trabalho

| Membro | Responsável por | Ficheiros principais | Secções do main |
|---|---|---|---|
| **A** | Dados, exploração, protocolo de avaliação, métrica, agregação por paciente | `src/data.py`, `src/splits.py`, `src/metrics.py`, `src/aggregate.py` | §1, §2, §3, §5.6 |
| **B** | CNN 1D (pré-processamento por bandas, arquitetura, treino) | `src/preprocessing.py`, `src/models.py:build_cnn1d`, `src/training.py` | §4.1, §5 |
| **C** | BiGRU multitarefa (log-mel, estados por frame, cabeça auxiliar) | `src/features.py`, `src/models.py:build_bigru_mt` | §4.2, §6 |

Cada membro tem de conseguir explicar **todo** o projeto, porque o enunciado penaliza quem não souber. Antes de cada
reunião, cada um lê as secções dos outros.

---

## 2. Estado por etapa (secções 3.1–3.4 do enunciado)

### Comum

| Tarefa | Estado |
|---|---|
| Visão geral dos dados (classes, campanhas, IDs ligados) | ✅ §1 |
| Divisão: teste fechado de 20% + CV de 5 folds, agrupada por sujeito e estratificada por sopro × campanha | ✅ §2, `splits/splits.csv` |
| Métrica WA + matriz de confusão (linhas = previsto) + testes unitários | ✅ §2.2, `tests/` (8 testes) |
| Exploração do áudio **só no conjunto de desenvolvimento** | ✅ §3 |
| Agregação gravação → paciente (média, máx.-Present) e regras de decisão (argmax, Bayes) | ✅ |
| Melhorar a deteção de *Unknown* (limiares nos OOF, sinal de qualidade) | ⬜ **prioridade** |
| Ensemble dos modelos dos 5 folds | ⬜ |
| Avaliação **única** no teste (no fim, uma vez por modelo) | ⬜ |
| Script de previsão para os dados ocultos (`patient_id,murmur`) | ⬜ |
| Slides (10 min + apêndice) | ⬜ |
| Autoavaliação | ⬜ |

### CNN 1D (membro B)

| Etapa | Estado |
|---|---|
| 3.1 Pré-processamento: 2 kHz, 5 bandas (25–800 Hz), normalização robusta, janelas de 4 s | ✅ |
| 3.2 Arquitetura: 5 blocos Conv1D-BN-ReLU-MaxPool, campo recetivo ≈ 0,92 s, GAP, dropout 0,5 (247k parâmetros) | ✅ |
| 3.3 Treino: Adam 1e-3, batch 32, cortes aleatórios + ganho + polaridade, early stopping no WA por paciente | ✅ base |
| 3.4 CV de 5 folds, duas configurações (E1 vs E2) | ✅ |
| Afinação com E2 como base: janela (3/4/6 s), GAP vs GAP+GMP, bandas, com/sem recorte em ±20 | ⬜ |

### BiGRU multitarefa (membro C)

| Etapa | Estado |
|---|---|
| 3.1 Log-mel (32 bandas, 25–800 Hz, 50/20 ms) + estados por frame a partir dos .tsv | ✅ |
| 3.2 2×BiGRU(64) + cabeça de segmentação + cabeça de sopro com *attention pooling* (117k parâmetros) | ✅ |
| 3.3 Treino multitarefa (λ = 0,5): *smoke run* no fold 0, 10 épocas | 🟡 |
| 3.4 CV completa de 5 folds, com mais épocas | ⬜ |
| **Ablação-chave:** λ = 0 (sem cabeça auxiliar) vs λ > 0 | ⬜ |

---

## 3. Registo de decisões

Cada decisão tem a evidência que a justifica (com a secção do `main.ipynb`) e a alternativa "por defeito" que
rejeitámos. Esta tabela é a base da parte de implementação dos slides.

| # | Decisão | Evidência | Alternativa rejeitada |
|---|---|---|---|
| 1 | Agrupar os splits por **sujeito** (`Patient ID` ∪ `Additional ID`) | 70 crianças aparecem com dois IDs, uma visita em cada campanha (§1.3) | Dividir só por `Patient ID`, o que poria a mesma criança no treino e no teste |
| 2 | Estratificar por **sopro × campanha** | CC2014 tem 16% Present e CC2015 tem 21% (§1.2) | Estratificar só pelo sopro |
| 3 | **Teste de 20% fechado** + CV de 5 folds no resto | Só 68 Unknown no total: um único split de validação seria muito ruidoso | Um único train/val/test |
| 4 | **Exploração do áudio só no desenvolvimento** | O enunciado proíbe usar o teste em decisões de pré-processamento | Explorar os 942 pacientes (a 1.ª versão fazia isto; corrigido a 3 nov) |
| 5 | **Rótulos por gravação conforme o local** | 54/143 Present (dev) não têm sopro audível em todos os locais (§3.2) | Copiar o rótulo do paciente para todas as gravações |
| 6 | **2 kHz + banco de 5 filtros 25–800 Hz** | Na sístole, Present tem 2,5–5× mais potência em 25–800 Hz e ≈ igual acima de 800 Hz; na gravação inteira quase não há diferença (§3.5) | Banda completa a 4 kHz |
| 7 | **CNN 1D sobre as bandas** (não CNN 2D sobre mel) | Bandas de Potes et al. (vencedor da PhysioNet 2016) + evidência espectral; mantém a resolução temporal fina de S1/S2 | CNN 2D sobre mel-espectrograma (a escolha mais comum) |
| 8 | Normalização robusta (mediana/MAD) + recorte em ±20 | Picos de fricção; o recorte afeta ≤1,2% das amostras por banda (§4.1). **Não testado ainda** | z-score simples |
| 9 | Janelas de 4 s, campo recetivo ≈ 0,92 s | Ciclo cardíaco: mediana 0,58 s, p95 0,81 s; todas as gravações têm >5 s (§3.1, §3.3) | Gravação inteira com *padding* |
| 10 | **Cabeça auxiliar de segmentação** na RNN (os .tsv só servem de alvo no treino) | Sopros quase sempre sistólicos; Unknown tem 69% do tempo não anotado vs 41–45% (§3.3) | LSTM sobre MFCC; ou usar os .tsv como entrada (podem não existir nos dados ocultos) |
| 11 | GRU em vez de LSTM | ~2000 gravações de treino por fold: menos parâmetros | LSTM |
| 12 | **Early stopping no WA por paciente** (callback próprio) | É a métrica avaliada; a `val_loss` é calculada por corte | `val_loss` |
| 13 | **CE sem pesos + P(Present) = máximo sobre os locais + regra de Bayes** `argmax w·p` | WA = Σ w·acertos / constante → a decisão ótima é `argmax w_c p_c`. Confirmado: E2 > E1 nos 5 folds (§5.5–5.6) | CE com pesos de classe + argmax |
| 14 | Modelo final = ensemble dos 5 modelos dos folds | Cada um tem *early stopping* honesto; não é preciso treinar de novo sem validação | Treinar de novo em todo o dev com um número fixo de épocas |

---

## 4. Resultados intermédios (CV, conjunto de desenvolvimento, por paciente)

Fonte: `results/cnn_cv_summary.csv`, `results/cnn_posthoc_rules.csv`, `main.ipynb` §5.5–5.8 e §6.

| Modelo | WA por fold | WA média ± dp | Recall P / U / A (OOF agregado) |
|---|---|---|---|
| Sempre *Present* (referência) | – | 0,499 | 1 / 0 / 0 |
| CNN E1 (CE com pesos, média, argmax) | 0,768 · 0,768 · 0,751 · 0,817 · 0,790 | 0,779 ± 0,025 | 0,88 / 0,30 / 0,79 |
| **CNN E2** (CE sem pesos, máx.-Present, Bayes) | 0,796 · 0,825 · 0,799 · 0,827 · 0,857 | **0,821 ± 0,025** | 0,91 / 0,11 / 0,91 |
| BiGRU, *smoke run* (fold 0, 10 épocas) | 0,806 (CNN E2 no mesmo fold: 0,796) | – | Unknown: 0/11 |

O que aprendemos:

1. **A regra de decisão é o ingrediente principal.** O mesmo modelo E2 com argmax fica em 0,693; com a regra de Bayes
   sobe para 0,821. O máximo sobre os locais acrescenta ≈ +0,045 em relação à média.
2. **Contar os custos duas vezes estraga tudo.** E1 (perda com pesos) + regra de Bayes cai para 0,54–0,57, porque
   quase toda a gente passa a ser classificada como *Present*.
3. **Os pesos de classe tornam o treino instável.** Uma gravação *Unknown* pesa ~16× uma *Absent*, e as curvas de
   validação de E1 oscilam muito.
4. **Ponto fraco: *Unknown*.** E2 só reconhece 6 de 54 (37 vão para *Absent*). É a prioridade das próximas semanas.
5. **Os números são otimistas.** A época é escolhida no mesmo fold que reportamos. A estimativa honesta será o
   ensemble no teste. Não dizer nos slides que "igualamos o vencedor do Challenge".

---

## 5. Calendário

| Período | Tarefas | Quem |
|---|---|---|
| até 5 nov | Tudo o que está ✅ acima; preparar o checkpoint (`docs/checkpoint.md`) | todos |
| 6–12 nov | CNN: afinação com E2 como base. RNN: CV completa + ablação λ | B, C |
| 6–12 nov | *Unknown*: limiares nos OOF, sinal de qualidade; análise de erros por campanha, idade e grau do sopro | A |
| 13–19 nov | Escolher a configuração final de cada modelo **só com a CV**; ensemble dos folds; avaliação única no teste | todos |
| 20–26 nov | Previsões nos dados ocultos (quando forem libertados); slides | A (script), todos (slides) |
| 27–30 nov | Correr o `main.ipynb` de raiz com `RETRAIN_*=True` (reprodutibilidade), autoavaliação, submissão | todos |
| 3 / 10 dez | Apresentação (presença de todos obrigatória) | todos |

---

## 6. Riscos e questões em aberto

- **Formato dos dados ocultos.** Os .tsv podem não vir (nenhum dos modelos precisa deles na inferência). Se não
  vierem os `<paciente>.txt`, é preciso agrupar os .wav pelo prefixo do nome do ficheiro. **Perguntar ao professor
  no checkpoint.**
- **Classe Unknown pequena** (54 pacientes no dev, ~11 por fold): o recall de Unknown por fold é muito ruidoso;
  olhar para a matriz agregada dos OOF e não só para a média.
- **Ruído nos rótulos**: em 15 dos 70 pares ligados, as duas visitas têm rótulos diferentes.
- **Otimismo da CV** (ver §4, ponto 5). Uma alternativa a considerar: reportar também a CV com um número fixo de
  épocas.
- **Disciplina com o teste**: ninguém usa pacientes com `split == "test"` antes da semana de 13–19 nov. O
  `main.ipynb` tem um `assert` para isso.
- **Tempo de CPU**: a CV da CNN demora ~35 min. Correr as afinações à noite ou dividir os folds entre membros.

## 7. Regras do grupo

1. Nenhum número entra nos slides ou neste documento sem vir de um output executado (`main.ipynb` ou CSV em `results/`).
2. Cada experiência nova tem nome (`cnn_E3`, ...), OOF guardado em `results/` e uma linha no registo de decisões.
3. Mudanças em `src/` correm `pytest tests` antes de serem partilhadas.

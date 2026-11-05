# Deteção de sopros cardíacos em fonocardiogramas (CirCor DigiScope)

Machine Learning II (CC3043), FCUP, 2026/2027. Projeto de grupo.

Para cada paciente queremos prever, a partir das gravações de auscultação, se tem um sopro cardíaco (Present), se não
tem (Absent) ou se não dá para saber (Unknown). Os modelos são avaliados com a weighted accuracy do PhysioNet
Challenge 2022 (pesos 5 / 3 / 1).

Dos três classificadores do enunciado implementámos dois:

- uma CNN 1D sobre o sinal dividido em 5 bandas de frequência, entre 25 e 800 Hz;
- uma GRU bidirecional multitarefa sobre frames log-mel, com uma saída auxiliar que prevê o estado do ciclo cardíaco
  (S1, sístole, S2, diástole ou não anotado) em cada frame.

O entregável é o [`main.ipynb`](main.ipynb), e é lá que está quase todo o código. Segue a ordem em que pensámos o
problema:

1. Regras do problema: o que temos de prever, quanto custa cada erro e que regras saem daí (a métrica e as regras de
   decisão)
2. Dados: limpeza e divisão em teste e desenvolvimento
3. Do áudio às entradas das redes
4. Modelos
5. Treino e validação
6. Avaliação e métricas
7. Consequências práticas
8. Próximos passos

Fora do notebook só está `src/training.py`, com a maquinaria de treino do Keras (o dataset de crops, a inferência por
janelas, o callback de early stopping e o ciclo de CV). A métrica e as regras de decisão são definidas no notebook e
passadas a esse código como argumento.

Estado no checkpoint de 5 de novembro de 2026: já fizemos a exploração dos dados, o protocolo de avaliação, a
métrica, o pré-processamento dos dois modelos e uma primeira CNN avaliada com cross-validation. A RNN já corre do
início ao fim (smoke run num fold), mas ainda não foi avaliada com CV nem afinada. O conjunto de teste ainda não foi
usado.

## Instalação

```bash
conda activate ml2
pip install -r requirements.txt
```

Qualquer ambiente com Python 3.11 serve. O dataset não vem incluído. Por defeito, o código procura os dados públicos
do CirCor (v1.0.3) ao lado deste repositório:

```
../the-circor-digiscope-phonocardiogram-dataset-1.0.3/
    training_data.csv
    training_data/   (<patient>.txt, <patient>_<loc>.wav/.hea/.tsv)
```

Para usar outra pasta, basta definir a variável de ambiente `CIRCOR_DIR`.

## Como correr

```bash
jupyter notebook main.ipynb
```

Depois é só correr as células por ordem. As verificações que antes estavam em testes à parte (a métrica num exemplo
feito à mão, as regras de decisão, os pares de IDs ligados, a divisão e os rótulos por local) são agora asserts no
próprio notebook, junto do código que verificam. O `splits/splits.csv` vem no repositório, e o notebook
volta a calcular a divisão e confirma que dá exatamente o mesmo. Se o ficheiro não existir, é o notebook que o cria.

Na célula de setup do `main.ipynb` há dois interruptores:

| Interruptor | `False` (por defeito) | `True` |
|---|---|---|
| `RETRAIN_CNN` | usa as previsões out-of-fold guardadas em `results/` (e treina se não existirem) | CV de 5 folds das duas configurações, uns 35 min no CPU de um portátil |
| `RETRAIN_RNN` | usa os resultados guardados do smoke run (e treina se não existirem) | smoke run no fold 0, poucos minutos |

Com os resultados guardados, o notebook corre em poucos minutos. Os arrays pré-processados ficam em `cache/` e os
modelos treinados em `results/models/`. As duas pastas são criadas automaticamente e não estão no git. A exceção é o
modelo do smoke run da RNN, que a secção 5 do notebook precisa.

## Protocolo de avaliação

Agrupamos por criança e não só pelo Patient ID. Há 70 crianças que foram gravadas nas duas campanhas e que aparecem
com dois Patient ID, ligados pelo `Additional ID`. Os IDs ligados ficam com o mesmo `group_id` e vão sempre para o
mesmo subconjunto.

O conjunto de teste tem 20% dos pacientes (189) e está estratificado por sopro x campanha. Ainda não foi usado e só
vai ser usado uma vez, no fim. Para todas as decisões sobre os modelos usamos cross-validation de 5 folds nos
restantes 753 pacientes (o conjunto de desenvolvimento). As análises do áudio que levaram às escolhas de
pré-processamento também usam só o conjunto de desenvolvimento.

Fazemos uma previsão por paciente. Na matriz de confusão por paciente, as linhas são a classe prevista e as colunas a
verdadeira, como na Tabela 1 do enunciado.

## Resultados até agora (desenvolvimento, CV de 5 folds, out-of-fold, por paciente)

| Modelo | Loss | Agregação / decisão | WA (média ± desvio padrão nos folds) |
|---|---|---|---|
| prever sempre Present (referência) | - | - | 0.499 |
| CNN 1D, E1 | CE com class weights | média / argmax | 0.779 ± 0.025 |
| CNN 1D, E2 | CE sem pesos | máximo para Present / regra de Bayes para o WA | 0.821 ± 0.025 |
| BiGRU multitarefa (smoke run) | CE + 0.5 x CE da segmentação | máximo para Present / Bayes | 0.806 (só no fold 0, não dá para comparar) |

São valores de validação: a época do early stopping é escolhida no mesmo fold, por isso são otimistas. As matrizes de
confusão, as métricas adicionais e a interpretação estão na secção 6 do `main.ipynb`, e o que isto quer dizer na
prática está na secção 7.

## Estrutura do repositório

```
main.ipynb           o notebook entregável (executado, com outputs), com quase todo o código
src/training.py      maquinaria de treino: dataset de crops, inferência por janelas, early stopping, ciclo de CV
splits/splits.csv    a divisão fixa (o notebook confirma que é reproduzível)
results/             previsões out-of-fold, resumos da CV, figuras
docs/                notas de planeamento e do checkpoint (internas)
```

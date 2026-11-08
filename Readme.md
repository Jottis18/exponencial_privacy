# Trabalho 4: Privacidade Diferencial – Mecanismo Exponencial no r-N

- Disciplina: Privacidade de Dados
- Professor: Javam Machado
- Linguagem: Python

---

## Descrição do Projeto

Este projeto implementa do zero (sem bibliotecas de machine learning como o *scikit-learn*) o classificador **r-N (radius neighbors)** e a sua versão **diferencialmente privada** com o **Mecanismo Exponencial**, seguindo o Algoritmo 2 de Gursoy et al. (2017).

O objetivo é medir o quanto a privacidade custa em acurácia na base *Adult Income Dataset*. Para isso, comparamos o r-N tradicional com o r-N privado para os raios r ∈ {0,5; 1; 1,5; 2; 6} e os orçamentos ε ∈ {0,1; 0,5; 1; 5; 10; 100}.

---

## Como Executar

Requisitos: Python 3 e as bibliotecas listadas em `requirements.txt` (numpy, pandas, matplotlib e tqdm).

1. Deixe o arquivo `adult.csv` na pasta do projeto. Se ele não estiver lá, baixe-o em
   https://www.kaggle.com/datasets/wenruliu/adult-income-dataset/

2. Instale as dependências:

   ```
   pip install -r requirements.txt
   ```

3. Execute o programa (no Windows, use `python` no lugar de `python3`):

   ```
   python3 trabalho_exponencial.py
   ```

A execução leva cerca de 10 segundos e salva todos os resultados na pasta `resultados/`. No final, duas janelas mostram os gráficos, e o programa termina quando elas são fechadas. Como a semente é fixa, os resultados são os mesmos a cada execução.

Opções:

- `--detalhar ID`: mostra o passo a passo da tupla de teste de número ID. Primeiro aparecem os atributos originais da tupla. Depois, para cada raio, aparecem as contagens de cada classe no raio e a predição do r-N e, para cada ε, as probabilidades do mecanismo Exponencial e a classe sorteada. Usada na demo, por exemplo: `python3 trabalho_exponencial.py --detalhar 0`
- `--dados CAMINHO`: lê o dataset de outro caminho.
- `--saida PASTA`: salva os resultados em outra pasta (padrão: `resultados/`).
- `--semente N`: altera a semente da divisão treino/teste, dos desempates e dos sorteios (padrão: 42).

---

## Pré-processamento dos Dados

A base original tem 48.842 tuplas e 15 atributos. Ela passou pelas seguintes etapas:

1. **Remoção de atributos:** `fnlwgt`, `education`, `capital-gain`, `capital-loss` e `hours-per-week`.
2. **Valores faltantes:** foram removidas as 3.620 tuplas que contêm `"?"`, restando 45.222 tuplas.
3. **One-Hot Encoding nos 7 atributos categóricos** (`workclass`, `marital-status`, `occupation`, `relationship`, `race`, `gender` e `native-country`). Cada valor distinto vira uma coluna 0/1, e a tupla tem 1 apenas na coluna do seu valor.
4. **MinMax nos 2 atributos numéricos** (`age` e `educational-num`): v' = (v − mín) / (máx − mín), o que leva cada um para [0, 1]. Os limites usados são os da base: 17 a 90 anos e 1 a 16.
5. **Classe** (`income`): `<=50K` = 0 e `>50K` = 1.

No total, os 9 atributos viram 84 colunas, todas em [0, 1]. A tabela completa está em `resultados/codificacao.csv`.

**Consequência para a distância.** Dois valores diferentes de um atributo categórico somam 1² + 1² = 2 ao quadrado da distância euclidiana. Um atributo MinMax soma no máximo 1. Portanto:

- a maior distância possível entre duas tuplas é √(2·7 + 2) = **4**;
- com r = 0,5 ou r = 1, só entram no raio tuplas com **todos** os atributos categóricos iguais (uma diferença já custaria √2 ≈ 1,41);
- com r = 1,5, entram tuplas com até **1** atributo categórico diferente;
- com r = 2, entram tuplas com até **2** atributos categóricos diferentes;
- com r = 6, entra **todo** o conjunto de treino, para qualquer tupla de teste.

---

## Divisão Treino/Teste

Com semente 42, foram sorteadas, sem reposição, **500 tuplas de cada classe** para o teste (X_test, exatamente 1.000 tuplas, balanceado). As **44.222** tuplas restantes formam o treino (X_train).

---

## r-N Tradicional (sem privacidade)

Segue a Definição 9 do artigo:

1. Para cada tupla de teste x, calcula-se a distância euclidiana até todas as tuplas de treino.
2. Para cada classe I, faz-se a **consulta de contagem** c_{x,r}(I): o número de tuplas de treino da classe I a distância ≤ r de x (Definição 8 do artigo).
3. A classe prevista é a de maior contagem: argmax_I c_{x,r}(I).
4. Empates, inclusive o raio vazio (0 × 0), são decididos por sorteio entre as classes empatadas, como sugere o artigo.

As distâncias de cada tupla de teste são calculadas uma vez só e reaproveitadas para os 5 raios. O NumPy é usado apenas para operações sobre vetores.

---

## r-N Diferencialmente Privado (Mecanismo Exponencial)

### 1. Função de utilidade

A utilidade de responder a classe I para a tupla de teste x é a própria contagem no raio:

    u(D, I) = c_{x,r}(I) = número de tuplas de treino da classe I a distância ≤ r de x

Quanto mais vizinhos de uma classe dentro do raio, maior a chance de ela ser escolhida.

### 2. Sensibilidade global

**Δu = 1.** Incluir ou remover uma tupla do treino altera a contagem de **uma única** classe em no máximo 1 unidade. Isso porque cada tupla tem uma só classe e só conta se estiver dentro do raio. A utilidade das outras classes não muda.

### 3. Como o mecanismo foi aplicado (Algoritmo 2)

Para cada tupla de teste, o mecanismo devolve a classe I com probabilidade

    P(I) = exp(ε' · c_{x,r}(I) / (2Δu)) / Σ_J exp(ε' · c_{x,r}(J) / (2Δu))

Esse é o Mecanismo Exponencial (Definição 5 do artigo), que satisfaz ε'-DP. Na implementação:

- antes da exponencial, subtrai-se de todos os expoentes o maior deles. As probabilidades não mudam (o fator cancela na divisão) e isso evita overflow: com r = 6 e ε = 100, o expoente passaria de 1.600;
- a classe é sorteada pela CDF: com U ~ Uniforme[0, 1), escolhe-se a primeira classe I tal que P(0) + … + P(I) > U.

A saída privada é **somente a classe sorteada**. Por isso, os arquivos de predição do r-N privado não trazem as contagens nem as probabilidades: elas dependem das contagens exatas e quebrariam a privacidade se fossem publicadas.

Com duas classes, a probabilidade de escolher a classe majoritária no raio depende apenas da margem Δ entre as duas contagens:

    P(majoritária) = 1 / (1 + exp(−ε' · Δ / 2))

Exemplo da tupla de teste 0 (`--detalhar 0`): com r = 0,5, há 1.420 vizinhos `<=50K` e 696 `>50K` (Δ = 724). Com ε = 1 (ε' = 0,001), P(`<=50K`) = 1 / (1 + e^(−0,362)) = 58,95%.

### 4. Orçamento usado em cada chamada

O mecanismo é chamado uma vez para cada uma das |X_test| = 1.000 tuplas de teste, e todas as chamadas consultam o mesmo conjunto de treino. Pela **composição sequencial**, os orçamentos se somam. Para o total ser ε, cada chamada recebe:

    ε' = ε / |X_test| = ε / 1.000

| ε total | 0,1    | 0,5    | 1     | 5     | 10   | 100 |
|---------|--------|--------|-------|-------|------|-----|
| ε'      | 0,0001 | 0,0005 | 0,001 | 0,005 | 0,01 | 0,1 |

Como pede o enunciado, o número de classes L não entra nessa divisão. Cada combinação (r, ε) é um experimento independente que gasta ε no total.

### 5. Diferença no uso do orçamento: Trabalho 3 vs. Trabalho 4

| | Trabalho 3 (Laplace, Algoritmo 1) | Trabalho 4 (Exponencial, Algoritmo 2) |
|---|---|---|
| Chamadas por tupla de teste | L = 2 (uma contagem ruidosa por classe) | 1 (o mecanismo já devolve a classe) |
| Orçamento por chamada | ε / (\|X_test\| · L) = ε / (13.566 · 2) = ε / 27.132 | ε / \|X_test\| = ε / 1.000 |
| Saída publicada | as L contagens com ruído | só a classe escolhida |

- No trabalho 3, o enunciado pediu que o Algoritmo 1 devolvesse as L contagens ruidosas. Cada contagem publicada é uma consulta e consome orçamento, então ε foi dividido também por L.
- No trabalho 4, o Mecanismo Exponencial olha as L contagens juntas e publica uma única resposta categórica (a classe). É uma só chamada ε'-DP por tupla de teste, então não há divisão por L.
- Além disso, o teste caiu de 13.566 para 1.000 tuplas. No papel, cada decisão recebe agora 27 vezes mais orçamento (ε/1.000 em vez de ε/27.132). Na prática, o ganho de não dividir por L é anulado pelo fator 1/2 no expoente do Exponencial (ε'/2). Num teste feito no mesmo ambiente do trabalho 4, o Exponencial com ε/1.000 deu praticamente a mesma acurácia que o Laplace com ε/(1.000 · 2), a regra do trabalho 3. O ganho real vem do teste menor (13,6 vezes menos tuplas dividindo ε).
- Observação: o próprio artigo (Teorema 1) mostra que o Algoritmo 1 já seria ε'-DP sem dividir por L, por composição paralela, pois cada tupla de treino entra na contagem de uma só classe. No trabalho 3 seguimos a divisão por L pedida no enunciado, que é a escolha conservadora.

---

## Resultados

### r-N sem privacidade

| r   | Vizinhos no raio (mediana) | Raio vazio | Empates | Margem Δ (mediana) | Acurácia |
|-----|----------------------------|------------|---------|--------------------|----------|
| 0,5 | 199,5                      | 5,6%       | 7,2%    | 71                 | 73,2%    |
| 1   | 235                        | 5,2%       | 6,5%    | 66                 | 72,2%    |
| 1,5 | 2.520                      | 0,2%       | 0,5%    | 566,5              | 64,0%    |
| 2   | 2.642                      | 0,2%       | 0,3%    | 700                | 60,2%    |
| 6   | 44.222                     | 0%         | 0%      | 22.806             | 50,0%    |

(Fonte: `resultados/estatisticas_raio.csv`. "Empates" inclui o raio vazio.)

### r-N privado: acurácia obtida no conjunto de teste (%)

| r \ ε | 0,1  | 0,5  | 1    | 5    | 10   | 100  | Sem privacidade |
|-------|------|------|------|------|------|------|-----------------|
| 0,5   | 50,3 | 51,1 | 51,2 | 58,0 | 57,7 | 67,1 | 73,2            |
| 1     | 49,5 | 49,0 | 51,4 | 58,4 | 61,3 | 66,9 | 72,2            |
| 1,5   | 53,1 | 53,4 | 54,7 | 59,5 | 61,8 | 62,4 | 64,0            |
| 2     | 51,4 | 52,3 | 56,7 | 56,0 | 57,0 | 59,0 | 60,2            |
| 6     | 50,5 | 50,4 | 50,0 | 50,0 | 50,0 | 50,0 | 50,0            |

### r-N privado: acurácia esperada (%)

Como as probabilidades do mecanismo são conhecidas, a acurácia esperada pode ser calculada exatamente: é a média, sobre as tuplas de teste, da probabilidade de sortear a classe correta. A acurácia obtida num único sorteio oscila em torno dela, com desvio padrão de no máximo 1,6 ponto percentual para 1.000 tuplas. Isso explica, por exemplo, por que r = 0,5 com ε = 10 (57,7%) ficou um pouco abaixo de ε = 5 (58,0%).

| r \ ε | 0,1  | 0,5  | 1    | 5    | 10   | 100  |
|-------|------|------|------|------|------|------|
| 0,5   | 50,1 | 50,6 | 51,2 | 55,2 | 58,3 | 67,7 |
| 1     | 50,1 | 50,6 | 51,2 | 55,1 | 58,2 | 67,5 |
| 1,5   | 50,5 | 52,3 | 54,4 | 59,8 | 61,0 | 63,0 |
| 2     | 50,4 | 52,0 | 53,6 | 56,6 | 56,9 | 59,6 |
| 6     | 50,0 | 50,0 | 50,0 | 50,0 | 50,0 | 50,0 |

A tabela completa, com ε' e a concordância entre o r-N privado e o r-N tradicional, está em `resultados/acuracias.csv`. Os gráficos estão em `resultados/grafico_acuracia_exponencial.png` (o gráfico pedido: uma linha por raio, com a referência sem privacidade tracejada na mesma cor) e `resultados/grafico_acuracia_por_raio.png` (um painel por raio, com a acurácia obtida e a esperada).

---

## Análise

1. **O raio define o teto.** Sem privacidade, os raios pequenos são os melhores (73,2% com r = 0,5). Eles só aceitam vizinhos com todos os atributos categóricos iguais. Com r = 1,5 e r = 2, entram tuplas com 1 ou 2 atributos categóricos diferentes, e a vizinhança fica mais de 10 vezes maior (mediana de 199,5 para 2.520 vizinhos) e menos parecida com x: a acurácia cai para 64,0% e 60,2%.

2. **r = 6 não classifica.** Como a maior distância possível é 4, o raio 6 contém o treino inteiro para toda tupla de teste. As contagens são as mesmas para todo mundo (33.514 `<=50K` contra 10.708 `>50K`), e o r-N prevê sempre `<=50K`. Como o teste é balanceado, a acurácia é 50%, com ou sem privacidade. Com ε = 0,1, o mecanismo ainda sorteia `>50K` em cerca de 24% das vezes, mas isso não muda a acurácia, porque o sorteio não depende de x.

3. **A privacidade custa mais quando as contagens são pequenas.** A chance de o mecanismo acertar a classe majoritária é 1 / (1 + e^(−ε'Δ/2)). Com r = 0,5, a margem mediana é Δ = 71:
   - ε = 1 (ε' = 0,001): ε'Δ/2 ≈ 0,036, e a escolha é praticamente uma moeda (50,9%);
   - ε = 100 (ε' = 0,1): ε'Δ/2 ≈ 3,6, e a classe majoritária sai em cerca de 97% dos casos.

   Por isso, para ε ≤ 1 todas as curvas ficam perto de 50%, e só com ε = 100 os raios pequenos chegam a 67%, ainda abaixo do teto de 73%.

4. **Compromisso entre raio e orçamento.** Raios maiores têm margens maiores (mediana de 566 a 700 para r = 1,5 e 2). Assim, o mecanismo reproduz o r-N tradicional com menos orçamento: com ε = 100, a concordância com o r-N sem privacidade é de 96% para r = 1,5 e 2, contra 85% para r = 0,5. Só que o teto desses raios é menor. Com ε = 5 e 10, r = 1,5 é o melhor raio privado; com ε = 100, os raios pequenos voltam a ganhar.

5. **O gargalo é a divisão do orçamento entre as 1.000 tuplas de teste.** É o problema descrito na Seção 3.1 do artigo, e o motivo de os autores proporem o Algoritmo 3. Os números não devem ser comparados direto com os do trabalho 3. Lá mudavam a codificação, o classificador (kNN), a escala do raio, o tamanho do teste e o balanceamento: o teste do trabalho 3 tinha 75,5% de `<=50K`, e o deste tem 50%. No mesmo ambiente, o Exponencial com ε/1.000 e o Laplace com ε/(1.000 · 2) dão praticamente a mesma acurácia. O que pesa no resultado é quantas tuplas de teste dividem o orçamento.

---

## Estrutura do Projeto

```text
.
├── trabalho_exponencial.py                 # Código-fonte (é também o executável: python3 trabalho_exponencial.py)
├── requirements.txt                        # Dependências
├── adult.csv                               # Dataset Adult Income (Kaggle)
├── Readme.txt                              # Este arquivo
└── resultados/
    ├── predicoes/
    │   ├── predicoes_rn_tradicional_r_<r>.csv      # r-N sem privacidade (5 arquivos, um por raio)
    │   └── predicoes_rn_dp_r_<r>_eps_<ε>.csv       # r-N privado (30 arquivos, um por raio e ε)
    ├── acuracias.csv                       # Acurácia obtida e esperada, ε' e concordância, por raio e ε
    ├── estatisticas_raio.csv               # Vizinhança de cada raio e acurácia do r-N sem privacidade
    ├── codificacao.csv                     # As 84 colunas codificadas (One-Hot e MinMax)
    ├── grafico_acuracia_exponencial.png    # Gráfico de acurácia por ε, uma linha por raio
    └── grafico_acuracia_por_raio.png       # O mesmo resultado, um painel por raio
```

Os arquivos de predição têm uma linha por tupla de teste, com as colunas `id_teste`, `classe_real` e `classe_prevista`. Os do r-N tradicional trazem também as contagens exatas no raio (`vizinhos_<=50K` e `vizinhos_>50K`). Os do r-N privado trazem só a classe sorteada.

---

## Referência

Gursoy, M. E.; Inan, A.; Nergiz, M. E.; Saygin, Y. Differentially private nearest neighbor classification. *Data Mining and Knowledge Discovery*, v. 31, p. 1544–1575, 2017. DOI: 10.1007/s10618-017-0532-z.

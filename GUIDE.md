# ft_linear_regression — Guia Completo

## O que é esse projeto?

Você vai construir, do zero, um algoritmo de **Machine Learning** — sem usar bibliotecas que façam o trabalho pesado por você.

O exemplo concreto: dado a **quilometragem de um carro**, o programa deve **prever o preço** desse carro. Simples assim. Mas para chegar lá, você vai ter que entender como um computador "aprende" com dados.

A grande sacada é que o algoritmo não é programado com regras fixas do tipo "se km > 100.000, preço cai X". Ao invés disso, ele **observa exemplos reais** (um dataset com km e preços de carros) e **ajusta sozinho** uma equação matemática até ela ficar boa o suficiente para fazer previsões.

---

## Objetivo

Implementar **regressão linear simples** treinada com **gradiente descendente** — o algoritmo de aprendizado mais fundamental do Machine Learning.

Ao terminar, você será capaz de usar o mesmo algoritmo com qualquer outro dataset, não apenas carros.

---

## Visão Holística do Projeto

```
┌─────────────────────────────────────────────────────────────┐
│                      ft_linear_regression                    │
└─────────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────┐          ┌──────────────────────┐
│  Programa 1:     │          │  Programa 2:          │
│  PREDIÇÃO        │          │  TREINAMENTO          │
│                  │          │                       │
│  - Lê theta0 e   │          │  - Lê o dataset CSV   │
│    theta1 do     │          │  - Roda gradient      │
│    arquivo       │          │    descent            │
│  - Pede km       │          │  - Atualiza theta0    │
│  - Retorna preço │          │    e theta1           │
│                  │          │  - Salva thetas       │
└──────────────────┘          └──────────────────────┘
         │                              │
         └──────────────┬───────────────┘
                        │
              ┌─────────▼──────────┐
              │  CONCEITOS-CHAVE   │
              │                    │
              │  estimatePrice()   │
              │  θ0 + θ1 * km      │
              │                    │
              │  Gradient Descent  │
              │  (ajuste iterativo │
              │   dos thetas)      │
              │                    │
              │  Normalização      │
              │  (para o algoritmo │
              │   convergir)       │
              └────────────────────┘
                        │
              ┌─────────▼──────────┐
              │  BONUS (opcional)  │
              │                    │
              │  - Gráfico dos     │
              │    dados + linha   │
              │  - Cálculo de      │
              │    precisão (R²)   │
              └────────────────────┘
```

**As 3 peças que você vai construir:**

| Peça | O que faz | Arquivo sugerido |
|------|-----------|-----------------|
| Predição | Recebe km, devolve preço | `predict.py` |
| Treinamento | Aprende com o dataset, salva thetas | `train.py` |
| Comunicação entre os dois | Arquivo com os valores aprendidos | `thetas.csv` ou `model.txt` |

---

## O que você precisa aprender (passo a passo)

### Etapa 1 — Matemática mínima necessária

**O que estudar:**
- O que é uma equação de reta: `y = a + b*x`
- O que são variáveis dependentes e independentes
- O conceito de "erro" entre valor previsto e valor real

**Por quê:** A regressão linear é literalmente uma equação de reta. Se você entender `y = a + b*x`, já entendeu 70% do projeto.

**Buscar:** "equação da reta explicada", "o que é variável dependente e independente"

---

### Etapa 2 — O que é Machine Learning (nível básico)

**O que estudar:**
- Definição de ML (a citação do Mitchell no PDF resume bem)
- Diferença entre ML supervisionado e não supervisionado
- O que é um dataset, features e labels

**Por quê:** Contexto. Você precisa entender o vocabulário para não se perder.

**Buscar:** "machine learning para iniciantes", "supervised learning explained simply"

---

### Etapa 3 — Regressão Linear

**O que estudar:**
- O que é regressão linear simples (1 variável)
- O que são os parâmetros theta0 (intercepto) e theta1 (coeficiente)
- A função hipótese: `estimatePrice(km) = theta0 + theta1 * km`
- O que é a função de custo (Mean Squared Error)

**Por quê:** Essa é a estrutura central do projeto. Theta0 e theta1 são os "parâmetros que o modelo aprende".

**Buscar:** "linear regression explained", "what is theta0 theta1 linear regression", "mean squared error"

---

### Etapa 4 — Gradiente Descendente

**O que estudar:**
- O conceito intuitivo: "descer a montanha do erro"
- O que é learning rate (taxa de aprendizado)
- As fórmulas de atualização dos thetas (estão no PDF)
- Por que atualizar theta0 e theta1 **simultaneamente**

**Por quê:** Esse é o algoritmo de aprendizado. É o coração do projeto. Sem entender isso, você não consegue defender na avaliação.

**Buscar:** "gradient descent explained visually", "gradient descent linear regression from scratch", "why update theta simultaneously"

---

### Etapa 5 — Normalização de dados

**O que estudar:**
- O que é feature scaling / normalização
- Min-max normalization: `x_norm = (x - min) / (max - min)`
- Por que normalizar antes de rodar gradient descent

**Por quê:** Sem normalizar, os valores de km (ex: 50.000) e preço (ex: 8.000) têm escalas muito diferentes e o gradient descent diverge ou converge extremamente devagar. Isso não é exigido explicitamente pelo PDF, mas **é necessário na prática**.

**Buscar:** "feature normalization machine learning", "why normalize data before gradient descent"

---

### Etapa 6 — Leitura/escrita de arquivos e I/O

**O que estudar:**
- Como ler um arquivo CSV em Python (ou na linguagem escolhida)
- Como salvar e carregar valores de um arquivo
- Como pedir input do usuário no terminal

**Por quê:** O treinamento lê o dataset e salva os thetas. A predição carrega os thetas e pede o km ao usuário.

---

### Etapa 7 — (Bonus) Visualização e Métricas

**O que estudar:**
- Como plotar gráficos simples (matplotlib em Python)
- O que é o coeficiente de determinação R² (precisão do modelo)

**Por quê:** Os bônus exigem isso. Visualizar os dados facilita muito o debugging durante o desenvolvimento.

---

## Como construir o projeto (etapas e por quê)

### Fase 1 — Entender antes de codar
**Ação:** Leia sobre regressão linear e gradiente descendente. Faça no papel as contas com 3 pontos fictícios antes de escrever código.

**Por quê:** Se você não entendeu a matemática, o código vira copiar-e-colar que você não consegue defender.

---

### Fase 2 — Programa de Predição (o mais simples)
**Ação:** Crie `predict.py`.
- Carrega theta0 e theta1 de um arquivo (se não existir, usa 0 e 0)
- Pede o km ao usuário
- Calcula e imprime: `theta0 + theta1 * km`

**Por quê:** Começar pelo mais simples te dá confiança e um ponto de partida funcional. O treinamento vai precisar dessa função também.

---

### Fase 3 — Leitura do Dataset
**Ação:** Leia o CSV do dataset e armazene os valores de km e preço em listas/arrays.

**Por quê:** Você precisa dos dados antes de treinar.

---

### Fase 4 — Normalização
**Ação:** Normalize os valores de km (e opcionalmente de preço) para o intervalo [0, 1].

**Por quê:** Sem isso, o gradient descent provavelmente não vai convergir com os dados reais de carros.

---

### Fase 5 — Programa de Treinamento
**Ação:** Crie `train.py`.
- Inicialize theta0 = 0 e theta1 = 0
- Para cada iteração (ex: 1000 vezes):
  - Calcule `tmp_theta0` e `tmp_theta1` usando as fórmulas do PDF
  - Atualize theta0 e theta1 **ao mesmo tempo**
- Desnormalize os thetas (se normalizou os dados)
- Salve theta0 e theta1 em arquivo

**Por quê:** Esse é o coração do projeto. As fórmulas do PDF calculam o quanto cada theta precisa ser ajustado para diminuir o erro.

---

### Fase 6 — Testar tudo junto
**Ação:** Rode o treinamento → confira os thetas salvos → rode a predição com diferentes km → verifique se os preços fazem sentido.

**Por quê:** Integração. O projeto são dois programas que se comunicam via arquivo.

---

### Fase 7 — Bônus (se a parte obrigatória estiver perfeita)
**Ação:**
1. Plote os dados e a linha de regressão com matplotlib
2. Calcule o R² (coeficiente de determinação) como métrica de precisão

**Por quê:** O bônus só é avaliado se a parte obrigatória estiver 100%. Faça só depois.

---

## Funções que você deve implementar e por quê

### `estimatePrice(mileage, theta0, theta1)`
**O que faz:** Retorna `theta0 + (theta1 * mileage)`

**Por quê:** É a hipótese do modelo — a fórmula que o avaliador vai checar. Deve estar exatamente assim.

**Validações:**
- `mileage` deve ser um número (não string, não negativo inválido)
- Se theta0 e theta1 não existirem no arquivo, usar 0

---

### `gradient_descent(kms, prices, learning_rate, iterations)`
**O que faz:** Loop que calcula os deltas e atualiza theta0 e theta1

**Por quê:** Esse é o mecanismo de aprendizado. O avaliador vai verificar se você usa **as fórmulas corretas do PDF** e se atualiza os dois thetas **simultaneamente**.

**Validações:**
- Calcular `tmp_theta0` e `tmp_theta1` ANTES de atualizar qualquer um
- Nunca usar theta0 já atualizado para calcular theta1 na mesma iteração
- `m` nas fórmulas = número de exemplos no dataset

---

### `load_thetas(filepath)` e `save_thetas(filepath, theta0, theta1)`
**O que fazem:** Lê e escreve os parâmetros aprendidos em arquivo

**Por quê:** Os dois programas se comunicam via arquivo. Se o arquivo não existir, predição usa theta0 = theta1 = 0.

**Validações:**
- Tratar ausência do arquivo (FileNotFoundError)
- Tratar arquivo corrompido ou com formato errado

---

### `load_dataset(filepath)`
**O que faz:** Lê o CSV e retorna listas de km e preço

**Validações:**
- Arquivo existe
- Tem pelo menos 2 colunas
- Valores são numéricos
- Tratar linhas vazias ou malformadas

---

### `normalize(values)` e `denormalize(theta0, theta1, km_min, km_max, price_min, price_max)`
**O que fazem:** Escalam os dados para [0,1] antes do treino e revertem os thetas para a escala original

**Por quê:** Sem normalização, o gradient descent falha com dados reais. A desnormalização é necessária para os thetas funcionarem com o programa de predição (que trabalha com km real, não normalizado).

---

## Validações críticas e por quê

| Validação | Por quê é importante |
|-----------|---------------------|
| Theta0 = Theta1 = 0 quando arquivo não existe | O PDF exige esse comportamento explicitamente |
| Atualização simultânea dos thetas | Atualizar sequencialmente produz resultado matematicamente errado |
| Input de km deve ser número | Evitar crash por input inválido do usuário |
| Dataset deve existir e ser válido | Fail gracioso, não crash com traceback |
| Learning rate adequada | Muito alto: diverge. Muito baixo: nunca converge. Testar ~0.1 com dados normalizados |
| Número de iterações suficiente | Poucas iterações = modelo não aprende. Testar com 1000–10000 |

---

## Estimativa de tempo

| Etapa | Horas estimadas |
|-------|----------------|
| Estudar os conceitos (etapas 1–4) | 4–6h |
| Estudar normalização + I/O (etapas 5–6) | 1–2h |
| Programa de predição | 1h |
| Leitura do dataset | 1h |
| Normalização | 1–2h |
| Programa de treinamento | 3–5h |
| Testes e debugging | 2–4h |
| **Total parte obrigatória** | **~13–21h** |
| Bônus (gráfico + R²) | +3–5h |

**Ritmo real:** Para alguém aprendendo os conceitos do zero, espere de 2 a 4 semanas com ~1h por dia. Se dedicar blocos de 3–4h por vez, em uma semana intensa é possível.

---

## Pontos de atenção (o que costuma ferrar na avaliação)

1. **Não usar a função `estimatePrice` com esse nome e essa assinatura exata** — o avaliador vai checar.
2. **Usar `numpy.polyfit` ou similar** — isso é explicitamente proibido. Você deve implementar o gradient descent manualmente.
3. **Não atualizar os thetas simultaneamente** — erro clássico que produz resultados errados.
4. **Não tratar o caso onde o arquivo de thetas não existe** — o programa de predição deve funcionar mesmo sem ter treinado antes.
5. **Não normalizar os dados** — provavelmente o gradient descent vai divergir ou levar milhões de iterações.
6. **Não conseguir explicar o código** — o projeto permite usar AI como parceiro, mas você precisa defender cada decisão técnica na avaliação.

---

## Recursos recomendados

- **Gradient Descent visual:** buscar "gradient descent linear regression StatQuest" no YouTube (canal StatQuest with Josh Starmer)
- **Regressão linear from scratch:** buscar "linear regression from scratch python no libraries"
- **Normalização:** buscar "min max normalization machine learning"
- **Dataset do projeto:** provavelmente será fornecido como `data.csv` com colunas `km,price`

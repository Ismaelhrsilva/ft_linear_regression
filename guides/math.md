# math.md — A matemática do ft_linear_regression

> Guia de estudo passo a passo. Escrito em 2026-10-20.
> Objetivo: você conseguir **deduzir no papel** as duas fórmulas do subject e
> **defender cada uma** na avaliação — não só copiar.

---

## 0. Como usar este guia

- **Papel e lápis.** Cada seção tem exercícios com números pequenos. Faça na mão
  antes de olhar o gabarito (seção 13). Esse é o estudo — ler não basta.
- **Uma seção por vez, na ordem.** Cada uma usa a anterior.
- **Não escreva código durante este guia.** Os números dos exercícios viram, depois,
  os valores esperados dos seus testes.
- Onde aparece 🎯, é pergunta provável de defesa.

### Mapa

| # | Seção | O que você ganha | Tempo |
|---|-------|------------------|-------|
| 1 | Notação e reta | Ler as fórmulas do subject sem travar | 30 min |
| 2 | O modelo | Saber o que θ0 e θ1 significam no problema | 20 min |
| 3 | O erro (função de custo) | Saber o que "aprender" quer dizer | 45 min |
| 4 | A paisagem do custo | Intuição visual do gradiente descendente | 20 min |
| 5 | Derivadas (só o necessário) | A ferramenta da dedução | 1–2h |
| 6 | Deduzir as fórmulas do subject | O coração do projeto | 1h |
| 7 | Gradiente descendente na mão | Ver o algoritmo funcionar | 45 min |
| 8 | Learning rate e convergência | Escolher hiperparâmetros com critério | 30 min |
| 9 | Normalização (e desfazer ela) | O que mais trava gente no projeto | 1h |
| 10 | Solução exata (pra conferir) | Um gabarito independente | 30 min |
| 11 | Métricas de precisão (bônus) | O programa de precisão | 30 min |
| 12 | Perguntas de defesa | Checklist final | — |

Total: **~7–9h** de estudo. Dá pra espalhar em 3–4 dias.

### O que você NÃO precisa

Matrizes, álgebra linear, equação normal, estatística avançada, cálculo integral.
Se um vídeo ou artigo começar com matriz `X^T X`, pule — é a versão com várias features.

---

## 1. Notação e a equação da reta

### 1.1 Reta

```
y = θ0 + θ1 · x
```

| Símbolo | Nome | Significado geométrico |
|---------|------|------------------------|
| `θ0` | intercepto | onde a reta corta o eixo y (o valor de y quando x = 0) |
| `θ1` | inclinação (coeficiente angular) | quanto y muda quando x aumenta 1 |

- `θ1 > 0` → reta sobe. `θ1 < 0` → reta desce. `θ1 = 0` → reta horizontal.
- Dois pontos determinam uma reta: `θ1 = (y₂ − y₁) / (x₂ − x₁)`.

### 1.2 Índices — o subject conta de 0

O subject escreve `mileage[i]` e `price[i]` com `i` indo de `0` a `m − 1`.
Vamos abreviar:

```
xᵢ = mileage[i]        yᵢ = price[i]        m = número de linhas do dataset
```

### 1.3 Somatório Σ

```
 m−1
  Σ  aᵢ   =   a₀ + a₁ + a₂ + ... + a_{m−1}
 i=0
```

É literalmente um `for` que acumula. Duas regras que você vai usar:

```
Σ (aᵢ + bᵢ) = Σ aᵢ + Σ bᵢ          (soma se distribui)
Σ (c · aᵢ)  = c · Σ aᵢ             (constante sai pra fora)
```

E a média:

```
x̄ = (1/m) · Σ xᵢ
```

> O símbolo que aparece como `X` ou some no PDF impresso pelo Firefox é este `Σ`.

### ✏️ Exercícios 1

Use o **dataset de brinquedo** — ele aparece no guia inteiro:

| i | x | y |
|---|---|---|
| 0 | 1 | 2 |
| 1 | 2 | 3 |
| 2 | 3 | 5 |

- **1a.** Quanto vale `m`?
- **1b.** Calcule `Σ xᵢ`, `Σ yᵢ`, `x̄`, `ȳ`.
- **1c.** Calcule `Σ xᵢ · yᵢ` e `Σ xᵢ²`.
- **1d.** A reta `y = 1 + 2x` passa por quais pontos quando `x = 0`, `1`, `2`?

---

## 2. O modelo (a "hipótese")

O subject chama de `estimatePrice`. Em ML se chama **hipótese**, `h`:

```
h(x) = estimatePrice(mileage) = θ0 + θ1 · mileage
```

**"Treinar" = achar os valores de θ0 e θ1 que fazem essa reta passar o mais perto
possível dos pontos do `data.csv`.** Só isso. O resto do projeto é *como* achar.

No problema dos carros:

- `θ0` = preço estimado de um carro com **0 km** (um carro novo).
- `θ1` = quanto o preço **muda a cada km rodado**. Você espera `θ1 < 0`: quanto mais
  rodado, mais barato. A unidade de θ1 é "moeda por km" — vai ser um número pequeno,
  tipo centavos.

🎯 *"O que significa o seu θ1?"* — responda com unidade e sinal, não "é a inclinação".

### Regressão "simples"

Uma feature só (km). Por isso é uma reta, não um plano. "Linear" porque `h` é linear
nos parâmetros θ.

### ✏️ Exercícios 2

- **2a.** Com `θ0 = 0, θ1 = 0` (o estado inicial do subject), quanto o modelo prevê
  pra qualquer km? O que o `predict.py` deve imprimir antes de treinar?
- **2b.** Com `θ0 = 9000, θ1 = −0.02`, quanto vale um carro com 100.000 km?
  E com 500.000 km? Esse segundo valor faz sentido? (Guarde a resposta — vira
  discussão de defesa sobre extrapolação.)

---

## 3. Medindo o erro: a função de custo

Pra "achar a melhor reta" precisa de um número que diga **o quão ruim** uma reta é.

### 3.1 Resíduo

Pra cada ponto:

```
eᵢ = h(xᵢ) − yᵢ          (previsto − real)
```

Positivo = o modelo chutou alto. Negativo = chutou baixo.

### 3.2 Por que não só somar os resíduos?

Porque positivos e negativos se cancelam. Uma reta péssima pode ter soma zero.

Opções: valor absoluto `|eᵢ|` ou quadrado `eᵢ²`. Usamos o **quadrado** porque:

1. Sempre ≥ 0, então não cancela.
2. Pune erros grandes mais que proporcionalmente (errar 10 custa 100, não 10).
3. **É derivável em todo lugar.** `|e|` tem um bico em 0 que atrapalha a derivada.
   Esse é o motivo que importa pro gradiente descendente.

### 3.3 MSE e a função de custo J

**Erro quadrático médio** (Mean Squared Error):

```
MSE = (1/m) · Σ (h(xᵢ) − yᵢ)²
```

A função de custo que vamos minimizar é quase a mesma, com um `½` a mais:

```
                    1     m−1
J(θ0, θ1)  =  ───── ·  Σ  (θ0 + θ1·xᵢ − yᵢ)²
                   2m    i=0
```

- O `½` é cosmético: ele **cancela o 2** que aparece quando derivar um quadrado
  (seção 6). Multiplicar por constante não muda **onde** está o mínimo.
- J é função de **θ0 e θ1**, não de x. Os dados (xᵢ, yᵢ) são fixos; o que varia
  durante o treino são os thetas.

🎯 *"O que o seu treino minimiza?"* → J, o erro quadrático médio (com fator ½).

### ✏️ Exercícios 3 (dataset de brinquedo)

- **3a.** Calcule `J(0, 0)`.
- **3b.** Calcule `J(0, 1)` — a reta `y = x`.
- **3c.** Calcule `J(1, 1)` — a reta `y = 1 + x`.
- **3d.** Calcule `J(0, 1.5)`.
- **3e.** Qual das quatro retas é a melhor? Desenhe os 3 pontos e as 4 retas no papel.

---

## 4. A paisagem do custo

Imagine um mapa onde o chão é o plano (θ0, θ1) e a **altura** é J(θ0, θ1).

Como J é uma soma de quadrados de expressões lineares em θ, essa superfície é uma
**tigela** (paraboloide). Consequências:

- Tem **um único ponto mais baixo** — o mínimo global. Não existe "mínimo local"
  pra cair numa armadilha. (Em redes neurais existe; aqui não.)
- Em qualquer ponto da tigela, dá pra saber **pra que lado é descida**: é o lado
  oposto à inclinação.

Gradiente descendente = você está vendado na encosta da tigela; sente a inclinação
com o pé e dá um passo morro abaixo. Repete até o chão ficar plano.

```
J
│ ╲                     ╱
│  ╲                   ╱
│   ╲    passo →      ╱
│    ●──╮            ╱
│        ╲         ╱
│          ●──╮   ╱
│               ●╱   ← mínimo: inclinação = 0
└─────────────────────────── θ
```

A "inclinação sentida com o pé" é a **derivada**. Próxima seção.

🎯 *"Por que o gradiente descendente acha o mínimo certo e não um qualquer?"*
→ J é convexa (tigela): só tem um mínimo.

---

## 5. Derivadas — só o necessário

### 5.1 Intuição

A derivada de `f` em um ponto é a **inclinação da reta tangente** ali.
Diz: "se eu mexer um pouquinho no x, quanto o f muda, e em que direção".

- `f'(x) > 0` → f sobe se x aumenta → pra descer, **diminua** x.
- `f'(x) < 0` → f desce se x aumenta → pra descer, **aumente** x.
- `f'(x) = 0` → chão plano (mínimo, no nosso caso).

Ou seja: **pra descer, ande no sentido de −f'(x).** Esse sinal de menos é o
gradiente descendente inteiro.

### 5.2 As quatro regras que você precisa

| Regra | Fórmula | Exemplo |
|-------|---------|---------|
| Constante | `(c)' = 0` | `(7)' = 0` |
| Potência | `(xⁿ)' = n · xⁿ⁻¹` | `(x²)' = 2x`, `(x)' = 1` |
| Constante multiplicando | `(c · f)' = c · f'` | `(5x²)' = 10x` |
| Soma | `(f + g)' = f' + g'` | `(x² + 3x)' = 2x + 3` |

E a mais importante pra nós:

**Regra da cadeia** — derivada de uma função dentro de outra:

```
[ f(g(x)) ]'  =  f'(g(x)) · g'(x)
"derivada de fora (com o de dentro intacto)  ×  derivada de dentro"
```

Exemplo que é exatamente o padrão do projeto:

```
[ (3x − 5)² ]'  =  2·(3x − 5)  ·  3  =  6·(3x − 5)
                   └ de fora ┘   └ de dentro ┘
```

### 5.3 Derivada parcial

J tem **duas** variáveis. A derivada parcial em relação a θ0, escrita `∂J/∂θ0`,
é: **finja que θ1 é uma constante** e derive normalmente em θ0. E vice-versa.

```
f(a, b) = a² + 3ab + b²

∂f/∂a = 2a + 3b        (b tratado como número)
∂f/∂b = 3a + 2b        (a tratado como número)
```

O par `(∂J/∂θ0, ∂J/∂θ1)` se chama **gradiente**. Ele aponta pra subida mais íngreme;
o gradiente descendente anda no sentido contrário.

### ✏️ Exercícios 5

- **5a.** Derive `f(x) = 4x³ − 2x + 9`.
- **5b.** Derive `f(x) = (2x + 1)²` de duas formas: expandindo o quadrado primeiro,
  e pela regra da cadeia. Confira que dá o mesmo.
- **5c.** `f(θ) = (θ − 4)²`. Onde `f'(θ) = 0`? Isso é mínimo ou máximo?
- **5d.** `g(a, b) = (a + 2b − 7)²`. Calcule `∂g/∂a` e `∂g/∂b`.
  (Dica: regra da cadeia. A derivada "de dentro" em relação a `a` é 1; em relação a
  `b` é 2.)
- **5e.** No 5d, compare as duas respostas. Uma é a outra multiplicada por quê?
  **Guarde isso** — é o mesmo padrão da seção 6.

---

## 6. Deduzindo as fórmulas do subject

Esta é a seção que você precisa saber refazer na frente do avaliador.

### 6.1 Ponto de partida

```
J(θ0, θ1) = (1/2m) · Σ (θ0 + θ1·xᵢ − yᵢ)²
```

Chame o de dentro de `eᵢ = θ0 + θ1·xᵢ − yᵢ` (o resíduo). Então `J = (1/2m) · Σ eᵢ²`.

### 6.2 Derivada em relação a θ0

1. Constante `1/2m` sai pra fora; derivada da soma = soma das derivadas:
   ```
   ∂J/∂θ0 = (1/2m) · Σ ∂(eᵢ²)/∂θ0
   ```
2. Regra da cadeia em `eᵢ²`: de fora `2·eᵢ`, de dentro `∂eᵢ/∂θ0`.
3. `eᵢ = θ0 + θ1·xᵢ − yᵢ`. Tratando θ1, xᵢ, yᵢ como constantes: `∂eᵢ/∂θ0 = 1`.
4. Junta:
   ```
   ∂J/∂θ0 = (1/2m) · Σ 2·eᵢ·1 = (1/m) · Σ eᵢ
   ```
   O 2 cancelou o ½ — por isso o ½ estava lá.

```
∂J/∂θ0 = (1/m) · Σ (estimatePrice(xᵢ) − yᵢ)
```

### 6.3 Derivada em relação a θ1

Mesmos passos, só muda o passo 3: `∂eᵢ/∂θ1 = xᵢ` (θ1 está multiplicado por xᵢ).

```
∂J/∂θ1 = (1/m) · Σ (estimatePrice(xᵢ) − yᵢ) · xᵢ
```

### 6.4 Compare com o subject

```
tmpθ0 = learningRate · (1/m) · Σ (estimatePrice(mileage[i]) − price[i])
tmpθ1 = learningRate · (1/m) · Σ (estimatePrice(mileage[i]) − price[i]) · mileage[i]
```

**É exatamente o gradiente vezes o learning rate.** `tmpθ` é o tamanho do passo.

E a atualização (o subject não escreve o sinal, você tem que saber):

```
θ0 ← θ0 − tmpθ0
θ1 ← θ1 − tmpθ1
```

**Menos**, porque o gradiente aponta pra subida (seção 5.1).

🎯 *"De onde vem essa fórmula?"* → derivada parcial do erro quadrático médio em
relação a cada theta, pela regra da cadeia.
🎯 *"Por que o tmpθ1 tem `· mileage[i]` e o tmpθ0 não?"* → é a derivada de dentro:
`∂eᵢ/∂θ1 = xᵢ` e `∂eᵢ/∂θ0 = 1`.
🎯 *"O que é o m?"* (o subject provoca: "I let you guess") → o número de exemplos do
dataset; o `1/m` transforma a soma em média, pra o passo não depender do tamanho do
dataset.

### ✏️ Exercícios 6

- **6a.** Refaça as seções 6.2 e 6.3 numa folha em branco, sem olhar.
- **6b.** No dataset de brinquedo, calcule o gradiente em `θ0 = 0, θ1 = 0`.
  (Primeiro os resíduos `eᵢ`; depois `(1/m)Σeᵢ` e `(1/m)Σeᵢ·xᵢ`.)
- **6c.** Os dois componentes deram negativos. Pela seção 5.1, θ0 e θ1 devem
  aumentar ou diminuir? Faz sentido olhando os pontos?

---

## 7. Gradiente descendente na mão

### 7.1 O algoritmo (em português, não em código)

```
θ0 = 0, θ1 = 0
repita N vezes:
    calcule todos os resíduos eᵢ usando os θ ATUAIS
    tmpθ0 = α · (1/m) · Σ eᵢ
    tmpθ1 = α · (1/m) · Σ eᵢ · xᵢ
    θ0 = θ0 − tmpθ0
    θ1 = θ1 − tmpθ1
```

`α` = learning rate.

### 7.2 Atualização simultânea

"Simultaneamente" = **os dois tmp são calculados com os mesmos θ antigos**, antes de
qualquer um mudar. O erro clássico:

```
θ0 = θ0 − α·(...)          ← θ0 já mudou
θ1 = θ1 − α·(... usando o θ0 NOVO ...)   ← ERRADO
```

Isso calcula θ1 com um gradiente de um ponto que não é onde você estava. Não é mais
o gradiente de J; é outro algoritmo.

🎯 *"Por que atualizar simultaneamente?"* → o gradiente é a inclinação em **um**
ponto (θ0, θ1); os dois componentes têm que ser medidos no mesmo ponto.

### ✏️ Exercícios 7 (dataset de brinquedo, α = 0.1)

- **7a.** Faça a **iteração 1** a partir de (0, 0). Você já tem o gradiente do 6b.
  Quanto ficam θ0 e θ1?
- **7b.** Calcule `J` com os novos θ. Diminuiu em relação a `J(0,0)` do 3a?
- **7c.** Faça a **iteração 2**.
- **7d.** Agora faça a iteração 1 do **jeito errado** (atualiza θ0, recalcula os
  resíduos com o θ0 novo, aí atualiza θ1). Compare o θ1 com o do 7a.

---

## 8. Learning rate e convergência

### 8.1 O papel do α

- **α grande demais** → o passo pula o fundo da tigela, cai mais alto do outro lado,
  e a cada iteração piora. Os θ explodem pra `±inf` e depois `nan`. **Diverge.**
- **α pequeno demais** → passos minúsculos. Converge, mas leva uma eternidade.
- **α bom** → J cai rápido no começo e depois estabiliza.

```
α pequeno          α bom              α grande
  ╲                  ╲                 ╲     ╱
   ╲                  ╲                 ╲ ●─╱──●
    ●●●●●●●●→          ●─●─●●           ●╲╱  diverge
```

### 8.2 Quando parar

Duas estratégias, ambas defensáveis:

1. **Número fixo de iterações** (ex.: 1.000 ou 10.000). Simples.
2. **Critério de parada**: pare quando J quase não muda entre iterações
   (ex.: a diferença ficou menor que 1e-9). Mais elegante.

Em ambos, **acompanhe J ao longo das iterações**. Se J sobe, α está grande demais.
Plotar J × iteração é a melhor ferramenta de debug do projeto (e um bônus bonito).

### ✏️ Exercícios 8 (dataset de brinquedo)

Estes são pra testar **depois** que o `train.py` existir — ou na mão por 2–3 passos
se quiser sentir:

- **8a.** Com α = 0.3, os θ convergem? E com α = 0.4? E 0.5?
- **8b.** Por que existe um α máximo? (Pense na tigela: o que acontece se o passo for
  maior que a distância até o outro lado?)

---

## 9. Normalização — a parte que trava

### 9.1 O problema com os dados reais

No `data.csv`: km vai de **22.899 a 240.000**; preço de **3.650 a 8.290**.

Calcule o gradiente em (0, 0) com os dados reais:

```
∂J/∂θ0 ≈       −6.332      (média dos preços, com sinal trocado)
∂J/∂θ1 ≈ −582.902.525      (multiplicado pelos km, que são enormes)
```

O componente do θ1 é **~100.000× maior**. Mas o α é **um só** pros dois:

- α = 0.1 → o passo de θ1 é gigante → em **5 iterações** θ1 já está em ~10⁴⁴. Diverge.
- α = 1e-10 → θ1 fica estável, mas o passo de θ0 é minúsculo → depois de
  **100.000 iterações**, θ0 ≈ 0.018. Deveria ser ~8.500. Nunca chega.

Geometricamente: a tigela fica **esticada** — um vale estreitíssimo e comprido.
Qualquer α que serve num eixo é errado pro outro.

### 9.2 A solução: pôr x numa escala comparável

**Min-max** (leva pra [0, 1]):

```
x' = (x − min) / (max − min)
```

Alternativa: **z-score** `x' = (x − x̄) / σ` (média 0, desvio 1). Os dois funcionam;
min-max é mais fácil de explicar. Escolha um e saiba justificar.

Você treina com `x'` no lugar de x e obtém dois números — chame de **a** e **b**:

```
preço ≈ a + b · x'
```

### 9.3 O pulo do gato: desfazer a normalização

O `predict.py` tem que usar **exatamente** `θ0 + θ1 · mileage` com o km **cru**
(é o que o avaliador checa). Então a e b não servem direto — precisa convertê-los.

Substitua x' pela definição e reorganize:

```
preço ≈ a + b · (x − min) / (max − min)

      = a + (b / (max − min)) · x  −  b · min / (max − min)

      = [ a − b·min/(max − min) ]  +  [ b / (max − min) ] · x
        └──────── θ0 ────────────┘     └────── θ1 ──────┘
```

```
θ1 = b / (max − min)
θ0 = a − θ1 · min
```

**Deduza isso você mesmo** — é álgebra de ensino médio, e é a pergunta que mais
derruba gente na defesa.

> **Precisa normalizar o preço também?** Não é obrigatório. Normalizar só o km já
> resolve a divergência (o problema era a escala de x multiplicando o gradiente de θ1).
> Se quiser normalizar os dois, a desnormalização ganha mais termos — deduza com
> `y' = (y − ymin)/(ymax − ymin)` como exercício extra.

🎯 *"Por que você normalizou?"* → as escalas de θ0 e θ1 eram tão diferentes que
nenhum learning rate servia pros dois.
🎯 *"Mas o subject não fala em normalização. Você não mudou a fórmula?"* → não: o
algoritmo e a hipótese são os mesmos; mudei só a escala da entrada durante o treino,
e converti os θ de volta. O `predict` usa a hipótese do subject com km real.
🎯 *"Seus θ salvos funcionam com km normalizado ou cru?"* → cru.

### ✏️ Exercícios 9 (dataset de brinquedo)

- **9a.** Normalize os x `[1, 2, 3]` com min-max.
- **9b.** Suponha que o treino com x normalizado deu `a = 11/6 ≈ 1.8333` e `b = 3`.
  Desnormalize: quanto valem θ0 e θ1?
- **9c.** Confira: com os seus θ0, θ1 do 9b, quanto o modelo prevê pra x = 2?
  E com a, b usando x' de x = 2? Tem que dar igual.
- **9d.** Deduza a fórmula de desnormalização pro **z-score**.

---

## 10. Solução exata — um gabarito independente

Regressão linear simples tem fórmula fechada (mínimos quadrados):

```
        Σ (xᵢ − x̄)(yᵢ − ȳ)
θ1  =  ────────────────────          θ0 = ȳ − θ1 · x̄
          Σ (xᵢ − x̄)²
```

**Não use isso no `train.py`** — o subject exige gradiente descendente e o avaliador
checa "the use of the specified training function". Use só pra **conferir**: se seu
gradiente descendente convergiu, os θ dele têm que bater com estes até várias casas
decimais. É um teste ótimo.

(De onde vem: igualar ∂J/∂θ0 = 0 e ∂J/∂θ1 = 0 da seção 6 e resolver o sistema.
Fazer isso é um exercício excelente pra fixar a seção 6.)

### ✏️ Exercícios 10

- **10a.** Calcule θ0 e θ1 exatos pro dataset de brinquedo.
- **10b.** Calcule J com eles. Compare com todos os J da seção 3.
- **10c.** (Opcional, desafio) Deduza a fórmula fechada a partir de
  `∂J/∂θ0 = 0` e `∂J/∂θ1 = 0`.

---

## 11. Métricas de precisão (bônus)

O bônus pede "um programa que calcula a precisão do algoritmo". As opções clássicas:

| Métrica | Fórmula | Leitura |
|---------|---------|---------|
| MSE | `(1/m) Σ eᵢ²` | erro médio ao quadrado (unidade: moeda²) — difícil de interpretar |
| RMSE | `√MSE` | erro típico **na unidade do preço** |
| MAE | `(1/m) Σ |eᵢ|` | erro absoluto médio, na unidade do preço; menos sensível a outliers |
| R² | `1 − Σeᵢ² / Σ(yᵢ − ȳ)²` | fração da variação do preço que o km explica (0 a 1) |

**R²**, em português:

- `Σ(yᵢ − ȳ)²` = o erro de um modelo burro que sempre chuta a média.
- `Σeᵢ²` = o erro do seu modelo.
- `R² = 1 − (seu erro / erro do burro)`. R² = 1 → perfeito. R² = 0 → igual ao burro.

Recomendação: mostre **R²** e **RMSE**. Um diz "quão bom" e o outro "quanto erra em
dinheiro".

🎯 *"Seu R² é ~0.73. Isso é bom?"* → o km explica ~73% da variação do preço; o resto
vem de coisas que o modelo não vê (marca, ano, estado do carro). Pra uma única
feature, é razoável.

### ✏️ Exercícios 11

- **11a.** Calcule MSE, RMSE e R² da reta exata do 10a no dataset de brinquedo.

---

## 12. Checklist de defesa

Você deve conseguir responder sem olhar:

- [ ] O que é θ0 e θ1 **no problema dos carros** (com unidade e sinal)?
- [ ] O que é `m`?
- [ ] O que o treino minimiza? Por que o erro é **ao quadrado**?
- [ ] Deduzir no papel `∂J/∂θ0` e `∂J/∂θ1` (seção 6).
- [ ] Por que o `tmpθ1` tem `· mileage[i]`?
- [ ] Por que **subtrair** o tmpθ?
- [ ] O que é learning rate? O que acontece se for grande / pequeno demais?
- [ ] Por que atualizar θ0 e θ1 **simultaneamente**? O que dá errado se não?
- [ ] Por que normalizar? Por que não muda a hipótese do subject?
- [ ] Deduzir a desnormalização (seção 9.3).
- [ ] Como você sabe que convergiu?
- [ ] Por que o gradiente descendente não cai num mínimo errado? (convexidade)
- [ ] Qual a sua precisão e o que ela significa?
- [ ] O modelo prevê bem pra 500.000 km? (extrapolação — fora do intervalo dos dados,
      a reta pode até dar preço negativo)

---

## 13. Gabarito

> Só depois de tentar. Os números do dataset de brinquedo também servem de **valores
> esperados no `test.py`**.

<details>
<summary>Exercícios 1</summary>

- 1a. `m = 3`
- 1b. `Σx = 6`, `Σy = 10`, `x̄ = 2`, `ȳ = 10/3 ≈ 3.3333`
- 1c. `Σxy = 1·2 + 2·3 + 3·5 = 23`; `Σx² = 1 + 4 + 9 = 14`
- 1d. (0, 1), (1, 3), (2, 5)
</details>

<details>
<summary>Exercícios 2</summary>

- 2a. Prevê 0 pra qualquer km. O `predict.py` deve imprimir 0 antes de treinar.
- 2b. 100.000 km → `9000 − 2000 = 7000`. 500.000 km → `9000 − 10000 = −1000`.
  Preço negativo: a reta não vale fora do intervalo dos dados (extrapolação).
</details>

<details>
<summary>Exercícios 3</summary>

- 3a. resíduos `−2, −3, −5` → `(4 + 9 + 25)/6 = 38/6 ≈ 6.3333`
- 3b. resíduos `−1, −1, −2` → `(1 + 1 + 4)/6 = 1.0`
- 3c. resíduos `0, 0, −1` → `1/6 ≈ 0.1667`
- 3d. resíduos `−0.5, 0, −0.5` → `0.5/6 ≈ 0.0833`
- 3e. `y = 1.5x` (J menor). Mas nenhuma das quatro é a melhor possível — veja o 10b.
</details>

<details>
<summary>Exercícios 5</summary>

- 5a. `12x² − 2`
- 5b. Expandindo: `4x² + 4x + 1` → `8x + 4`. Cadeia: `2(2x + 1)·2 = 8x + 4`. ✓
- 5c. `f'(θ) = 2(θ − 4) = 0` → `θ = 4`. Mínimo (é uma parábola pra cima; f ≥ 0 e
  f(4) = 0).
- 5d. `∂g/∂a = 2(a + 2b − 7)`; `∂g/∂b = 4(a + 2b − 7)`
- 5e. `∂g/∂b = ∂g/∂a × 2` — o 2 é o coeficiente de b lá dentro. Na seção 6, o papel
  do "2" é feito pelo `xᵢ`.
</details>

<details>
<summary>Exercícios 6</summary>

- 6b. resíduos em (0,0): `−2, −3, −5`.
  `∂J/∂θ0 = (−2 −3 −5)/3 = −10/3 ≈ −3.3333`
  `∂J/∂θ1 = (−2·1 −3·2 −5·3)/3 = −23/3 ≈ −7.6667`
- 6c. Gradiente negativo → aumentar os dois. Faz sentido: a reta `y = 0` está abaixo
  de todos os pontos e é horizontal, mas os pontos sobem.
</details>

<details>
<summary>Exercícios 7</summary>

- 7a. `θ0 = 0 − 0.1·(−3.3333) = 0.3333`; `θ1 = 0 − 0.1·(−7.6667) = 0.7667`
- 7b. `J ≈ 1.2826` (caiu de 6.3333) ✓
- 7c. `θ0 ≈ 0.48`, `θ1 ≈ 1.1089` (J ≈ 0.2807). Iteração 3: `θ0 ≈ 0.5436`,
  `θ1 ≈ 1.2621` (J ≈ 0.0819).
- 7d. Jeito errado: `θ0 = 0.3333`, mas `θ1 = 0.7` (em vez de 0.7667). Já divergiu
  do algoritmo correto na primeira iteração.
</details>

<details>
<summary>Exercícios 8</summary>

- 8a. α = 0.3 converge (pra perto de θ0 = 0.333, θ1 = 1.5). α = 0.4 e 0.5 divergem
  (depois de 30 iterações já estão em centenas e dezenas de milhões, negativos).
- 8b. Se o passo é grande demais, você salta além do fundo e aterrissa mais alto do
  outro lado; o próximo gradiente é maior ainda, o passo também, e explode.
  O α máximo depende da curvatura da tigela — que depende da escala dos dados.
  (Por isso normalizar muda qual α funciona.)
</details>

<details>
<summary>Exercícios 9</summary>

- 9a. `min = 1`, `max = 3` → `x' = [0, 0.5, 1]`
- 9b. `θ1 = 3 / (3 − 1) = 1.5`; `θ0 = 11/6 − 1.5·1 = 1/3 ≈ 0.3333`
  (são os θ exatos do 10a — a normalização não muda a reta final.)
- 9c. `0.3333 + 1.5·2 = 3.3333`; `x' = 0.5` → `1.8333 + 3·0.5 = 3.3333` ✓
- 9d. `x' = (x − x̄)/σ` → `θ1 = b/σ`, `θ0 = a − b·x̄/σ = a − θ1·x̄`
</details>

<details>
<summary>Exercícios 10</summary>

- 10a. `x̄ = 2`, `ȳ = 10/3`. Desvios de x: `−1, 0, 1`. Desvios de y: `−4/3, −1/3, 5/3`.
  `Σ(x−x̄)(y−ȳ) = 4/3 + 0 + 5/3 = 3`; `Σ(x−x̄)² = 2` → `θ1 = 1.5`,
  `θ0 = 10/3 − 1.5·2 = 1/3 ≈ 0.3333`
- 10b. resíduos `−1/6, 1/3, −1/6` → `J = (1/36 + 4/36 + 1/36)/6 = 1/36 ≈ 0.0278`.
  Menor que todas as da seção 3, como tinha que ser.
- 10c. `∂J/∂θ0 = 0` → `θ0 = ȳ − θ1·x̄`. Substitui em `∂J/∂θ1 = 0` e isola θ1.
</details>

<details>
<summary>Exercícios 11</summary>

- 11a. `Σe² = 1/6` → `MSE = 1/18 ≈ 0.0556`, `RMSE ≈ 0.2357`.
  `Σ(y − ȳ)² = 16/9 + 1/9 + 25/9 = 42/9 ≈ 4.6667` → `R² = 1 − (1/6)/(42/9) ≈ 0.9643`
</details>

<details>
<summary>⚠️ Valores do data.csv real — só pra conferir o train.py pronto</summary>

Calculados pela fórmula fechada (seção 10). Seu gradiente descendente deve chegar
perto disto:

- `θ0 ≈ 8499.5996`
- `θ1 ≈ −0.0214490`
- 100.000 km → ≈ 6354.70
- Precisão: `R² ≈ 0.7330`, `RMSE ≈ 667.57`, `MAE ≈ 557.84`

Referência de convergência (x normalizado por min-max, só o km, α = 0.1):
100 iterações → longe (θ0 ≈ 6519); 1.000 → perto (θ0 ≈ 8477);
10.000 → bate com o exato.
</details>

---

## 14. Recursos (na ordem em que ajudam)

Busque pelo nome — são canais e cursos conhecidos e estáveis.

1. **StatQuest with Josh Starmer** (YouTube) — *"Linear Regression, Clearly
   Explained"*, *"Gradient Descent, Step-by-Step"*, *"The Chain Rule"*. O melhor ponto
   de entrada pras seções 3, 6 e 7. Faz as contas na tela, devagar.
2. **3Blue1Brown** (YouTube) — série *"Essence of Calculus"*, capítulos 1–4
   (derivada, regras, regra da cadeia). Pra seção 5, se derivada estiver enferrujada.
3. **Khan Academy** — *Derivadas: regras básicas* e *regra da cadeia*. Exercícios com
   correção automática; bom pra treinar a seção 5.
4. **Andrew Ng — Machine Learning Specialization** (Coursera, curso 1, semana 1).
   É **exatamente** este projeto: mesma notação de θ, mesmo J com ½, mesmas fórmulas.
   Dá pra assistir só as aulas sem pagar.
5. **StatQuest** — *"R-squared, Clearly Explained"*. Pra seção 11.

Sugestão de encaixe: assista o StatQuest de gradient descent **depois** de tentar a
seção 6 sozinho — vai fazer muito mais sentido.

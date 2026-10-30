# starting.md — ft_linear_regression

> Documento de retomada. Escrito em 2026-09-08, atualizado em 2026-10-20.
> Se o contexto se perdeu, comece lendo daqui.
> **Recomeço: 2026-10-21.** Estudo da matemática: ver `math.md`.

---

## 1. Verificação do subject (RESOLVIDO — não precisa refazer)

O PDF `linear_regression.pdf` no disco foi sobrescrito em 2026-09-08 22:07 por uma
versão impressa do Firefox. Surgiu a dúvida: a versão do subject mudou?

**Não mudou. Ambos são Version: 4.1.**

Como foi verificado:

| | PDF antigo | PDF novo |
|---|---|---|
| Origem | banco de objetos do git, blob `527e5eb1` (commit `09c1a07`, 2026-06-28) | disco, sobrescrito hoje |
| `Creator` interno | LaTeX with hyperref | Mozilla Firefox 155.0.1 |
| `CreationDate` interna | Thu Dec 18 15:18:51 2025 | Tue Sep 8 22:07:05 2026 |
| Tamanho | 1.353.377 bytes | 535.654 bytes |
| Páginas | 11 | 11 |
| **Versão declarada** | **4.1** | **4.1** |

O antigo foi extraído com `git show HEAD:linear_regression.pdf` — do banco de objetos,
não do disco. Sobrescrever o arquivo não altera esse blob, então a comparação é válida.

O `diff` do texto extraído dos dois só acusou **artefatos de extração**, não mudança de
conteúdo: o PDF do Firefox perde as ligaduras tipográficas (`first` → `rst`,
`definition` → `denition`, `efficiency` → `eciency`) e destrói os símbolos de
somatório `Σ` das fórmulas do gradiente. Conteúdo idêntico palavra por palavra.

**Consequência prática:** o PDF que está no disco é ruim de ler justamente nas fórmulas
do gradiente descendente. Se precisar delas, use a versão do git:
`git show HEAD:linear_regression.pdf > /tmp/subject_original.pdf`
(Decisão tomada: NÃO restaurar o arquivo no disco. Foi escolha consciente do Ismael.)

**Atualização 2026-10-20:** o Ismael baixou de novo o subject da intra
(`~/Downloads/en.subject_ft_linear.pdf`). É **idêntico byte a byte** ao PDF do git
(md5 `50adb95c…`) — mesma Version 4.1. O `data.csv` baixado junto também é idêntico
ao da pasta. Nada mudou no subject desde junho.

---

## 2. Modo de trabalho acordado com o Claude

Loop de colaboração, nesta ordem:

1. Claude **ensina o conceito** e indica o que estudar
2. Claude **escreve os testes** (`test.py`) — a rede de segurança
3. **Ismael escreve o código de produção** (`data.py`, `train.py`, ...)
4. Claude **revisa e dá feedback**
5. Ismael edita → volta ao passo 4 até passar

**Claude não escreve o código de produção.** Ismael escreve. Isso está alinhado com o
Capítulo III do subject ("AI Instructions"), que exige que o aluno mantenha a liderança
intelectual e consiga defender cada decisão na peer-evaluation.

---

## 3. Roadmap (revisado em 2026-10-20)

Meta: **obrigatório em ~1 semana** com 1–2h/dia, em paralelo com um exercício por dia
da piscine Python. Até 2 semanas se a matemática pedir mais calma — vale a pena,
porque a defesa pergunta de onde saem as fórmulas.

A lógica do Ismael continua: **esqueleto rodando ponta a ponta antes da matemática.**
Detalhe que ajuda: o subject manda os thetas começarem em 0, então o `predict.py`
funciona (prevê 0) antes de existir qualquer treino. A "conexão" entre os dois
programas é só um arquivo com `theta0` e `theta1`.

| Dia | Código (Ismael escreve) | Estudo (`math.md`) |
|-----|-------------------------|--------------------|
| 1 | `data.py` — `load_data` (seção 4 abaixo) | §1–2: notação, reta, o modelo |
| 2 | "Fake algoritmo": `predict.py` lê o arquivo de thetas (0, 0 se não existir), pede km, valida, imprime `θ0 + θ1·km`. Plot dos pontos do CSV (bônus 1) | §3–4: função de custo |
| 3 | — (dia de papel) | §5–6: derivadas e a dedução das fórmulas |
| 4 | `train.py`: loop do gradiente descendente, sem normalização ainda — **ver ele divergir** | §7–8: GD na mão, learning rate |
| 5 | Normalização + desnormalização, salvar thetas, testar `predict` com eles | §9–10: normalização e solução exata pra conferir |
| 6 | Bônus: reta no gráfico, curva J × iteração, programa de precisão | §11: métricas |
| 7 | Folga, revisão, `flake8`, ensaio de defesa | §12: checklist |

Em todo dia de código vale o loop da seção 2: Claude escreve o `test.py` daquele
pedaço (os números do dataset de brinquedo do `math.md` servem de valores esperados),
Ismael escreve o código, Claude revisa.

### Pontos de atenção (resumo — detalhe no `math.md`)

1. **Normalização** (§9): km até 240.000 faz o gradiente de θ1 ser ~100.000× o de θ0.
   Sem normalizar, diverge ou não converge nunca. Treinar com km normalizado e
   **desnormalizar os thetas** — o `predict` usa a hipótese do subject com km cru.
2. **Atualização simultânea** (§7.2): os dois `tmpθ` com os thetas antigos.
3. **`m`** = número de linhas do dataset (24).
4. **Bibliotecas**: nada que faça a regressão (`numpy.polyfit`, `sklearn`,
   `scipy.stats.linregress`). Leitura do CSV: decisão já tomada de usar só `csv` da
   stdlib (seção 4). `matplotlib` pro bônus é ok.
5. **Capítulo III (IA)**: saber explicar cada decisão. Por isso o modo da seção 2.

---

## 4. Tarefa do DIA 1 (aprovada, ainda NÃO iniciada)

### Contrato de `load_data`

```python
load_data("data.csv")  ->  ([240000.0, 139800.0, ...], [3650.0, 3800.0, ...])
```

Uma função, dois retornos: lista de kms e lista de prices, ambas de `float`, mesma
ordem, mesmo tamanho.

### Decisões e o porquê de cada uma

- **Duas listas paralelas** (e não lista de tuplas ou dict) — casa direto com as
  fórmulas do subject, que indexam `mileage[i]` e `price[i]`. No loop do gradiente
  você vai iterar por índice.
- **`float` e não `int`** — os preços no CSV são inteiros hoje, mas o gradiente vai
  produzir números quebrados. Converter já na entrada evita surpresa lá na frente.
- **Pular o cabeçalho `km,price`** — se não pular, `float("km")` levanta `ValueError`.
- **Módulo separado `data.py`** (e não tudo dentro do `train.py`) — amanhã o `train.py`
  faz só `from data import load_data`. Zero retrabalho, e o `train.py` não vira uma
  mistura de leitura de CSV com matemática.
- **Só a stdlib (`csv`)** — o subject proíbe bibliotecas que façam o trabalho por você.
  `pandas` aqui seria peso morto e ponto de atrito na defesa.

### Conceitos a estudar (Claude explica antes de você escrever)

1. Abrir arquivo com `with open(...)` — e por que `with` em vez de `open`/`close` na mão
2. Módulo `csv` da stdlib — `csv.reader` vs `csv.DictReader`
3. Conversão de string pra float — por que CSV só entrega texto
4. Construir listas em paralelo dentro de um loop
5. Erros que valem tratar já (arquivo inexistente, linha malformada) e quais não valem ainda

### Arquivos

| Arquivo | Quem escreve | Estado |
|---------|-------------|--------|
| `test.py` | **Claude** — testes unitários de `load_data` com `unittest` (stdlib) | não criado |
| `data.py` | **Ismael** | não criado |
| `data.csv` | ninguém — não tocar | existe, 24 linhas + cabeçalho |

### Como validar

`python3 test.py` verde, mais um print dos dados na tela pra bater o olho.

### Fora do escopo do dia 1

Nada de normalização, nada de theta, nada de gradiente descendente.

---

## 5. Estado atual do repositório

```
data.csv                 24 pares km,price + cabeçalho
GUIDE.md                 guia longo em PT-BR sobre o projeto (escrito antes)
math.md                  guia de estudo da matemática, passo a passo, com exercícios e gabarito (2026-10-20)
linear_regression.pdf    subject v4.1 (versão Firefox, ligaduras quebradas; modificado em relação ao git)
predict.py               stub: só pede mileage e valida se é dígito
train.py                 vazio (0 bytes)
starting.md              este arquivo
```

Commits: `09c1a07 Starting`, `3fe11a0 docs: adiciona starting.md`.

---

## 6. Próximo passo quando retomar (2026-10-21)

1. Ler `math.md` §1–2 e fazer os exercícios no papel.
2. Claude explica os 5 conceitos da seção 4 e escreve o `test.py` de `load_data`.
3. Ismael escreve o `data.py`. Depois entra o loop de revisão.

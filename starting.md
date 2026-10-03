# starting.md — ft_linear_regression

> Documento de retomada. Escrito em 2026-09-08.
> Se o contexto se perdeu, comece lendo daqui.

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

## 3. Roadmap da semana (plano do Ismael)

| Dia | Objetivo |
|-----|----------|
| 1 | Ler `data.csv`, printar na tela, guardar em estruturas úteis pro algoritmo |
| 2 | "Fake algoritmo" — uma continha idiota só pra fechar o encaixe: entrada → cálculo → output. Define ONDE o algoritmo real vai morar |
| 3+ | O algoritmo de verdade: gradiente descendente |

A lógica por trás disso: fazer o esqueleto rodar ponta a ponta antes de encarar a
matemática. Bom instinto — o dia 2 elimina todo risco de integração antes de você ter
que pensar em convergência.

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
linear_regression.pdf    subject v4.1 (versão Firefox, ligaduras quebradas)
predict.py               stub: só pede mileage e valida se é dígito
train.py                 vazio (0 bytes)
starting.md              este arquivo
```

Único commit: `09c1a07 Starting`.

---

## 6. Próximo passo quando retomar

Claude explica os 5 conceitos da seção 4 e escreve o `test.py`.
Ismael escreve o `data.py`. Depois entra o loop de revisão.

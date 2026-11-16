# Classificação Temporal e Mitigação de Concept Drift em Sensores de Gás

Projeto de Aprendizado de Máquina em **R** que classifica gases a partir de um nariz eletrônico (*e-nose*) e avalia como o desempenho dos modelos se degrada com o tempo devido ao *sensor drift* (*concept drift*).

## Contexto

O dispositivo utiliza uma matriz de 16 sensores semicondutores de óxido metálico (MOS). Ao entrar em contato com um gás, a resistência elétrica de cada sensor se altera, gerando uma "impressão digital olfativa" da substância.

Sensores físicos sofrem degradação contínua: o aquecimento constante (acima de 300 °C), as flutuações de umidade e a contaminação da camada de óxido mudam a resposta elétrica ao longo do tempo. Assim, a assinatura elétrica da Amônia no mês 1 é estatisticamente diferente da do mês 36, e um modelo que não é atualizado perde acurácia com o passar dos meses.

O objetivo é percorrer o ciclo completo de um problema real de ML, enfrentando a quebra da premissa de distribuição estática e comparando estratégias adaptativas de treinamento.

## Base de dados

[Gas Sensor Array Drift Dataset](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset) (UCI Machine Learning Repository), de Vergara et al. (2012).

| Item | Descrição |
|---|---|
| Janela temporal | 36 meses, organizados cronologicamente em 10 lotes (*batches*) |
| Volume | 13.910 medições |
| Classes (6 gases) | Etanol, Etileno, Amônia, Acetaldeído, Acetona e Tolueno |
| Atributos | 128 variáveis contínuas: 16 sensores × 8 características da curva de resposta (amplitude em estado estacionário e dinâmica transiente de subida e recuperação) |

Os arquivos estão no formato LIBSVM (`classe indice:valor indice:valor ...`).

## Metodologia

A avaliação respeita a ordem temporal dos dados, e **não** utiliza K-Fold Cross-Validation tradicional nem divisão aleatória global, pois isso causaria vazamento de dados (*data leakage*) do futuro para o passado.

Três abordagens são comparadas:

| Estratégia | Treino na iteração *k* (*k* = 1…9) | Teste |
|---|---|---|
| **Linha de Base** | Lote 1 (fixo, sem retreinamento) | Lotes 2 a 10 |
| **Janela Expandida** | Lotes 1 a *k* | Lote *k*+1 |
| **Janela Deslizante** | Lote *k* | Lote *k*+1 |

Regras seguidas no código:

- **Pré-processamento isolado:** normalização (escore-z), remoção de atributos constantes e seleção por correlação são ajustadas apenas com o treinamento de cada iteração e somente aplicadas ao teste.
- **Tuning sem vazamento:** a validação cruzada para estimar hiperparâmetros é feita estritamente dentro do conjunto de treinamento da iteração.
- **Reprodutibilidade:** todas as rotinas aleatórias usam `set.seed()` com a semente definida no topo do script.

### Classificadores

Dois classificadores de famílias distintas, ambos via `caret`:

- **LDA** (`method = "lda"`): linear, generativo e paramétrico.
- **MLP** (`method = "nnet"`): rede neural não linear, discriminativa.

### Métricas

Acurácia e F1-Score Macro por lote de teste, além do F1 por classe e das matrizes de confusão.

## Estrutura do repositório

```
.
├── main.r                 # Pipeline completo (carga, preparação, experimentos e resultados)
├── Dataset/               # Arquivos batch1.dat ... batch10.dat (não versionada)
├── figuras/               # Gráficos gerados pelo script
├── resultados/            # Tabelas e objetos gerados pelo script
└── README.md
```

As pastas `figuras/` e `resultados/` são criadas automaticamente na execução.

## Como executar

### 1. Requisitos

- [R](https://cran.r-project.org/) (versão 4.x recomendada) e, de preferência, o [RStudio](https://posit.co/download/rstudio-desktop/).
- Pacotes R: `caret`, `ggplot2`, `tictoc`. O `caret` usa os pacotes `MASS` (LDA) e `nnet` (MLP), que acompanham a instalação padrão do R.

```r
install.packages(c("caret", "ggplot2", "tictoc"))
```

### 2. Dados

Baixe o dataset no [repositório da UCI](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset) e coloque os 10 arquivos na pasta `Dataset/`, na raiz do projeto:

```
Dataset/batch1.dat
Dataset/batch2.dat
...
Dataset/batch10.dat
```

### 3. Execução

1. Abra o `main.r` no RStudio.
2. Defina o diretório de trabalho como a raiz do projeto (a pasta que contém `Dataset/`), por exemplo com `Session > Set Working Directory > To Source File Location` ou descomentando e ajustando a linha `#setwd("")` no início do script.
3. Execute o script inteiro (`Ctrl + Shift + Enter`, ou `Source`).

> **Tempo de execução:** o treinamento da MLP com validação cruzada interna nas janelas maiores (principalmente na Janela Expandida) pode levar vários minutos. O progresso é impresso no console e o tempo de cada etapa é medido com `tictoc`.

### 4. Saídas

| Arquivo | Conteúdo |
|---|---|
| `figuras/distribuicao_classes_lote.png` | Distribuição das classes em cada lote |
| `figuras/baseline_degradacao.png` | Degradação temporal do modelo treinado só no Lote 1 |
| `figuras/comparacao_acuracia.png` | Acurácia por lote de teste, comparando as três estratégias |
| `figuras/comparacao_f1macro.png` | F1 Macro por lote de teste, comparando as três estratégias |
| `resultados/metricas_gerais.csv` | Acurácia e F1 Macro de todas as iterações |
| `resultados/metricas_por_classe.csv` | F1 por classe em cada iteração |
| `resultados/distribuicao_treino_teste.csv` | Amostras por classe nos conjuntos de treino e teste de cada iteração |
| `resultados/matrizes_confusao.rds` | Lista com as matrizes de confusão (`readRDS()` para carregar) |

## Referência

Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., Huerta, R. (2012). *Chemical gas sensor drift compensation using classifier ensembles*. Sensors and Actuators B: Chemical, 166–167, 320–329.

################################################################################
# Classificação Temporal e Mitigação de Concept Drift em Sensores de Gás
#
# Objetivo : classificar 6 gases (Etanol, Etileno, Amônia, Acetaldeído, Acetona
#            e Tolueno) a partir de uma matriz de 16 sensores MOS (128
#            atributos) e avaliar a degradação do modelo ao longo de 36 meses
#            (10 lotes), comparando Linha de Base, Janela Expandida e Janela
#            Deslizante com um classificador LDA e uma rede neural MLP
#
# Dataset  : Gas Sensor Array Drift Dataset (UCI) - arquivos batch1.dat a
#            batch10.dat, na pasta "Dataset"
################################################################################

# Configuração do diretório de Trabalho (troque pelo caminho no seu computador,
# a pasta deve conter o diretório "Dataset")
#setwd("")

# Remoção de qualquer variável do ambiente
rm(list=ls())

# Bibliotecas necessárias
library(caret)
library(ggplot2)
library(tictoc)

# Pastas onde serão salvos as figuras e os resultados
dir.create("figuras", showWarnings = FALSE)
dir.create("resultados", showWarnings = FALSE)

# Semente utilizada em todas as rotinas que envolvem aleatoriedade
semente = 4561

################################################################################
# 1. LEITURA DOS DADOS                                                         #
################################################################################

nomesClasses = c("Etanol", "Etileno", "Amonia", "Acetaldeido", "Acetona",
                 "Tolueno")

listaLotes = list()

# Cada arquivo está no formato "classe indice:valor indice:valor ..."
for (i in 1:10) {
  linhas = readLines(paste0("./Dataset/batch", i, ".dat"))

  # Separa a classe dos pares "indice:valor" e mantém apenas os valores
  dadosLote = t(sapply(linhas, function(linha) {
    partes  = unlist(strsplit(linha, "\\s+"))
    valores = as.numeric(sub("^\\d+:", "", partes[-1]))
    return(c(as.numeric(partes[1]), valores))
  }))
  rownames(dadosLote) = NULL

  dfLote = as.data.frame(dadosLote)
  colnames(dfLote) = c("Classe", paste0("V", 1:(ncol(dfLote) - 1)))

  # Identificação do lote ao qual cada amostra pertence
  dfLote$Lote = i

  listaLotes[[i]] = dfLote
  cat("Lote", i, "carregado. Amostras:", nrow(dfLote), "\n")
}

# Junta os 10 lotes em uma única tabela
dados = do.call(rbind, listaLotes)
rm(listaLotes, dfLote, dadosLote, linhas, i)

# Estrutura do dataset (esperado: 13910 amostras e 130 colunas)
dim(dados)

################################################################################
# 2. PREPARAÇÃO DOS DADOS                                                      #
################################################################################

################################################################################
# 2.1 PREPARAÇÃO DA ESTRUTURA

# A variável alvo precisa ser factor para o caret tratar o problema como classificação
dados$Classe = factor(dados$Classe, levels = 1:6, labels = nomesClasses)

# Nomes das variáveis de entrada (os 128 atributos)
atributos = paste0("V", 1:128)

# Total de amostras por lote
table(dados$Lote)

# Total de amostras por classe em cada lote
tabelaDistribuicao = table(dados$Lote, dados$Classe)
tabelaDistribuicao

# Percentual de amostras por classe em cada lote
round(prop.table(tabelaDistribuicao, margin = 1) * 100, 1)

# Gráfico de barras: classes por lote
dfDistribuicao = as.data.frame(tabelaDistribuicao)
colnames(dfDistribuicao) = c("Lote", "Classe", "Amostras")

ggplot(dfDistribuicao, aes(x = Classe, y = Amostras, fill = Lote)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_brewer(palette = "Paired") +
  ggtitle("Distribuição de Classes por Lote")
ggsave("figuras/distribuicao_classes_lote.png", width = 9, height = 5)

rm(tabelaDistribuicao, dfDistribuicao)

################################################################################
# 2.2 DIVISÃO EM TREINAMENTO E TESTE

# ATENÇÃO: não é feito holdout aleatório nem validação cruzada global, pois
# isso levaria dados do futuro para o treinamento (vazamento de dados). A
# divisão respeita a ordem cronológica dos lotes, e a normalização e a seleção
# de características são estimadas sempre apenas com o treinamento da iteração
#
#   Linha de Base : treina no Lote 1            -> testa nos Lotes 2 a 10
#   Expandida     : treina nos Lotes 1..k       -> testa no Lote k+1
#   Deslizante    : treina apenas no Lote k     -> testa no Lote k+1

# Validação cruzada interna, usada só para estimar os hiperparâmetros com os
# dados de treinamento da iteração
paramsTreinamento = trainControl(method = "cv", number = 5)

# Hiperparâmetros testados na rede neural MLP
grid_mlp = expand.grid(size = c(5, 10), decay = c(0.01, 0.1))

# Data frame com as medidas de avaliação de todos os experimentos
resultado = data.frame(Estrategia=character(), Classificador=character(),
                       LoteTeste=integer(), AmostrasTreino=integer(),
                       AmostrasTeste=integer(), Acuracia=double(),
                       F1Macro=double(), stringsAsFactors = FALSE)

# Data frame com o F1 de cada classe
resultado.classes = data.frame(Estrategia=character(),
                               Classificador=character(), LoteTeste=integer(),
                               Classe=character(), F1=double(),
                               stringsAsFactors = FALSE)

# Data frame com a distribuição das classes em cada conjunto
distribuicao = data.frame(Estrategia=character(), LoteTeste=integer(),
                          Conjunto=character(), Classe=character(),
                          Amostras=integer(), stringsAsFactors = FALSE)

# Matrizes de confusão (a chave é Estrategia_Classificador_LoteTeste)
matrizes = list()

################################################################################
# 3. EXPERIMENTO DE REFERÊNCIA (LINHA DE BASE)                                 #
################################################################################

# Treinamento apenas com o Lote 1
dadosTreinamento   = dados[dados$Lote == 1, atributos]
rotulosTreinamento = dados$Classe[dados$Lote == 1]

# Normalização estimada no Lote 1
normalizacaoParametros = preProcess(dadosTreinamento,
                                    method = c("zv", "center", "scale"))
dadosTreinamento = predict(normalizacaoParametros, dadosTreinamento)

# Remoção das variáveis fortemente correlacionadas (estimada no Lote 1)
indicesCorrelacaoForte = findCorrelation(cor(dadosTreinamento), cutoff = 0.95)
dadosTreinamento[,indicesCorrelacaoForte] = NULL

# Modelo LDA
set.seed(semente)
tic("Baseline - LDA")
lda.baseline = train(dadosTreinamento, rotulosTreinamento, method = "lda",
                     trControl = paramsTreinamento)
toc()

# Modelo MLP
set.seed(semente)
tic("Baseline - MLP")
mlp.baseline = train(dadosTreinamento, rotulosTreinamento, method = "nnet",
                     trControl = paramsTreinamento, tuneGrid = grid_mlp,
                     MaxNWts = 20000, maxit = 200, trace = FALSE)
toc()

# Avaliação sequencial nos Lotes 2 a 10, sem nenhum retreinamento
for (lote in 2:10) {
  rotulosTeste = dados$Classe[dados$Lote == lote]
  dadosTeste   = dados[dados$Lote == lote, atributos]

  # Apenas aplica a normalização e a seleção estimadas no Lote 1
  dadosTeste = predict(normalizacaoParametros, dadosTeste)
  dadosTeste[,indicesCorrelacaoForte] = NULL

  # - LDA ----
  predicao = factor(predict(lda.baseline, dadosTeste), levels = nomesClasses)
  res = confusionMatrix(predicao, rotulosTeste)
  f1 = res$byClass[,"F1"]
  f1[is.na(f1) & table(rotulosTeste) > 0] = 0

  resultado[nrow(resultado)+1,] = data.frame("Baseline", "LDA", lote,
                                             nrow(dadosTreinamento),
                                             nrow(dadosTeste),
                                             res$overall["Accuracy"],
                                             mean(f1, na.rm = TRUE))
  resultado.classes = rbind(resultado.classes,
                            data.frame(Estrategia = "Baseline",
                                       Classificador = "LDA", LoteTeste = lote,
                                       Classe = nomesClasses,
                                       F1 = as.numeric(f1)))
  matrizes[[paste0("Baseline_LDA_", lote)]] = res$table

  # - MLP ----
  predicao = factor(predict(mlp.baseline, dadosTeste), levels = nomesClasses)
  res = confusionMatrix(predicao, rotulosTeste)
  f1 = res$byClass[,"F1"]
  f1[is.na(f1) & table(rotulosTeste) > 0] = 0

  resultado[nrow(resultado)+1,] = data.frame("Baseline", "MLP", lote,
                                             nrow(dadosTreinamento),
                                             nrow(dadosTeste),
                                             res$overall["Accuracy"],
                                             mean(f1, na.rm = TRUE))
  resultado.classes = rbind(resultado.classes,
                            data.frame(Estrategia = "Baseline",
                                       Classificador = "MLP", LoteTeste = lote,
                                       Classe = nomesClasses,
                                       F1 = as.numeric(f1)))
  matrizes[[paste0("Baseline_MLP_", lote)]] = res$table

  # Distribuição das classes nos conjuntos desta iteração
  distribuicao = rbind(distribuicao,
    data.frame(Estrategia = "Baseline", LoteTeste = lote,
               Conjunto = "Treinamento", Classe = nomesClasses,
               Amostras = as.integer(table(rotulosTreinamento))),
    data.frame(Estrategia = "Baseline", LoteTeste = lote,
               Conjunto = "Teste", Classe = nomesClasses,
               Amostras = as.integer(table(rotulosTeste))))
}

rm(dadosTreinamento, rotulosTreinamento, normalizacaoParametros,
   indicesCorrelacaoForte, lda.baseline, mlp.baseline)

################################################################################
# 4. VALIDAÇÃO POR FORWARD CHAINING                                            #
################################################################################

################################################################################
# 4.1. ESTRATÉGIA A - JANELA EXPANDIDA

tic("Janela Expandida")

for (k in 1:9) {
  # Treinamento com os Lotes 1 a k e teste no Lote k+1
  dadosTreinamento   = dados[dados$Lote <= k, atributos]
  rotulosTreinamento = dados$Classe[dados$Lote <= k]
  dadosTeste         = dados[dados$Lote == k + 1, atributos]
  rotulosTeste       = dados$Classe[dados$Lote == k + 1]

  # Normalização estimada apenas no treinamento
  normalizacaoParametros = preProcess(dadosTreinamento,
                                      method = c("zv", "center", "scale"))
  dadosTreinamento = predict(normalizacaoParametros, dadosTreinamento)
  dadosTeste       = predict(normalizacaoParametros, dadosTeste)

  # Seleção de características estimada apenas no treinamento
  indicesCorrelacaoForte = findCorrelation(cor(dadosTreinamento), cutoff = 0.95)
  dadosTreinamento[,indicesCorrelacaoForte] = NULL
  dadosTeste[,indicesCorrelacaoForte] = NULL

  # Estimação dos modelos
  set.seed(semente)
  lda.modelo = train(dadosTreinamento, rotulosTreinamento, method = "lda",
                     trControl = paramsTreinamento)

  set.seed(semente)
  mlp.modelo = train(dadosTreinamento, rotulosTreinamento, method = "nnet",
                     trControl = paramsTreinamento, tuneGrid = grid_mlp,
                     MaxNWts = 20000, maxit = 200, trace = FALSE)

  # - LDA ----
  predicao = factor(predict(lda.modelo, dadosTeste), levels = nomesClasses)
  res = confusionMatrix(predicao, rotulosTeste)
  f1 = res$byClass[,"F1"]
  f1[is.na(f1) & table(rotulosTeste) > 0] = 0

  resultado[nrow(resultado)+1,] = data.frame("Expandida", "LDA", k + 1,
                                             nrow(dadosTreinamento),
                                             nrow(dadosTeste),
                                             res$overall["Accuracy"],
                                             mean(f1, na.rm = TRUE))
  resultado.classes = rbind(resultado.classes,
                            data.frame(Estrategia = "Expandida",
                                       Classificador = "LDA",
                                       LoteTeste = k + 1, Classe = nomesClasses,
                                       F1 = as.numeric(f1)))
  matrizes[[paste0("Expandida_LDA_", k + 1)]] = res$table

  # - MLP ----
  predicao = factor(predict(mlp.modelo, dadosTeste), levels = nomesClasses)
  res = confusionMatrix(predicao, rotulosTeste)
  f1 = res$byClass[,"F1"]
  f1[is.na(f1) & table(rotulosTeste) > 0] = 0

  resultado[nrow(resultado)+1,] = data.frame("Expandida", "MLP", k + 1,
                                             nrow(dadosTreinamento),
                                             nrow(dadosTeste),
                                             res$overall["Accuracy"],
                                             mean(f1, na.rm = TRUE))
  resultado.classes = rbind(resultado.classes,
                            data.frame(Estrategia = "Expandida",
                                       Classificador = "MLP",
                                       LoteTeste = k + 1, Classe = nomesClasses,
                                       F1 = as.numeric(f1)))
  matrizes[[paste0("Expandida_MLP_", k + 1)]] = res$table

  # Distribuição das classes nos conjuntos desta iteração
  distribuicao = rbind(distribuicao,
    data.frame(Estrategia = "Expandida", LoteTeste = k + 1,
               Conjunto = "Treinamento", Classe = nomesClasses,
               Amostras = as.integer(table(rotulosTreinamento))),
    data.frame(Estrategia = "Expandida", LoteTeste = k + 1,
               Conjunto = "Teste", Classe = nomesClasses,
               Amostras = as.integer(table(rotulosTeste))))

  cat("Expandida - treino: lotes 1 a", k, "- teste: lote", k + 1, "\n")
}

toc()

################################################################################
# 4.2. ESTRATÉGIA B - JANELA DESLIZANTE

tic("Janela Deslizante")

for (k in 1:9) {
  # Treinamento apenas com o Lote k e teste no Lote k+1
  dadosTreinamento   = dados[dados$Lote == k, atributos]
  rotulosTreinamento = dados$Classe[dados$Lote == k]
  dadosTeste         = dados[dados$Lote == k + 1, atributos]
  rotulosTeste       = dados$Classe[dados$Lote == k + 1]

  # Normalização estimada apenas no treinamento
  normalizacaoParametros = preProcess(dadosTreinamento,
                                      method = c("zv", "center", "scale"))
  dadosTreinamento = predict(normalizacaoParametros, dadosTreinamento)
  dadosTeste       = predict(normalizacaoParametros, dadosTeste)

  # Seleção de características estimada apenas no treinamento
  indicesCorrelacaoForte = findCorrelation(cor(dadosTreinamento), cutoff = 0.95)
  dadosTreinamento[,indicesCorrelacaoForte] = NULL
  dadosTeste[,indicesCorrelacaoForte] = NULL

  # Estimação dos modelos
  set.seed(semente)
  lda.modelo = train(dadosTreinamento, rotulosTreinamento, method = "lda",
                     trControl = paramsTreinamento)

  set.seed(semente)
  mlp.modelo = train(dadosTreinamento, rotulosTreinamento, method = "nnet",
                     trControl = paramsTreinamento, tuneGrid = grid_mlp,
                     MaxNWts = 20000, maxit = 200, trace = FALSE)

  # - LDA ----
  predicao = factor(predict(lda.modelo, dadosTeste), levels = nomesClasses)
  res = confusionMatrix(predicao, rotulosTeste)
  f1 = res$byClass[,"F1"]
  f1[is.na(f1) & table(rotulosTeste) > 0] = 0

  resultado[nrow(resultado)+1,] = data.frame("Deslizante", "LDA", k + 1,
                                             nrow(dadosTreinamento),
                                             nrow(dadosTeste),
                                             res$overall["Accuracy"],
                                             mean(f1, na.rm = TRUE))
  resultado.classes = rbind(resultado.classes,
                            data.frame(Estrategia = "Deslizante",
                                       Classificador = "LDA",
                                       LoteTeste = k + 1, Classe = nomesClasses,
                                       F1 = as.numeric(f1)))
  matrizes[[paste0("Deslizante_LDA_", k + 1)]] = res$table

  # - MLP ----
  predicao = factor(predict(mlp.modelo, dadosTeste), levels = nomesClasses)
  res = confusionMatrix(predicao, rotulosTeste)
  f1 = res$byClass[,"F1"]
  f1[is.na(f1) & table(rotulosTeste) > 0] = 0

  resultado[nrow(resultado)+1,] = data.frame("Deslizante", "MLP", k + 1,
                                             nrow(dadosTreinamento),
                                             nrow(dadosTeste),
                                             res$overall["Accuracy"],
                                             mean(f1, na.rm = TRUE))
  resultado.classes = rbind(resultado.classes,
                            data.frame(Estrategia = "Deslizante",
                                       Classificador = "MLP",
                                       LoteTeste = k + 1, Classe = nomesClasses,
                                       F1 = as.numeric(f1)))
  matrizes[[paste0("Deslizante_MLP_", k + 1)]] = res$table

  # Distribuição das classes nos conjuntos desta iteração
  distribuicao = rbind(distribuicao,
    data.frame(Estrategia = "Deslizante", LoteTeste = k + 1,
               Conjunto = "Treinamento", Classe = nomesClasses,
               Amostras = as.integer(table(rotulosTreinamento))),
    data.frame(Estrategia = "Deslizante", LoteTeste = k + 1,
               Conjunto = "Teste", Classe = nomesClasses,
               Amostras = as.integer(table(rotulosTeste))))

  cat("Deslizante - treino: lote", k, "- teste: lote", k + 1, "\n")
}

toc()

# Libera objetos que não serão mais usados
rm(dadosTreinamento, rotulosTreinamento, dadosTeste, rotulosTeste,
   normalizacaoParametros, indicesCorrelacaoForte, lda.modelo, mlp.modelo,
   predicao, res, f1, k, lote)

################################################################################
# 5. RESULTADOS                                                                #
################################################################################

resultado$Estrategia = factor(resultado$Estrategia,
                              levels = c("Baseline", "Expandida", "Deslizante"))

# Medidas de avaliação de todas as iterações
resultado

# Desempenho médio por estratégia e classificador
aggregate(cbind(Acuracia, F1Macro) ~ Estrategia + Classificador,
          data = resultado, FUN = mean)

# Salva os resultados para a geração de tabelas do relatório
write.csv(resultado, "resultados/metricas_gerais.csv", row.names = FALSE)
write.csv(resultado.classes, "resultados/metricas_por_classe.csv",
          row.names = FALSE)
write.csv(distribuicao, "resultados/distribuicao_treino_teste.csv",
          row.names = FALSE)
saveRDS(matrizes, "resultados/matrizes_confusao.rds")

################################################################################
# 5.1. DEGRADAÇÃO TEMPORAL DA LINHA DE BASE

ggplot(subset(resultado, Estrategia == "Baseline"),
       aes(x = LoteTeste, y = Acuracia, color = Classificador)) +
  geom_line() + geom_point() + ylim(0, 1) +
  scale_x_continuous(breaks = 2:10) +
  ggtitle("Degradação do modelo treinado apenas no Lote 1")
ggsave("figuras/baseline_degradacao.png", width = 8, height = 4.5)

################################################################################
# 5.2. COMPARAÇÃO ENTRE AS ESTRATÉGIAS

ggplot(resultado, aes(x = LoteTeste, y = Acuracia, color = Estrategia)) +
  geom_line() + geom_point() + ylim(0, 1) +
  facet_wrap(~ Classificador) +
  scale_x_continuous(breaks = 2:10) +
  ggtitle("Acurácia por lote de teste")
ggsave("figuras/comparacao_acuracia.png", width = 10, height = 4.5)

ggplot(resultado, aes(x = LoteTeste, y = F1Macro, color = Estrategia)) +
  geom_line() + geom_point() + ylim(0, 1) +
  facet_wrap(~ Classificador) +
  scale_x_continuous(breaks = 2:10) +
  ggtitle("F1-Score Macro por lote de teste")
ggsave("figuras/comparacao_f1macro.png", width = 10, height = 4.5)

################################################################################
# 5.3. MATRIZES DE CONFUSÃO (Lote 2 x Lote 10)

matrizes[["Expandida_MLP_2"]]
matrizes[["Expandida_MLP_10"]]

setwd("C:/Backup outro PC/Programming/R/Clasificacao-Temporal-E-noses")

rm(list=ls())

lista_lotes <- list()

# 3. Loop para ler os 10 arquivos (assumindo que se chamam b1.dat até b10.dat ou batch_1.dat...)
for (i in 1:10) {
  # Monta o nome do arquivo dinamicamente (ex: batch1.dat, batch2.dat...)
  nome_arquivo <- paste0("./Dataset/batch", i, ".dat")
  
  # Verifica se o arquivo existe antes de tentar ler
  if (file.exists(nome_arquivo)) {
    
    # Lê o arquivo linha a linha como texto puro
    linhas <- readLines(nome_arquivo)
    
    # Processa os pares "índice:valor" removendo o prefixo
    dados_processados <- t(sapply(linhas, function(linha) {
      partes <- unlist(strsplit(linha, "\\s+"))
      classe <- partes[1]
      pares <- partes[-1]
      valores <- as.numeric(sub("^\\d+:", "", pares))
      return(c(as.numeric(classe), valores))
    }))
    
    # Transforma em data.frame
    df_lote <- as.data.frame(dados_processados)
    
    # Padroniza os nomes das colunas: Primeira coluna é a Classe/Target, o resto são as features V1, V2...
    colnames(df_lote) <- c("Classe", paste0("V", 1:(ncol(df_lote)-1)))
    
    # CRUCIAL: Adiciona a coluna de identificação do lote (batch)
    df_lote$Lote <- paste0("Batch_", i)
    
    # Armazena na lista
    lista_lotes[[i]] <- df_lote
    
    cat("Lote", i, "carregado com sucesso. Amostras:", nrow(df_lote), "\n")
  } else {
    warning(paste("Arquivo não encontrado:", nome_arquivo))
  }
}

# 4. Consolida todos os 10 lotes em uma única grande tabela estruturada
dataset_completo <- do.call(rbind, lista_lotes)

# Converte a coluna 'Lote' para fator (ótimo para análises estatísticas e ML)
dataset_completo$Lote <- as.factor(dataset_completo$Lote)

# 5. Validação rápida do resultado
cat("\n--- Resumo do Pipeline Concluído ---\n")
print(dim(dataset_completo)) # Mostra Total de Linhas e Colunas
table(dataset_completo$Lote) # Confere a contagem de amostras por lote

# Cria a tabela cruzada de Lotes x Classes
tabela_distribuicao <- table(dataset_completo$Lote, dataset_completo$Classe)

# Exibe a tabela com as contagens absolutas
print(tabela_distribuicao)

# Se quiser ver em porcentagens (%) por linha (proporção dentro de cada lote):
prop.table(tabela_distribuicao, margin = 1) * 100

library(ggplot2)

# 1. Converte a tabela em data.frame
df_plot <- as.data.frame(tabela_distribuicao)

# 2. Renomeia as colunas garantindo as aspas nas strings
colnames(df_plot) <- c("Lote", "Classe", "Freq")

# 3. Plota o gráfico e força a exibição usando print()
grafico <- ggplot(df_plot, aes(x = Classe, y = Freq, fill = Lote)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  scale_fill_brewer(palette = "Paired") +
  labs(title = "Distribuição de Classes por Lote",
       x = "Classes",
       y = "Quantidade de Amostras") +
  theme(legend.position = "right")

# Exibe o gráfico na aba 'Plots' do RStudio
print(grafico)

library(dplyr)

# 1. Garante que todas as combinações de Lote e Classe existam (preenchendo com 0 se faltar alguma classe)
df_plot_completo <- expand.grid(
  Lote = unique(df_plot$Lote),
  Classe = unique(df_plot$Classe)
) %>%
  left_join(df_plot, by = c("Lote", "Classe")) %>%
  mutate(Freq = ifelse(is.na(Freq), 0, Freq))

# 2. Calcula a porcentagem estritamente dentro de cada lote (a soma de cada lote será exatamente 100%)
df_pizza <- df_plot_completo %>%
  group_by(Lote) %>%
  mutate(
    Total_Lote = sum(Freq),
    Porcentagem = ifelse(Total_Lote > 0, (Freq / Total_Lote) * 100, 0)
  ) %>%
  ungroup()

# 3. Plota o gráfico usando 'Porcentagem' no eixo y para forçar cada lote a fechar 100% perfeitamente
grafico_pizza <- ggplot(df_pizza, aes(x = "", y = Porcentagem, fill = as.factor(Classe))) +
  geom_bar(stat = "identity", width = 1, color = "white", position = "stack") +
  coord_polar("y", start = 0) +               # Transforma em pizza
  facet_wrap(~ Lote) +                        # Um gráfico separado para cada lote (1 a 10)
  scale_fill_brewer(palette = "Set1") +       # Cores sólidas
  theme_void() +                              # Remove fundos desnecessários
  labs(title = "Distribuição Percentual de Classes por Lote",
       fill = "Classe") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
    legend.position = "right",
    strip.text = element_text(face = "bold", size = 11)
  )

# Exibe o gráfico
print(grafico_pizza)
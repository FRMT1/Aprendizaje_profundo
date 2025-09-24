library(keras3); library(dplyr); library(caret)

data <- read.csv("~/Modelado_predictivo/deep_learning_r/Clasificacion_multiclase/creditcard.csv",
                 header = T)
str(data)
data <- data %>% select(-Time) 
str(data)

table(data$Class)      # Diagnóstico del imbalance de clases
NF_to_F <- 284315/492
NF_to_F                # Corrección del imbalance (es muy drástico y la red "temerá")



data <- as.matrix(data)
dimnames(data) <- NULL
data[,1:29] <- normalize(data[,1:29])


set.seed(1234)
id <- sample(2, nrow(data), replace = T, prob = c(0.80,0.20))

trainx <- data[id == 1, 1:29]
testx <- data[id == 2, 1:29]
traintarget <- data[id == 1, 30]
testtarget <- data[id == 2, 30]

trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)


model1 <- keras_model_sequential()
model1 %>%
  layer_dense(units = 50, activation = 'relu', input_shape = c(29)) %>%
  layer_dropout(rate = 0.30) %>%
  layer_dense(units = 30, activation = 'relu') %>%
  layer_dropout(rate = 0.20) %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'sigmoid')
summary(model1)

model1 %>%
  compile(loss = 'binary_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')
history <- model1 %>%
  fit(trainx,
      trainlabels,
      epochs = 50,
      batch_size = 120,
      validation_split = 0.20,
      class_weight = list("0" = 1, "1" = 577)
      )
plot(history)

model1 %>%
  evaluate(testx, testlabels)

pred_proba <- model1 %>%
  predict(testx) %>%
  op_argmax(axis = -1)



pred_classes <- ifelse(pred_proba < 0.05, 1, 0)


confusionMatrix(
  data = factor(pred_classes, levels = c("0", "1")),
  reference = factor(testtarget),
  positive = "1"
)

# El modelo neuronal construido mediante aprendizaje supervisado, NO es capaz
# de detectar anomnalías (fraude). Por el contrario, es muy bueno para predecir
# operaciones no fraudulentas.




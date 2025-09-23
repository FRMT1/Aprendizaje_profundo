library(keras3); library(dplyr); library(ggplot2); library(caret)

data <- read.csv("~/Modelado_predictivo/machine_learning/deep_learning_r/Autoencoders/creditcard.csv",
                 header = T)
str(data)

data <- data %>% select(-Time)
str(data)

set.seed(1234)
muestreo <- sample(2, nrow(data), replace = T, prob = c(0.80,0.20))

pre_train_data <- data[muestreo == 1,]
pre_test_data <- data[muestreo == 2,]
trainx <- pre_train_data %>%
  filter(Class == 0) %>%
  select(-Class)
testx <- pre_test_data %>%
  select(-Class)

minimos <- apply(trainx,2,min)
maximos <- apply(trainx,2,max)

normalize <- function(x){
  scale(x, center = minimos, scale = (maximos-minimos))
}

trainxN <- normalize(as.matrix(trainx))
testxN <- normalize(as.matrix(testx))

input_dim <- ncol(trainxN)
input_layer <- layer_input(shape = input_dim)

encoder <- input_layer %>%
  layer_dense(units = input_dim, activation = 'relu') %>%
  layer_dropout(rate = 0.20) %>%
  layer_dense(units = 15, activation = 'relu')
decoder <- encoder %>%
  layer_dense(units = 15, activation = 'relu') %>%
  layer_dropout(rate = 0.20) %>%
  layer_dense(units = 29, activation = 'sigmoid')

autoencoder <- keras_model(input_layer, decoder)

autoencoder %>%
  compile(loss = 'mse',
          optimizer = 'adam')

model <- autoencoder %>%
  fit(x = trainxN,
      y = trainxN,
      epochs = 25,
      batch_size = 100,
      validation_split = 0.20)
plot(model)

testxN_pred <- autoencoder %>%
  predict(testxN)

mse <- apply((testxN-testxN_pred)**2,1,mean)

results <- data.frame(
  mse = mse,
  Clase = pre_test_data$Class
)
head(results)

umbral <- quantile(mse, 0.95)

ggplot(results, aes(mse, fill = factor(Clase)))+
  geom_histogram(bins = 100, position = 'identity')+
  coord_cartesian(xlim = c(0,0.1))+
  geom_vline(xintercept = umbral, linetype = "dashed", col = "darkblue")+
  labs(
    title = "Distribución del mse (reconstrucción)",
    fill = "clase"
  )+
  theme_minimal()

ypred <- ifelse(mse > umbral,1,0)

# Con caret

MC <- confusionMatrix(
  data = factor(ypred),
  reference = factor(pre_test_data$Class),
  positive = "1"
)
MC

# La sensibilidad es de 90% para detectar anomalías. En este caso, fraude
# Excelente

# El autoencoder es efectivamente capaz de detectar anomalías en los datos (fraude).

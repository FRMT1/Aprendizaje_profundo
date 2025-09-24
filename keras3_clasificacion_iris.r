library(keras3)
library(tensorflow)
library(reticulate)
mydata <- iris
View(mydata)
str(mydata)
#Normalizacion
min_value <- apply(mydata[,1:4],2,min)
max_value <- apply(mydata[,1:4],2,max)
mydata[,1:4] <- scale(mydata[,1:4], center = min_value,
                      scale = max_value - min_value)
mydata[,5] <- as.numeric(mydata[,5]) - 1
#Transformcion de los datos en matriz
mydata <- as.matrix(mydata)
#Particion de los datos
set.seed(1234)
index <- sample(2, nrow(mydata), replace = T, prob = c(0.70,0.30))
training <- mydata[index == 1, 1:4]
test <- mydata[index == 2, 1:4]
trainingtarget <- mydata[index == 1, 5]
testtarget <- mydata[index == 2, 5]
#Transformacion del target
traininglabels <- to_categorical(trainingtarget)
testlabels <- to_categorical(testtarget)
#Arquitectura neuronal
model <- keras_model_sequential()
model %>%
  layer_dense(units = 10, activation = "relu", input_shape = 4) %>%
  layer_dropout(rate = 0.20 )%>%
  layer_dense(units = 10, activation = "relu") %>%
  layer_dropout(rate = 0.30)%>%
  layer_dense(units = 3, activation = "softmax")
summary(model)
#Instrucciones de compilacion
model %>%
  compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = 0.01),
    metrics = "accuracy"
  )
model %>%
  fit(training,
      traininglabels,
      epochs = 300,
      batch_size = 25,
      validation_split = 0.15)
#Evaluacion del modelo
model %>%
  evaluate(test,testlabels)
#Prediccion
pred <- model %>%
  predict(test) %>%
  op_argmax(axis = -1)
pred
pred <- as.array(pred)   # Importantísimo: transforma el tensor de tensorflow a R
pred
# Matriz de confusió
predicted <- as.vector(pred) 
classes <- as.vector(testtarget)

results <- data.frame(Actual = classes, Predicted = predicted)
results

table(results)

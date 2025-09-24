#Reading data
library(keras3)
data <- read.csv("~/Modelado_predictivo/machine_learning/deep_learning_r/Clasificacion_multiclase/CTG.csv",
                 header = T)
str(data)
#Normalization
data <- as.matrix(data)
dimnames(data) <- NULL
data[,1:21] <- normalize(data[,1:21])
data[,22] <- as.numeric(data[,22]) - 1
View(data)
str(data)
#Data partitioning
set.seed(1234)
id <- sample(2, nrow(data), replace = T, prob = c(.7,.3))
training <- data[id == 1, 1:21]
test <- data[id == 2, 1:21]
trainingtarget <- data[id == 1, 22]
testtarget <- data[id == 2, 22]
#One-hot encoding
trainlabels <- to_categorical(trainingtarget)
testlabels <- to_categorical(testtarget)
print(testlabels[1:10,])
#Neural architecture
model <- keras_model_sequential()
model %>%
  layer_dense(units = 8, activation = 'relu', input_shape = c(21)) %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)
#Compiling the model
model %>%
  compile(loss = 'categorical_crossentropy', 
          optimizer = 'adam', 
          metrics = 'accuracy')
#Entrenamiento
model_one <- model %>%
  fit(training,
      trainlabels,
      epochs = 200,
      batch_size = 32,
      validation_split = .20,
      )
plot(model_one)
#Evaluacion del modelo
model %>%
  evaluate(test, testlabels)
colSums(testlabels)
#Matriz de confusion
pred <- model %>%
  predict(test) %>%
  op_argmax(axis = -1)
pred
pred <- as.vector(pred)
classes <- as.vector(testtarget)
table(Predicted = pred, Actual = classes)
#Model optimization. Option 1
model <- keras_model_sequential()
model %>%
  layer_dense(units = 8, activation = 'relu', input_shape = c(21)) %>%
  layer_dense(units = 5, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)
model %>%
  compile(loss = 'categorical_crossentropy', 
          optimizer = 'adam', 
          metrics = 'accuracy')
model_two <- model %>%
  fit(training,
      trainlabels,
      epochs = 200,
      batch_size = 32,
      validation_split = .20,
  )
plot(model_two) 
model %>%
  evaluate(test, testlabels)
pred <- model %>%
  predict(test) %>%
  op_argmax(axis = -1)
pred <- as.vector(pred)
classes <- as.vector(testtarget)
table(Predicted = pred, Actual = classes)
#Model optimization. Option 2
model <- keras_model_sequential()
model %>%
  layer_dense(units = 30, activation = 'relu', input_shape = c(21)) %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)
model %>%
  compile(loss = 'categorical_crossentropy', 
          optimizer = 'adam', 
          metrics = 'accuracy')
model_three <- model %>%
  fit(training,
      trainlabels,
      epochs = 200,
      batch_size = 32,
      validation_split = .20,
  )
plot(model_two) 
model %>%
  evaluate(test, testlabels)
pred <- model %>%
  predict(test) %>%
  op_argmax(axis = -1)
pred <- as.vector(pred)
classes <- as.vector(testtarget)
table(Predicted = pred, Actual = classes)
#Optimization. Option 3
model <- keras_model_sequential()
model %>%
  layer_dense(units = 40, activation = 'relu', input_shape = c(21)) %>%
  layer_dropout(rate = 0.40) %>%
  layer_dense(units = 30, activation = 'relu') %>%
  layer_dropout(rate = 0.30) %>%
  layer_dense(units = 20, activation = 'relu') %>%
  layer_dropout( rate = 0.20) %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)
model %>%
  compile(loss = 'categorical_crossentropy', 
          optimizer = 'adam', 
          metrics = 'accuracy')
model_four <- model %>%
  fit(training,
      trainlabels,
      epochs = 200,
      batch_size = 32,
      validation_split = .20,
  )
plot(model_four) 
model %>%
  evaluate(test, testlabels)
pred <- model %>%
  predict(test) %>%
  op_argmax(axis = -1)
pred <- as.vector(pred)
classes <- as.vector(testtarget)
table(Predicted = pred, Actual = classes)
#Peoblema de clases desbalanceadas
NSP <- to_categorical(data[,22])
colSums(NSP)
N_to_S <- 1655/295
N_to_S
N_to_P <- 1655/176
N_to_P
#Optimization. Option 4
model <- keras_model_sequential()
model %>%
  layer_dense(units = 40, activation = 'relu', input_shape = c(21)) %>%
  layer_dropout(rate = 0.40) %>%
  layer_dense(units = 30, activation = 'relu') %>%
  layer_dropout(rate = 0.30) %>%
  layer_dense(units = 20, activation = 'relu') %>%
  layer_dropout( rate = 0.20) %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)
model %>%
  compile(loss = 'categorical_crossentropy', 
          optimizer = 'adam', 
          metrics = 'accuracy')
model_five <- model %>%
  fit(training,
      trainlabels,
      epochs = 200,
      batch_size = 32,
      validation_split = .20,
      class_weight = list("0" = 1, "1" = 5.61, "2" = 9.40)
  )
plot(model_five) 
model %>%
  evaluate(test, testlabels)
pred <- model %>%
  predict(test) %>%
  op_argmax(axis = -1)
pred <- as.vector(pred)
classes <- as.vector(testtarget)
table(Predicted = pred, Actual = classes)
#Accuracy per class
N <- ((361)/(361+79+20)) * 100
S <- ((73)/(10+73+11)) * 100
P <- ((43)/(43+4+2)) * 100
Results <- data.frame(Normal = N, Suspect = S, Pathological = P)
Results
#Saving, loading and reusing the latest model
save_model(model,"cardiotocography_model.keras")  
new_model <- load_model("cardiotocography_model.keras")
new_model %>%
  evaluate(test, testlabels)
pred <- new_model %>%
  predict(test) %>%
  op_argmax(axis = -1)
pred <- as.vector(pred)
classes <- as.vector(testtarget)
table(Predicted = pred, Actual = classes)

library(h2o)
h2o.init()
occupancy_train <- read.csv("~/Modelado predictivo/deep_learning/Regresion/occupancy_data/datatraining.txt",
                            header = T)
head(occupancy_train)
occupancy_test <- read.csv("~/Modelado predictivo/deep_learning/Regresion/occupancy_data/datatest.txt",
                           header = T)
head(occupancy_test)
#Definiendo inputs y outputs
x = c("Temperature","Humidity","Light","CO2","HumidityRatio")
y = "Occupancy"
#Definiendo la variable dependiente como factor
occupancy_train$Occupancy = as.factor(occupancy_train$Occupancy)
occupancy_test$Occupancy = as.factor(occupancy_test$Occupancy)
#Convirtiendo los data frame en objetos h2o
occupancy_train.h2o <- as.h2o(x=occupancy_train,
                              destination_frame = "occupancy_train.h2o")
occupancy_test.h2o <- as.h2o(x=occupancy_test,
                             destination_frame = "occupancy_test.h2o")
occupancy.deepmodel <- h2o.deeplearning(x = x,
                                        y = y,
                                        training_frame = 
                                            occupancy_train.h2o,
                                        validation_frame = 
                                            occupancy_test.h2o,
                                        hidden = c(3,3),
                                        standardize = T,
                                        activation = "RectifierWithDropout",
                                        adaptive_rate = T,
                                        variable_importances = T,
                                        epochs = 100)
# Obteniendo el desempeño del modelo en el entrenamiento
train_performance <- h2o.performance(occupancy.deepmodel,train = T)
train_performance@metrics$AUC
# Obteniendo la precisión de la red con los datos de validación
xval_performance <- h2o.performance(occupancy.deepmodel,valid = T)
xval_performance@metrics$AUC
# Capacidad de predicción del modelo neuronal
Temperature = 24.1
Humidity = 27.3
Light = 574
CO2 = 780
HumidityRatio = 0.004753
new_data <- data.frame(Temperature,Humidity,Light,CO2,
                       HumidityRatio)
new_data
predictions <- h2o.predict(occupancy.deepmodel,newdata = 
                as.h2o(new_data))
predictions

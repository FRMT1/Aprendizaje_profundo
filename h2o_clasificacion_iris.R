library(h2o)
h2o.init()

data <- iris
data <- as.h2o(iris)

data_split <- h2o.splitFrame(data = data, ratios = c(0.70), seed = 1234)

data_train <- data_split[[1]]
data_test <- data_split[[2]]

x <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width")
y <- "Species"


model <- h2o.deeplearning(x = x,
                          y = y,
                          training_frame = data_train,
                          hidden = 5,
                          standardize = TRUE,
                          activation = "RectifierWithDropout",
                          adaptive_rate = TRUE,
                          epochs = 100,
                          seed = 123456)


h2o.confusionMatrix(model,data_test)
#### OR
perf <- h2o.performance(model,data_test)
h2o.confusionMatrix(perf)

##### En detalle con las Probabilidades
predictions <- h2o.predict(model, newdata = data_test)
predictions <- as.data.frame(predictions)
actual <- as.vector(data_test[,5])

library(dplyr)
results <- predictions %>%
  mutate(actual = actual)
View(results)




h2o.shutdown()
Y


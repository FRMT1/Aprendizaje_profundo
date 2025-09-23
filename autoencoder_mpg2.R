library(keras3)
library(dplyr)
library(ggplot2)

View(mtcars)

mtcars$id <- ifelse(mtcars$mpg < 15, "Anómalo", "Normal")
View(mtcars)

trainx <- filter(mtcars, id == "Normal") %>%
  select(-id)
testx <- mtcars %>%
  select(-id)
View(trainx)
View(testx)

minimos <- apply(trainx,2,min)
maximos <- apply(trainx,2,max)

normalize <- function(x){
  scale(x, center = minimos, scale = (maximos - minimos))
}

trainxN <- normalize(as.matrix(trainx))
testxN <- normalize(as.matrix(testx))

input_dim <- ncol(trainxN)
input_layer <- layer_input(shape = input_dim)

encoder <- input_layer |>
  layer_dense(units = 14, activation = 'relu') |>
  layer_dropout(rate = 0.30) |>
  layer_dense(units = 7, activation = 'relu')
decoder <- encoder %>%
  layer_dense(units = 7, activation = 'relu') |>
  layer_dropout(rate = 0.20) |>
  layer_dense(units = input_dim, activation = 'linear')
autoencoder <- keras_model(input_layer, decoder)

autoencoder |>
  compile(loss = 'mse',
          optimizer = 'adam')

model <- autoencoder |>
  fit(x = trainxN,
      y = trainxN,
      epochs = 150,
      batch_size = 5,
      validation_split = 0.20)
plot(model)

test_predictions <- autoencoder |>
  predict(testxN)
reconstruction_error <- apply((testxN-test_predictions)**2,1,mean)

results <- data.frame(
  Nombre = rownames(mtcars),
  Error = reconstruction_error,
  Real = mtcars$id
)
results

ggplot(results, aes(Nombre,Error,fill = Real))+
  geom_bar(stat = 'identity')+
  coord_flip()+
  geom_hline(yintercept = quantile(reconstruction_error,0.95),
             linetype = 'dashed', col = 'darkblue')+
  labs(
    title = 'Detección de autos anómalos',
    x = 'Autos',
    y = 'Error'
  )
  
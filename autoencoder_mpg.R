# Instalar e importar paquetes necesarios
library(keras3)
library(dplyr)
library(ggplot2)

# 1. Cargar y preparar los datos
data("mtcars")
str(mtcars)
View(mtcars)

# Vamos a detectar autos con alto consumo (mpg muy bajo) como anomalías
# Supongamos que los autos con mpg < 15 son "anómalos"
mtcars$label <- ifelse(mtcars$mpg < 15, "anómalo", "normal")
View(mtcars)

# Usamos solo autos normales para entrenar
train_data <- mtcars %>% filter(label == "normal") %>% select(-label)
test_data <- mtcars %>% select(-label)
View(train_data)
View(test_data)

# Normalizar (es fundamental para redes)
means <- apply(train_data, 2, mean)
sds <- apply(train_data, 2, sd)

normalize <- function(x) {
  scale(x, center = means, scale = sds)
}

x_train <- normalize(as.matrix(train_data))
x_test <- normalize(as.matrix(test_data))

# 2. Construir el autoencoder
input_dim <- ncol(x_train)

input_layer <- layer_input(shape = input_dim)

encoder <- input_layer |>
  layer_dense(units = 8, activation = "relu") |>
  layer_dense(units = 3, activation = "relu")

decoder <- encoder |>
  layer_dense(units = 8, activation = "relu") |>
  layer_dense(units = input_dim, activation = "linear")

autoencoder <- keras_model(input_layer, decoder)

autoencoder |>
  compile(optimizer = "adam", loss = "mse")

# 3. Entrenar el autoencoder solo con datos normales
autoencoder |>
  fit(
    x = x_train,
    y = x_train,
    epochs = 100,
    batch_size = 4,
    validation_split = 0.2,
    verbose = 0
  )

# 4. Calcular error de reconstrucción en todos los datos (entrenamiento + test)
x_test_pred <- autoencoder |> predict(x_test)
reconstruction_error <- apply((x_test - x_test_pred)^2, 1, mean)

# 5. Crear un dataframe con los errores y etiquetas reales
results <- data.frame(
  nombre = rownames(mtcars),
  error = reconstruction_error,
  real = mtcars$label
)
results
# 6. Visualizar
ggplot(results, aes(x = nombre, y = error, fill = real)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  geom_hline(yintercept = quantile(reconstruction_error, 0.90), linetype = "dashed", color = "red") +
  labs(title = "Error de reconstrucción por auto", y = "Error", x = "Auto") +
  theme_minimal()

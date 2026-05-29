library(torch); library(brulee); library(tidymodels); library(synthpop)


# Generación de datos sintéticos ------------------------------------------



set.seed(1234)

iris_synthetic_obj <- syn(data = iris, k = 10000)
iris_synthetic <- iris_synthetic_obj$syn
head(iris_synthetic)
tail(iris_synthetic)


# Datos de entrenamiento y datos de prueba --------------------------------



iris_train <- iris_synthetic
iris_test <- iris 


# Normalización de datos de entrenamiento ---------------------------------



iris_recipe <- recipe(Species ~ ., data = iris_train) |> 
  step_normalize(all_numeric_predictors())


# Definición de arquitectura neuronal -------------------------------------



set.seed(123)

nn_model <- mlp(
  hidden_units = c(16, 8),
  epochs = 100,
  dropout = 0.20,
  activation = 'relu'
) |> 
  set_engine('brulee') |> 
  set_mode('classification')


# Juntar datos normalizados y arquitectura con tidymodels -----------------



iris_workflow <- workflow() |> 
  add_recipe(iris_recipe) |> 
  add_model(nn_model)


# Entrenamiento del modelo ------------------------------------------------



iris_fit <- fit(iris_workflow, data = iris_train) 


# Predicción de clases y probabilidades -----------------------------------




preds_class <- predict(iris_fit, iris_test) # Tidymodels (workflow), normaliza iris_test
preds_proba <- predict(iris_fit, iris_test, type = 'prob')

results <- iris_test |> 
  select(Species) |> 
  bind_cols(preds_class, preds_proba)
head(results, 10)


metricas <- metric_set(accuracy, kap)

results |> 
  metricas(
    truth = Species,
    estimate = ".pred_class"
  )


# Matriz de confusión -----------------------------------------------------



cm <- conf_mat(results, truth = Species, estimate = .pred_class)
summary(cm)
cm


# Guardar el modelo entrenado ---------------------------------------------



library(bundle)

iris_fit_bundle <- bundle(iris_fit)
iris_fit_bundle_saved <- saveRDS(iris_fit_bundle, file = "neural_model.rds")



# Predicción con nuevos datos ---------------------------------------------



new_data <- data.frame(
  Sepal.Length = 6.7,
  Sepal.Width = 3.3,
  Petal.Length = 5.7,
  Petal.Width = 2.5
)
new_data

model_bundle <- readRDS("neural_model.rds")
model <- unbundle(model_bundle)

predicted_class <- predict(model, new_data)
predicted_class

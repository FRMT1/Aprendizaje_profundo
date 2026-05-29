library(tidymodels); library(torch); library(brulee)

data(iris)
set.seed(42)

splits <- initial_split(data = iris, prop = 0.75, strata = Species)
iris_train <- training(splits)
iris_test <- testing(splits)


dim(iris_train)
dim(iris_test)


iris_recipe <- recipe(Species ~., data = iris_train) |> 
  step_normalize(all_numeric_predictors())

summary(iris_recipe)

nn_model <- mlp(hidden_units = c(7, 5),
                epochs = 100,
                learn_rate = 0.01,
                dropout = 0.10,
                activation = 'relu') |> 
  set_engine('brulee') |> 
  set_mode('classification')

 

iris_workflow <- workflow() |> 
  add_recipe(iris_recipe) |> 
  add_model(nn_model)


iris_fit <- fit(iris_workflow, data = iris_train)
iris_fit


# Predicción sobre conjunto de prueba -------------------------------------

#Clase predicha

preds_class <- predict(iris_fit, iris_test)
print(preds_class, n = 39)

#Probabilidad predicha por clase

preds_proba <- predict(iris_fit, iris_test, type = 'prob')
print(preds_proba)

#Uniendo con los datos observados

results <- iris_test |> 
  select(Species) |> 
  bind_cols(preds_class, preds_proba)

head(results,5)


# Métricas de evaluación --------------------------------------------------

metricas <- metric_set(accuracy)

results |> 
  metricas(
    truth = Species,
    estimate = ".pred_class",
    starts_with(".pred_"),
    estimator = "macro_weighted"
  )


# Matriz de confusión -----------------------------------------------------

conf_mat(results, truth = Species, estimate = .pred_class)

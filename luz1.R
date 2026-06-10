library(caret)
library(torch)
library(luz)

set.seed(123)
id <- createDataPartition(iris$Species, p = 0.75, list = F)
train <- iris[id,]
test <- iris[-id,]

scaled_data <- preProcess(train[,-5], method = c("center", "scale"))
train_scaled <- predict(scaled_data, train)
test_scaled <- predict(scaled_data, test)

iris_dataset <- dataset(
  name = "IrisDataset",
  initialize = function(df){
    self$x = torch_tensor(as.matrix(df[,1:4]), dtype = torch_float())
    self$y = torch_tensor(as.integer(df$Species), dtype = torch_long())
  },
  .getitem = function(i) list(x = self$x[i,], y = self$y[i]),
  .length = function() nrow(self$x) 
)

train_ds <- iris_dataset(train_scaled)
test_ds <- iris_dataset(test_scaled)

train_dl <- dataloader(train_ds, batch_size = 32, shuffle = T)
test_dl <- dataloader(test_ds, batch_size = 32, shuffle = F)

model <- nn_module(
  "NeuralNet",
  initialize = function(){
    self$fc1   <- nn_linear(4, 7)
    self$drop1 <- nn_dropout(p = 0.10)
    self$fc2   <- nn_linear(7, 5)
    self$drop2 <- nn_dropout(p = 0.10)
    self$fc3   <- nn_linear(5, 3)
  },
  forward = function(x){
    x |> 
      self$fc1() |> nnf_relu() |> self$drop1() |> 
      self$fc2() |> nnf_relu() |> self$drop2() |> 
      self$fc3()
  }
)

model_fit <- model |>
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(luz_metric_accuracy())
  ) |> 
  set_hparams() |> 
  set_opt_hparams(lr = 0.01) |> 
  fit(train_dl, epochs = 150, valid_data = test_dl)

print(model_fit)  


levels_iris <- levels(iris$Species)

pred_tensor <- predict(model_fit, test_dl)
pred_prob_tensor <- nnf_softmax(logistest, dim = 2)

clases <- as.integer(torch_argmax(pred_prob_tensor, dim = 2))

pred_class <- factor(levels_iris[clases], levels = levels_iris)


pred_df <- as.data.frame(as.matrix(pred_prob_tensor))
colnames(pred_df) <- paste0(".pred_", levels_iris)

results <- data.frame(
  Species = test$Species,
  pred_class = pred_class
)

print(confusionMatrix(results$Species, results$pred_class))

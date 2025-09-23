library(keras3)

mnist <- dataset_fashion_mnist()
str(mnist)

# Dado que se usará una red neuronal autocodificadora, no se utilizaran las etiquetas
# de los datos de netrenamiento y evaluacion. Es un aprendizaje NO SUPERVISADO

trainx <- mnist$train$x
testx <- mnist$test$x

# Imagenes

par(mfrow = c(8,8), mar = rep(0,4))
for(i in 1:64) plot(as.raster(trainx[i,,], max = 255))

# Formatear las imagenes

trainx <- array_reshape(trainx, c(nrow(trainx), 28, 28, 1))
testx <- array_reshape(testx, c(nrow(testx), 28, 28, 1))
trainx <- trainx/255  #Normalizacion
testx <- testx/255

# Red de autocodificacion

input_layer <- layer_input(shape = c(28,28,1))

encoder <- input_layer %>%    # Codificador
  layer_conv_2d(filters = 8,
                kernel_size = c(3,3),
                activation = 'relu',
                padding = 'same') %>%
  layer_max_pooling_2d(pool_size = c(2,2),
                       padding = 'same') %>%
  layer_conv_2d(filters = 4,
                kernel_size = c(3,3),
                activation = 'relu',
                padding = 'same') %>%
  layer_max_pooling_2d(pool_size = c(2,2),
                       padding = 'same')
summary(encoder)

decoder <- encoder %>%
  layer_conv_2d(filter = 4,
                kernel_size = c(3,3),
                activation = 'relu',
                padding = 'same') %>%
  layer_upsampling_2d(c(2,2)) %>%
  layer_conv_2d(filters = 8,
                kernel_size = c(3,3),
                activation = 'relu',
                padding = 'same') %>%
  layer_upsampling_2d(c(2,2)) %>%
  layer_conv_2d(filters = 1,
                kernel_size = c(3,3),
                activation = 'sigmoid',
                padding = 'same')
summary(decoder)

ae_model <- keras_model(inputs = input_layer, outputs = decoder)
summary(ae_model)


# Compile model

ae_model %>% compile(loss = 'mean_squared_error',
                     optimizer = 'adam')

# Fit model

model_one <- ae_model %>%
  fit(trainx,trainx,   # Dos veces porque el input es el output esperado
      epochs = 20,
      shuffle = TRUE,
      batch_size = 32,
      validation_data = list(testx,testx))

plot(model_one)


# Reconstrucción de las imagenes

reconstruct <- ae_model %>% keras3::predict_on_batch(x = trainx)

par(mfrow = c(2,5), mar = rep(0,4))
for(i in 1:5) plot(as.raster(trainx[i,,,]))
for(i in 1:5) plot(as.raster(reconstruct[i,,,]))
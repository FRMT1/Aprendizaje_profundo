library(torch)


# Diferentes tipos de tensores --------------------------------------------

t1 <- torch_tensor(array(1:24, dim = c(4, 3, 2)))
t1

t2 <- torch_randn(3, 3)
t2

t3 <- torch_rand(3, 3)
t3

t4 <- torch_zeros(3, 3)
t4

t5 <- torch_ones(3, 3)
t5

t6 <- torch_eye(3, 3)
t6

t7 <- torch_diag(c(1, 2, 3))
t7

t8 <- torch_tensor(c(1, NA, 5))
t8



# Tensores a partir de datasets -------------------------------------------


# Ejemplo 1 ---------------------------------------------------------------



library(dplyr)
glimpse(Orange)

torch_tensor(Orange)   #Da error porque hay un factor (Tree) 

orange <- Orange |> 
  mutate(Tree = as.numeric(Tree)) |>   # Todas las variables deben ser numéricas
  as.matrix()

t9 <- torch_tensor(orange)
print(t9, n = 10)


# Ejemplo 2 ---------------------------------------------------------------

glimpse(iris)

iris <- iris |> 
  mutate(Species = as.numeric(Species)) |> 
  as.matrix()

t10 <- torch_tensor(iris)
print(t10, n = 10)



# Operaciones con tensores ------------------------------------------------

t <- torch_outer(torch_tensor(1:3), torch_tensor(1:6))
t
t$sum()
t$sum(dim = 1)
t$sum(dim = 2)

# Example with a time series data

#dim 1: individuals (4 individuals)
#dim 2: time (3 times)
#dim 3: features (2 features)

ts <- torch_randn(4, 3, 2)
ts

#Si quiero calcular la media de los features de manera independiente,
#debo colapsar la dimensión 1 y 2

ts$mean(dim = c(1, 2))

#Si quiero la media de los features, pero por persona, debo colapsar la dimensión 2

ts$mean(dim = 2)


# Reshaping tensors -------------------------------------------------------

t1 <- torch_zeros(24)  #Aquí es un vector
print(t1, n = 3)

t2 <- t1$view(c(2, 12))#Aquí es una matriz
t2

#Aun cuando son dos objetos numéricos distintos, torch los guarda en el mismo sitio

t1$storage()$data_ptr()
t2$storage()$data_ptr()


# Diferenciación automática -----------------------------------------------

# 0.2*x1^2 + 0.2*x2^2 - 5

x1 <- torch_tensor(1, requires_grad = TRUE)
x2 <- torch_tensor(1, requires_grad = TRUE)

x3 <- x1$square()
x5 <- x3*0.2

x4 <- x2$square()
x6 <- x4*0.2

x7 <- x5+x6-5
x7

x7$backward()

x1$grad   #Derivada parcial con respecto a x1
x2$grad   #DErivada parciual con respecto a x2


# Visualizando la regla de la cadena --------------------------------------


x1 <- torch_tensor(1, requires_grad = TRUE)
x2 <- torch_tensor(1, requires_grad = TRUE)

x3 <- x1$square()
x3$retain_grad()

x5 <- x3*0.2
x5$retain_grad()

x4 <- x2$square()
x4$retain_grad()

x6 <- x4*0.2
x6$retain_grad()

x7 <- x5+x6-5
x7

x7$backward()

x3$grad
x4$grad
x5$grad
x6$grad


# Cómo torch automatiza esos cálculos -------------------------------------

#En el proceso de forward, torch toma nota de lo que debe calcular y lo anota en grad_fn

x3$grad_fn
x4$grad_fn
x5$grad_fn
x6$grad_fn


# Minimización de función de pérdida con autograd -------------------------



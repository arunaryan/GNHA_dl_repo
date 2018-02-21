## Please refer to the blog post for original problem and explanations
## https://tensorflow.rstudio.com/blog/keras-fraud-autoencoder.html

library(readr)
df <- read_csv("~\\Projects\\GNHA_Meetup\\creditcard.csv", col_types = list(Time = col_number()))

library(tidyr)
library(dplyr)
library(ggplot2)
library(ggridges)
df %>%
  gather(variable, value, -Class) %>%
  ggplot(aes(y = as.factor(variable), 
             fill = as.factor(Class), 
             x = percent_rank(value))) +
  geom_density_ridges()

df_train <- df %>% filter(row_number(Time) <= 200000) %>% select(-Time)
df_test <- df %>% filter(row_number(Time) > 200000) %>% select(-Time)

library(purrr)

#' Gets descriptive statistics for every variable in the dataset.
get_desc <- function(x) {
  map(x, ~list(
    min = min(.x),
    max = max(.x),
    mean = mean(.x),
    sd = sd(.x)
  ))
} 

#' Given a dataset and normalization constants it will create a min-max normalized
#' version of the dataset.
normalization_minmax <- function(x, desc) {
  map2_dfc(x, desc, ~(.x - .y$min)/(.y$max - .y$min))
}

desc <- df_train %>% 
  select(-Class) %>% 
  get_desc()

x_train <- df_train %>%
  select(-Class) %>%
  normalization_minmax(desc) %>%
  as.matrix()

x_test <- df_test %>%
  select(-Class) %>%
  normalization_minmax(desc) %>%
  as.matrix()

y_train <- df_train$Class
y_test <- df_test$Class

library(keras)
library(tensorflow)
model <- keras_model_sequential()
model %>%
  layer_dense(units = 15, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 10, activation = "tanh") %>%
  layer_dense(units = 15, activation = "tanh") %>%
  layer_dense(units = ncol(x_train))

summary(model)

model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam"
)

checkpoint <- callback_model_checkpoint(
  filepath = "model.hdf5", 
  save_best_only = TRUE, 
  period = 1,
  verbose = 1
)

early_stopping <- callback_early_stopping(patience = 5)

model %>% fit(
  x = x_train[y_train == 0,], 
  y = x_train[y_train == 0,], 
  epochs = 100, 
  batch_size = 32,
  validation_data = list(x_test[y_test == 0,], x_test[y_test == 0,]), 
  callbacks = list(checkpoint, early_stopping)
)

loss <- evaluate(model, x = x_test[y_test == 0,], y = x_test[y_test == 0,])
loss


##Tuning the model using cloudml

FLAGS <- flags(
        flag_string("normalization", "minmax", "One of minmax, zscore"),
        flag_string("activation", "relu", "One of relu, selu, tanh, sigmoid"),
        flag_numeric("learning_rate", 0.001, "Optimizer Learning Rate"),
        flag_integer("hidden_size", 15, "The hidden layer size")
)

model %>% compile(
        optimizer = optimizer_adam(lr = FLAGS$learning_rate), 
        loss = 'mean_squared_error'
)

library(cloudml)
cloudml_train("train.R", config = "tuning.yml")

job_collect()

latest_run()$run_dir

ls_runs(order = metric_val_loss, decreasing = FALSE)

model <- load_model_hdf5("runs/cloudml_2018_01_23_221244595-03/model.hdf5", compile = FALSE)

pred_train <- predict(model, x_train)
mse_train <- apply((x_train - pred_train)^2, 1, sum)

pred_test <- predict(model, x_test)
mse_test <- apply((x_test - pred_test)^2, 1, sum)

library(Metrics)
auc(y_train, mse_train)
auc(y_test, mse_test)

possible_k <- seq(0, 0.5, length.out = 100)
precision <- sapply(possible_k, function(k) {
        predicted_class <- as.numeric(mse_test > k)
        sum(predicted_class == 1 & y_test == 1)/sum(predicted_class)
})

qplot(possible_k, precision, geom = "line") + labs(x = "Threshold", y = "Precision")

recall <- sapply(possible_k, function(k) {
        predicted_class <- as.numeric(mse_test > k)
        sum(predicted_class == 1 & y_test == 1)/sum(y_test)
})
qplot(possible_k, recall, geom = "line") + labs(x = "Threshold", y = "Recall")

cost_per_verification <- 1

lost_money <- sapply(possible_k, function(k) {
        predicted_class <- as.numeric(mse_test > k)
        sum(cost_per_verification * predicted_class + (predicted_class == 0) * y_test * df_test$Amount) 
})

qplot(possible_k, lost_money, geom = "line") + labs(x = "Threshold", y = "Lost Money")

possible_k[which.min(lost_money)]
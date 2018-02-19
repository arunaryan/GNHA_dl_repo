# Tidyverse (readr, ggplot2, etc.)
install.packages("tidyverse")

# Packages for sequence logos and peptides
devtools::install_github("omarwagih/ggseqlogo")
devtools::install_github("leonjessen/PepTools")

library(keras)
library(tidyverse)
library(PepTools)

pep_file <- get_file("ran_peps_netMHCpan40_predicted_A0201_reduced_cleaned_balanced.tsv", 
                     origin = "https://git.io/vb3Xa") 
pep_dat <- read_tsv(file = pep_file)
pep_dat %>% head(5)

pep_dat %>% group_by(label_chr, data_type) %>% summarise(n = n())
pep_dat %>% filter(label_chr=='SB') %>% pull(peptide) %>% ggseqlogo()
pep_dat %>% filter(label_chr=='SB') %>% head(1) %>% pull(peptide) %>% pep_plot_images
str(pep_encode(c("LLTDAQRIV", "LLTDAQRIV")))

x_train <- pep_dat %>% filter(data_type == 'train') %>% pull(peptide)   %>% pep_encode
y_train <- pep_dat %>% filter(data_type == 'train') %>% pull(label_num) %>% array
x_test  <- pep_dat %>% filter(data_type == 'test')  %>% pull(peptide)   %>% pep_encode
y_test  <- pep_dat %>% filter(data_type == 'test')  %>% pull(label_num) %>% array

x_train <- array_reshape(x_train, c(nrow(x_train), 9 * 20))
x_test  <- array_reshape(x_test,  c(nrow(x_test), 9 * 20))

y_train <- to_categorical(y_train, num_classes = 3)
y_test  <- to_categorical(y_test,  num_classes = 3)

model <- keras_model_sequential() %>% 
        layer_dense(units  = 180, activation = 'relu', input_shape = 180) %>% 
        layer_dropout(rate = 0.4) %>% 
        layer_dense(units  = 90, activation  = 'relu') %>%
        layer_dropout(rate = 0.3) %>%
        layer_dense(units  = 3, activation   = 'softmax')

summary(model)

model %>% compile(
        loss      = 'categorical_crossentropy',
        optimizer = optimizer_rmsprop(),
        metrics   = c('accuracy')
)

history = model %>% fit(
        x_train, y_train, 
        epochs = 150, 
        batch_size = 50, 
        validation_split = 0.2
)

plot(history)

perf = model %>% evaluate(x_test, y_test)
perf

acc     = perf$acc %>% round(3)*100
y_pred  = model %>% predict_classes(x_test)
y_real  = y_test %>% apply(1,function(x){ return( which(x==1) - 1) })
results = tibble(y_real = y_real %>% factor, y_pred = y_pred %>% factor,
                 Correct = ifelse(y_real == y_pred,"yes","no") %>% factor)
title = 'Performance on 10% unseen data - CNN Neural Network'
xlab  = 'Measured (Real class, as predicted by netMHCpan-4.0)'
ylab  = 'Predicted (Class assigned by Keras/TensorFlow deep CNN)'
results %>%
        ggplot(aes(x = y_pred, y = y_real, colour = Correct)) +
        geom_point() +
        ggtitle(label = title, subtitle = paste0("Accuracy = ", acc,"%")) +
        xlab(xlab) +
        ylab(ylab) +
        scale_color_manual(labels = c('No', 'Yes'),
                           values = c('tomato','cornflowerblue')) +
        geom_jitter() +
        theme_bw()

model <- keras_model_sequential() %>%
        layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                      input_shape = c(9, 20, 1)) %>%
        layer_dropout(rate = 0.25) %>% 
        layer_flatten() %>% 
        layer_dense(units  = 180, activation = 'relu') %>% 
        layer_dropout(rate = 0.4) %>% 
        layer_dense(units  = 90, activation  = 'relu') %>%
        layer_dropout(rate = 0.3) %>%
        layer_dense(units  = 3, activation   = 'softmax')

model %>% compile(
        loss      = 'categorical_crossentropy',
        optimizer = optimizer_rmsprop(),
        metrics   = c('accuracy')
)

history = model %>% fit(
        x_train, y_train, 
        epochs = 150, 
        batch_size = 50, 
        validation_split = 0.2
)

# Setup training data
target  <- 'train'
x_train <- pep_dat %>% filter(data_type==target) %>% pull(peptide) %>%
        pep_encode_mat %>% select(-peptide)
y_train <- pep_dat %>% filter(data_type==target) %>% pull(label_num) %>% factor

# Setup test data
target <- 'test'
x_test <- pep_dat %>% filter(data_type==target) %>% pull(peptide) %>%
        pep_encode_mat %>% select(-peptide)
y_test <- pep_dat %>% filter(data_type==target) %>% pull(label_num) %>% factor

rf_classifier <- randomForest(x = x_train, y = y_train, ntree = 100)

y_pred    <- predict(rf_classifier, x_test)
n_correct <- table(observed = y_test, predicted = y_pred) %>% diag %>% sum
acc       <- (n_correct / length(y_test)) %>% round(3) * 100
results   <- tibble(y_real  = y_test,
                    y_pred  = y_pred,
                    Correct = ifelse(y_real == y_pred,"yes","no") %>% factor)


title = "Performance on 10% unseen data - Random Forest"
xlab  = "Measured (Real class, as predicted by netMHCpan-4.0)"
ylab  = "Predicted (Class assigned by random forest)"
f_out = "plots/03_rf_01_results_3_by_3_confusion_matrix.png"
results %>%
        ggplot(aes(x = y_pred, y = y_real, colour = Correct)) +
        geom_point() +
        xlab(xlab) +
        ylab(ylab) +
        ggtitle(label = title, subtitle = paste0("Accuracy = ", acc,"%")) +
        scale_color_manual(labels = c('No', 'Yes'),
                           values = c('tomato','cornflowerblue')) +
        geom_jitter() +
        theme_bw()

download.file("https://snap.stanford.edu/data/finefoods.txt.gz", "finefoods.txt.gz")
library(readr)
library(stringr)

reviews <- read_lines("finefoods.txt.gz") 
reviews <- reviews[str_sub(reviews, 1, 12) == "review/text:"]
reviews <- str_sub(reviews, start = 14)
reviews <- iconv(reviews, to = "UTF-8")

library(keras)
tokenizer <- text_tokenizer(num_words = 20000)
tokenizer %>% fit_text_tokenizer(reviews)

library(reticulate)
library(purrr)
skipgrams_generator <- function(text, tokenizer, window_size, negative_samples) {
        gen <- texts_to_sequences_generator(tokenizer, sample(text))
        function() {
                skip <- generator_next(gen) %>%
                        skipgrams(
                                vocabulary_size = tokenizer$num_words, 
                                window_size = window_size, 
                                negative_samples = 1
                        )
                x <- transpose(skip$couples) %>% map(. %>% unlist %>% as.matrix(ncol = 1))
                y <- skip$labels %>% as.matrix(ncol = 1)
                list(x, y)
        }
}

embedding_size <- 128  # Dimension of the embedding vector.
skip_window <- 5       # How many words to consider left and right.
num_sampled <- 1       # Number of negative examples to sample for each word.

input_target <- layer_input(shape = 1)
input_context <- layer_input(shape = 1)

embedding <- layer_embedding(
        input_dim = tokenizer$num_words + 1, 
        output_dim = embedding_size, 
        input_length = 1, 
        name = "embedding"
)

target_vector <- input_target %>% 
        embedding() %>% 
        layer_flatten()

context_vector <- input_context %>%
        embedding() %>%
        layer_flatten()

dot_product <- layer_dot(list(target_vector, context_vector), axes = 1)
output <- layer_dense(dot_product, units = 1, activation = "sigmoid")

model <- keras_model(list(input_target, input_context), output)
model %>% compile(loss = "binary_crossentropy", optimizer = "adam")

summary(model)

model %>%
        fit_generator(
                skipgrams_generator(reviews, tokenizer, skip_window, negative_samples), 
                steps_per_epoch = 100000, epochs = 5
        )

library(dplyr)

embedding_matrix <- get_weights(model)[[1]]

words <- data_frame(
        word = names(tokenizer$word_index), 
        id = as.integer(unlist(tokenizer$word_index))
)

words <- words %>%
        filter(id <= tokenizer$num_words) %>%
        arrange(id)

row.names(embedding_matrix) <- c("UNK", words$word)

library(text2vec)

find_similar_words <- function(word, embedding_matrix, n = 5) {
        similarities <- embedding_matrix[word, , drop = FALSE] %>%
                sim2(embedding_matrix, y = ., method = "cosine")
        
        similarities[,1] %>% sort(decreasing = TRUE) %>% head(n)
}

find_similar_words("2", embedding_matrix)

find_similar_words("little", embedding_matrix)

find_similar_words("delicious", embedding_matrix)

find_similar_words("cats", embedding_matrix)

library(Rtsne)
library(ggplot2)
library(plotly)

tsne <- Rtsne(embedding_matrix[2:500,], perplexity = 50, pca = FALSE)

tsne_plot <- tsne$Y %>%
        as.data.frame() %>%
        mutate(word = row.names(embedding_matrix)[2:500]) %>%
        ggplot(aes(x = V1, y = V2, label = word)) + 
        geom_text(size = 3)
tsne_plot
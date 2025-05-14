###############################################################################
# Fake-News Detection :  Naïve Bayes vs. Artificial Neural Network (ANN)
# Author : Divya Gunjan         Date : 2024-11
#
# Reproduces my assignment results:
#   • Naïve Bayes  ≈ 91.1 % accuracy   (precision 92.7 %, recall 90 %)
#   • ANN (2-layer) ≈ 63.9 % accuracy  (precision 56.8 %, recall 99.9 %) :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
#
# Folder structure
# ├─ data/        Fake.csv , True.csv   (UCI “Fake and real news” corpus)
# ├─ scripts/     this file
# └─ …
###############################################################################

## 0 ── Setup ------------------------------------------------------------------
library(tidyverse)
library(tm)          # text mining
library(SnowballC)
library(e1071)       # naiveBayes()
library(gmodels)     # CrossTable()
library(keras3)      # ANN
library(tensorflow)  # auto-installs TF on first use
tf$random$set_seed(1)

## 1 ── Load data --------------------------------------------------------------
fake  <- read.csv(here::here("data", "Fake.csv"))
true  <- read.csv(here::here("data", "True.csv"))
fake$label <- "fake";  true$label <- "true"
news <- bind_rows(fake, true) |>
        mutate(text_title = paste(title, text))

## 2 ── Initial split (80 % train+val, 20 % test) ------------------------------
set.seed(1)
idx   <- sample(nrow(news), 0.8 * nrow(news))
train_val <- news[idx, ];  test <- news[-idx, ]

## 3 ── ✱ Naïve Bayes pipeline  ===============================================-

### 3.1  Create corpus & DocumentTermMatrix (unigrams) -------------------------
build_corpus <- function(char_vec) {
  VCorpus(VectorSource(char_vec)) |>
    tm_map(content_transformer(tolower)) |>
    tm_map(removePunctuation) |>
    tm_map(removeNumbers) |>
    tm_map(removeWords, stopwords("english")) |>
    tm_map(stripWhitespace) |>
    tm_map(stemDocument)
}

corp_tv <- build_corpus(train_val$text_title)
dtm_tv  <- DocumentTermMatrix(corp_tv)
# keep tokens occurring ≥100×
terms   <- findFreqTerms(dtm_tv, 100)
dtm_tv  <- dtm_tv[, terms]
dtm_ts  <- DocumentTermMatrix(build_corpus(test$text_title), list(dictionary = terms))

### 3.2  Binarise counts -------------------------------------------------------
binarise <- function(dtm) { m <- as.matrix(dtm); m[m > 0] <- 1; m }
train_x <- binarise(dtm_tv)
test_x  <- binarise(dtm_ts)
train_y <- factor(train_val$label)
test_y  <- factor(test$label)

### 3.3  Train & evaluate NB ---------------------------------------------------
nb_mod <- naiveBayes(train_x, train_y, laplace = 1)
nb_pred <- predict(nb_mod, test_x)
nb_acc  <- mean(nb_pred == test_y)
cat(glue::glue("Naïve Bayes accuracy : {round(nb_acc,4)}\n"))

## 4 ── ✱ ANN pipeline  =======================================================-

### 4.1 Vectorise text with tf-idf (bigrams, top 5 000) ------------------------
vec <- layer_text_vectorization(output_mode = "tf_idf",
                                ngrams = 2, max_tokens = 5000)
vec$adapt(train_val$text_title)
train_mat <- as.matrix(vec(train_val$text_title))
test_mat  <- as.matrix(vec(test$text_title))
train_labels <- ifelse(train_val$label == "true", 0, 1)
test_labels  <- ifelse(test$label  == "true", 0, 1)

### 4.2 Define 2-hidden-layer model -------------------------------------------
build_ann <- function(input_shape,
                      units1 = 256, units2 = 32,
                      lr = 1e-3) {
  keras_model_sequential() |>
    layer_dense(units = units1, activation = "relu",
                input_shape = input_shape) |>
    layer_dense(units = units2, activation = "relu") |>
    layer_dense(units = 1, activation = "sigmoid") |>
    compile(
      loss = "binary_crossentropy",
      optimizer = optimizer_sgd(learning_rate = lr),
      metrics = "accuracy"
    )
}
ann <- build_ann(ncol(train_mat))

### 4.3 Fit (10 epochs, batch 64) ---------------------------------------------
history <- ann |>
  fit(train_mat, train_labels,
      epochs = 10, batch_size = 64,
      validation_split = 0.15, verbose = 2)

### 4.4 Evaluate ---------------------------------------------------------------
ann_metrics <- ann |> evaluate(test_mat, test_labels, verbose = 0)
ann_acc <- ann_metrics["accuracy"] |> as.numeric()
cat(glue::glue("ANN accuracy          : {round(ann_acc,4)}\n"))

###############################################################################
# Expect ≈ 0.911 NB  vs.  0.639 ANN (may vary ±2 pp by seed / TF version).
###############################################################################
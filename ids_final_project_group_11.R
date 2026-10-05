#library(readxl)
library(dplyr)
library(stringr)
library(e1071)
library(caret)
library(glmnet)
library(ggplot2)
library(reshape2)
library(gridExtra)

df <- read.csv("D:/University/Semesters/Semester 11/Data Science/Projects/FINAL/ids_final_project_group_11.csv")

head(df, 10)
print(table(df$label))
print(colSums(is.na(df)))
print(sum(duplicated(df)))
print(sum(duplicated(df$id)))
print(n_distinct(df$question))

df <- df %>%
  mutate(
    ref_len = str_count(reference_passage, "\\S+"),
    ans_len = str_count(llm_answer, "\\S+")
  )

print(round(mean(df$ref_len), 2))
print(round(mean(df$ans_len), 2))


clean_text <- function(text) {
  text <- str_trim(text)
  text <- str_replace_all(text, "[।,;:!?\"'()\\[\\]{}—–\\-…\\.]", " ")
  text <- str_squish(text)
  return(text)
}

bangla_digits <- c("০","১","২","৩","৪","৫","৬","৭","৮","৯")
english_digits <- as.character(0:9)

normalize_text <- function(text) {
  text <- str_to_lower(text)
  for (i in seq_along(bangla_digits)) {
    text <- str_replace_all(text, bangla_digits[i], english_digits[i])
  }
  return(text)
}

bangla_stopwords <- c(
  "এর","এই","ও","এবং","কিন্তু","যে","যা","তা","এটা","এটি","ওটা",
  "হয়","হয়েছে","হবে","ছিল","আছে","করে","করা","করেছে","হলো","হল",
  "কি","কী","কেন","কোথায়","কখন","কিভাবে","কাকে","কার","কারা",
  "আমি","আমরা","তুমি","তোমরা","সে","তারা","তিনি","এখানে","সেখানে",
  "থেকে","পর্যন্ত","জন্য","সাথে","দিয়ে","নিয়ে","মধ্যে","উপর","প্রতি",
  "একটি","একটা","কোনো","সব","সকল","অনেক","কিছু","আরও","আরো"
)

remove_stopwords <- function(tokens) {
  tokens[!(tokens %in% bangla_stopwords)]
}

bangla_stem <- function(word) {
  if (str_detect(word, "^[0-9]+(\\.[0-9]+)?[%°]?$")) return(word)
  
  suffixes <- c("দের","েরা","গুলো","গুলি","টির","টার","টি","টা",
                "কে","এর","তে","শে","ঠা","ও","ই","র","ে")
  
  suffixes <- suffixes[order(-nchar(suffixes))]
  for (suf in suffixes) {
    if (str_detect(word, paste0(suf, "$")) && (nchar(word) - nchar(suf)) >= 2) {
      word <- str_replace(word, paste0(suf, "$"), "")
      break
    }
  }
  return(word)
}

tokenize_text <- function(text) {
  tokens <- str_split(text, "\\s+")[[1]]
  tokens <- tokens[tokens != ""]
  sapply(tokens, bangla_stem, USE.NAMES = FALSE)
}

preprocess_text <- function(text) {
  text <- clean_text(text)
  text <- normalize_text(text)
  tokens <- tokenize_text(text)
  tokens <- remove_stopwords(tokens)
  paste(tokens, collapse = " ")
}

df <- df %>%
  mutate(
    clean_reference = sapply(reference_passage, preprocess_text),
    clean_answer = sapply(llm_answer, preprocess_text)
  )
print(df %>% select(reference_passage, clean_reference) %>% head(5))
df %>% select(llm_answer, clean_answer) %>% head(5)

df_preprocessed <- df

ref_tokens <- lapply(df$clean_reference, tokenize_text)
ans_tokens <- lapply(df$clean_answer, tokenize_text)

N <- nrow(df)

vocabulary <- sort(unique(c(unlist(ref_tokens), unlist(ans_tokens))))
V <- length(vocabulary)

build_bow_matrix <- function(doc_tokens, vocabulary) {
  n_docs <- length(doc_tokens)
  V <- length(vocabulary)
  bow <- matrix(0L, nrow = n_docs, ncol = V,
                dimnames = list(NULL, vocabulary))
  for (i in seq_len(n_docs)) {
    counts <- table(doc_tokens[[i]])
    bow[i, names(counts)] <- as.integer(counts)
  }
  return(bow)
}

bow_ref <- build_bow_matrix(ref_tokens, vocabulary)
bow_ans <- build_bow_matrix(ans_tokens, vocabulary)

extract_numbers <- function(tokens) {
  nums <- tokens[str_detect(tokens, "^[0-9]+(\\.[0-9]+)?$")]
  as.numeric(nums)
}

number_mismatch_features <- function(ref_toks, ans_toks) {
  n <- length(ref_toks)
  feat <- matrix(0, nrow = n, ncol = 3,
                 dimnames = list(NULL, c("num_in_ans_not_ref",
                                         "num_in_ref_not_ans",
                                         "num_overlap_ratio")))
  for (i in seq_len(n)) {
    rn <- extract_numbers(ref_toks[[i]])
    an <- extract_numbers(ans_toks[[i]])
    feat[i, "num_in_ans_not_ref"] <- length(setdiff(an, rn))
    feat[i, "num_in_ref_not_ans"] <- length(setdiff(rn, an))
    feat[i, "num_overlap_ratio"] <- ifelse(length(union(rn, an)) == 0, 1,
                                           length(intersect(rn, an)) / length(union(rn, an)))
  }
  feat
}

overlap_features <- function(ref_toks, ans_toks) {
  n <- length(ref_toks)
  feat <- matrix(0, nrow = n, ncol = 2,
                 dimnames = list(NULL, c("jaccard_sim", "ans_not_in_ref_ratio")))
  for (i in seq_len(n)) {
    r <- ref_toks[[i]]; a <- ans_toks[[i]]
    u <- union(r, a)
    feat[i, "jaccard_sim"] <- ifelse(length(u) == 0, 0, length(intersect(r, a)) / length(u))
    feat[i, "ans_not_in_ref_ratio"] <- ifelse(length(a) == 0, 0, length(setdiff(a, r)) / length(a))
  }
  feat
}

num_feat <- number_mismatch_features(ref_tokens, ans_tokens)
ovl_feat <- overlap_features(ref_tokens, ans_tokens)

bow_diff <- bow_ans - bow_ref
colnames(bow_diff) <- paste0(colnames(bow_diff), "_diff")

count_block <- cbind(bow_ans, bow_diff)
engineered_block <- cbind(num_feat, ovl_feat)

bow_matrix <- cbind(count_block, engineered_block)

doc_lengths <- mapply(function(r, a) length(r) + length(a), ref_tokens, ans_tokens)
doc_lengths[doc_lengths == 0] <- 1

tf_count_block <- count_block / doc_lengths
df_counts <- colSums(count_block > 0)

print(head(sort(df_counts, decreasing = TRUE), 10))
idf_values <- log10(N / (df_counts + 1)) + 1

print(head(sort(idf_values, decreasing = TRUE), 10))
tfidf_count_block <- sweep(tf_count_block, 2, idf_values, "*")

tfidf_matrix <- cbind(tfidf_count_block, engineered_block)

stopifnot(!any(duplicated(colnames(bow_matrix))))
stopifnot(!any(duplicated(colnames(tfidf_matrix))))

d1_terms <- unique(c(ref_tokens[[1]], ans_tokens[[1]]))
for (term in d1_terms) {
  if (term %in% colnames(tf_count_block)) {
    cat(sprintf("  %-20s TF=%.4f  DF=%d  IDF=%.4f  TF-IDF=%.4f\n",
                term,
                tf_count_block[1, term],
                df_counts[term],
                idf_values[term],
                tfidf_count_block[1, term]))
  }
}

set.seed(42)

labels <- factor(df$label, levels = c("faithful", "hallucinated"))

unique_questions <- unique(df$question)
K <- 5
question_folds <- createFolds(unique_questions, k = K, list = TRUE)

evaluate <- function(y_true, y_pred, model_name, fold_id = NA) {
  cm <- table(Predicted = y_pred, Actual = y_true)
  for (lev in c("faithful", "hallucinated")) {
    if (!(lev %in% rownames(cm))) cm <- rbind(cm, setNames(0, lev))
    if (!(lev %in% colnames(cm))) cm <- cbind(cm, setNames(0, lev))
  }
  cm <- cm[c("faithful", "hallucinated"), c("faithful", "hallucinated")]
  
  acc <- sum(diag(cm)) / sum(cm)
  tp <- cm["hallucinated", "hallucinated"]
  fp <- cm["hallucinated", "faithful"]
  fn <- cm["faithful", "hallucinated"]
  
  precision <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
  recall <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
  f1 <- ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
  
  list(model = model_name, fold = fold_id, accuracy = acc, precision = precision,
       recall = recall, f1 = f1, confusion = cm)
}

fit_logreg_cv <- function(X_train, y_train, X_test) {
  y_bin <- ifelse(y_train == "hallucinated", 1, 0)
  
  cvfit <- cv.glmnet(X_train, y_bin, family = "binomial", alpha = 0, nfolds = 5)
  probs <- as.numeric(predict(cvfit, newx = X_test, s = "lambda.min", type = "response"))
  pred <- ifelse(probs > 0.5, "hallucinated", "faithful")
  factor(pred, levels = c("faithful", "hallucinated"))
}

model_names <- c("LogReg_BoW", "LogReg_TFIDF", "SVM_BoW", "SVM_TFIDF")
fold_results <- list()
conf_matrices <- setNames(vector("list", length(model_names)), model_names)
for (m in model_names) conf_matrices[[m]] <- matrix(0, 2, 2, dimnames = list(c("faithful", "hallucinated"), c("faithful", "hallucinated")))

for (k in seq_len(K)) {
  cat(sprintf("\n============ Fold %d / %d ============\n", k, K))
  
  test_questions  <- unique_questions[question_folds[[k]]]
  train_questions <- setdiff(unique_questions, test_questions)
  
  train_idx <- which(df$question %in% train_questions)
  test_idx  <- which(df$question %in% test_questions)
  stopifnot(length(intersect(df$question[train_idx], df$question[test_idx])) == 0)
  
  y_train <- labels[train_idx]
  y_test  <- labels[test_idx]
  
  X_train_bow   <- bow_matrix[train_idx, ]
  X_test_bow    <- bow_matrix[test_idx, ]
  X_train_tfidf <- tfidf_matrix[train_idx, ]
  X_test_tfidf  <- tfidf_matrix[test_idx, ]
  
  # --- Logistic Regression (ridge, glmnet) ---
  pred_logreg_bow <- fit_logreg_cv(X_train_bow, y_train, X_test_bow)
  res <- evaluate(y_test, pred_logreg_bow, "LogReg_BoW", k); fold_results[[length(fold_results)+1]] <- res
  conf_matrices[["LogReg_BoW"]] <- conf_matrices[["LogReg_BoW"]] + res$confusion
  
  pred_logreg_tfidf <- fit_logreg_cv(X_train_tfidf, y_train, X_test_tfidf)
  res <- evaluate(y_test, pred_logreg_tfidf, "LogReg_TFIDF", k); fold_results[[length(fold_results)+1]] <- res
  conf_matrices[["LogReg_TFIDF"]] <- conf_matrices[["LogReg_TFIDF"]] + res$confusion
  
  # --- SVM (linear kernel, scaled so BoW's raw counts and TF-IDF's decimals are comparable) ---
  svm_bow <- svm(x = X_train_bow, y = y_train, kernel = "linear", scale = TRUE)
  pred_svm_bow <- predict(svm_bow, X_test_bow)
  res <- evaluate(y_test, pred_svm_bow, "SVM_BoW", k); fold_results[[length(fold_results)+1]] <- res
  conf_matrices[["SVM_BoW"]] <- conf_matrices[["SVM_BoW"]] + res$confusion
  
  svm_tfidf <- svm(x = X_train_tfidf, y = y_train, kernel = "linear", scale = TRUE)
  pred_svm_tfidf <- predict(svm_tfidf, X_test_tfidf)
  res <- evaluate(y_test, pred_svm_tfidf, "SVM_TFIDF", k); fold_results[[length(fold_results)+1]] <- res
  conf_matrices[["SVM_TFIDF"]] <- conf_matrices[["SVM_TFIDF"]] + res$confusion
}

fold_df <- do.call(rbind, lapply(fold_results, function(r) {
  data.frame(Model = r$model, Fold = r$fold, Accuracy = r$accuracy,
             Precision = r$precision, Recall = r$recall, F1 = r$f1)
}))

summary_df <- fold_df %>%
  group_by(Model) %>%
  summarise(
    Accuracy = mean(Accuracy),
    Precision = mean(Precision),
    Recall = mean(Recall),
    F1 = mean(F1)
  ) %>%
  arrange(desc(F1)) %>%
  as.data.frame()

print(summary_df, row.names = FALSE)

write.csv(fold_df, "model_results_per_fold.csv", row.names = FALSE)
write.csv(summary_df, "model_results_R.csv", row.names = FALSE)
for (m in model_names) {
  cat(sprintf("\n--- %s ---\n", m))
  print(conf_matrices[[m]])
}

plot_confusion <- function(cm, title) {
  cm_df <- as.data.frame(as.table(cm))
  colnames(cm_df) <- c("Predicted", "Actual", "Freq")
  cm_df$Actual <- factor(cm_df$Actual, levels = c("faithful", "hallucinated"))
  cm_df$Predicted <- factor(cm_df$Predicted, levels = c("hallucinated", "faithful"))
  
  ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile(color = "white", linewidth = 1) +
    geom_text(aes(label = Freq), color = "black", size = 5, fontface = "bold") +
    scale_fill_gradient(low = "#EAF2FB", high = "#2C6FBB") +
    labs(title = title, x = "Actual", y = "Predicted", fill = "Count") +
    theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 12),
          panel.grid = element_blank(),
          axis.text = element_text(size = 9))
}

cm_plots <- lapply(model_names, function(m) plot_confusion(conf_matrices[[m]], m))
cm_grid <- grid.arrange(grobs = cm_plots, ncol = 2,
                        top = "Confusion Matrices (summed over 5-fold CV) - All 4 Model x Representation Combinations")
ggsave("confusion_matrices_all_models.png", cm_grid, width = 14, height = 9, dpi = 150)

metrics_long <- melt(summary_df[, c("Model", "Accuracy", "Precision", "Recall", "F1")],
                     id.vars = "Model", variable.name = "Metric", value.name = "Score")

metrics_long$Representation <- ifelse(grepl("TFIDF", metrics_long$Model), "TF-IDF", "BoW")
metrics_long$Model <- factor(metrics_long$Model, levels = summary_df$Model)

benchmark_plot <- ggplot(metrics_long, aes(x = Model, y = Score, fill = Representation)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  geom_text(aes(label = sprintf("%.2f", Score)),
            position = position_dodge(width = 0.8), vjust = -0.35, size = 3) +
  facet_wrap(~ Metric, nrow = 1) +
  scale_fill_manual(values = c("BoW" = "#4C9F70", "TF-IDF" = "#2C6FBB")) +
  coord_cartesian(ylim = c(0, 1.05)) +
  labs(title = "5-Fold CV Benchmark Comparison: Accuracy, Precision, Recall, F1",
       subtitle = "4 technique x representation combinations (Logistic Regression / SVM x BoW / TF-IDF)",
       x = NULL, y = "Score") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        plot.title = element_text(face = "bold"),
        legend.position = "top",
        strip.text = element_text(face = "bold"))

ggsave("benchmark_comparison_plot.png", benchmark_plot, width = 12, height = 6, dpi = 150)

cat("\nSaved: confusion_matrices_all_models.png, benchmark_comparison_plot.png, ",
    "model_results_R.csv, model_results_per_fold.csv\n")
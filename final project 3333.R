##Author: Timothy Ogbonnaya
##Student Number: 219398221
##Final Project for Math 3333



# === Load Libraries ====
library(jpeg)
library(nnet)
library(caret)
library(pROC)


# === Load Metadata ====
meta <- read.csv("C:/Users/timoo/OneDrive/Documents/photoMetaData.csv")
img_dir <- "C:/Users/timoo/OneDrive/Documents/columbiaImages/"

n <- nrow(meta)
y <- as.numeric(meta$category == "outdoor-day")  # Outcome: 1 = outdoor, 0 = not

# === Feature Extraction Function ====
extract_grid_means <- function(path, size = 100, grids = 10) {
  img <- readJPEG(path)
  gray <- 0.2989 * img[,,1] + 0.5870 * img[,,2] + 0.1140 * img[,,3]  # Grayscale
  dim_grid <- size / grids
  resized <- matrix(gray[1:size, 1:size], nrow = size, ncol = size)
  features <- c()
  for (i in 0:(grids - 1)) {
    for (j in 0:(grids - 1)) {
      block <- resized[(i * dim_grid + 1):((i + 1) * dim_grid),
                       (j * dim_grid + 1):((j + 1) * dim_grid)]
      features <- c(features, mean(block, na.rm = TRUE))
    }
  }
  return(features)
}

# === Extract Features from All Images ====
X <- matrix(NA, nrow = n, ncol = 100)
for (i in 1:n) {
  path <- paste0(img_dir, meta$name[i])
  if (file.exists(path)) {
    cat(sprintf("Processing %03d/%03d: %s\n", i, n, meta$name[i]))
    X[i, ] <- tryCatch(extract_grid_means(path), error = function(e) rep(NA, 100))
  } else {
    cat(sprintf("Missing file: %s\n", meta$name[i]))
  }
}

# === Remove Rows with All NAs (Completely Invalid Images) ====
valid_rows <- complete.cases(X)
X <- X[valid_rows, ]
y <- y[valid_rows]

# === Train/Test Split ===
set.seed(123)
train_idx <- sample(1:nrow(X), size = 0.7 * nrow(X))
test_idx <- setdiff(1:nrow(X), train_idx)

X_train <- X[train_idx, ]
X_test <- X[test_idx, ]
y_train <- y[train_idx]
y_test <- y[test_idx]

# === Mean Imputation for Any Remaining NA ====
for (i in 1:ncol(X_train)) {
  X_train[is.na(X_train[, i]), i] <- mean(X_train[, i], na.rm = TRUE)
}
for (i in 1:ncol(X_test)) {
  X_test[is.na(X_test[, i]), i] <- mean(X_test[, i], na.rm = TRUE)
}

# === Final NA Check (Remove Bad Rows If Still Present) ====
X_train <- X_train[complete.cases(X_train), ]
y_train <- y_train[1:nrow(X_train)]

X_test <- X_test[complete.cases(X_test), ]
y_test <- y_test[1:nrow(X_test)]

# === Train Neural Network ====
nn_model <- nnet(X_train, y_train, size = 5, maxit = 500, decay = 0.01)

# === Predict and Evaluate ===
pred_probs <- predict(nn_model, X_test, type = "raw")
pred_class <- ifelse(pred_probs > 0.5, 1, 0)

conf <- confusionMatrix(factor(pred_class), factor(y_test), positive = "1")
print(conf)

# === Custom Summary ====
cat(sprintf("\nAccuracy: %.2f%%", mean(pred_class == y_test) * 100))
cat(sprintf("\nSensitivity: %.2f%%", conf$byClass["Sensitivity"] * 100))
cat(sprintf("\nSpecificity: %.2f%%", conf$byClass["Specificity"] * 100))
cat(sprintf("\nMisclassification Rate: %.2f%%", mean(pred_class != y_test) * 100))

#=== Bar Chart of distribution ====

# Load libraries
library(ggplot2)


#=== bar chart of category frequencies ====
ggplot(meta, aes(x = category)) +
  geom_bar(fill = "steelblue") +
  theme_minimal() +
  labs(title = "Distribution of Image Categories",
       x = "Category",
       y = "Number of Images") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))




#=== Comparison of the Logistic Model ====

## Read in the library and metadata
library(jpeg)
pm <- read.csv("C:/Users/timoo/OneDrive/Documents/photoMetaData.csv")
n <- nrow(pm)

trainFlag <- (runif(n) > 0.5)
y <- as.numeric(pm$category == "outdoor-day")

X <- matrix(NA, ncol=3, nrow=n)
for (j in 1:n) {
  img <- readJPEG(paste0("C:/Users/timoo/OneDrive/Documents/columbiaImages/",pm$name[j]))
  X[j,] <- apply(img,3,median)
  print(sprintf("%03d / %03d", j, n))
}


# build a glm model on these median values
out <- glm(y ~ X, family=binomial, subset=trainFlag)
out$iter
summary(out)

# How well did we do?
pred <- 1 / (1 + exp(-1 * cbind(1,X) %*% coef(out)))
y[order(pred)]
y[!trainFlag][order(pred[!trainFlag])]

mean((as.numeric(pred > 0.5) == y)[trainFlag])
mean((as.numeric(pred > 0.5) == y)[!trainFlag])





# === Train neural network on same data (3 median RGB features)
nn_model <- nnet(X[trainFlag, ], y[trainFlag], size = 5, maxit = 500, decay = 0.01)

# === Predict probabilities for test set
nn_pred <- predict(nn_model, X[!trainFlag, ], type = "raw")
glm_pred <- pred[!trainFlag]
actual <- y[!trainFlag]

# === Combine predictions for both models
df <- data.frame(
  prob = c(glm_pred, nn_pred),
  model = rep(c("Logistic Regression", "Neural Network"), each = length(glm_pred)),
  actual = factor(rep(actual, 2), levels = c(0, 1), labels = c("Not Outdoor", "Outdoor"))
)

# === Plot histogram comparison
library(ggplot2)
ggplot(df, aes(x = prob, fill = actual)) +
  geom_histogram(position = "identity", bins = 25, alpha = 0.6) +
  facet_wrap(~ model) +
  scale_fill_manual(values = c("steelblue", "darkorange")) +
  labs(title = "Histogram of Predicted Probabilities",
       x = "Predicted Probability",
       y = "Count",
       fill = "Actual Class") +
  theme_minimal()

install.packages('tinytex')

# Get predictions for test set
glm_pred <- pred[!trainFlag]
glm_class <- ifelse(glm_pred > 0.5, 1, 0)
actual <- y[!trainFlag]

conf_logit <- confusionMatrix(factor(glm_class), factor(actual), positive = "1")
print(conf_logit)

# Accuracy
mean(glm_class == actual) * 100

# Misclassification Rate
mean(glm_class != actual) * 100

# Sensitivity (True Positive Rate)
conf_logit$byClass["Sensitivity"] * 100

# Specificity (True Negative Rate)
conf_logit$byClass["Specificity"] * 100
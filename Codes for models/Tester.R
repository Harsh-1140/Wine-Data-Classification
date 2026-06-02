# --- Correct Local Test Script ---

# Load your saved function and coefficients
load('230213.Rdata')
test <- read.csv("wine_quality_train.csv")
# Assuming 'test' is your holdout set

# 1. Separate the response (y) from the predictors (X)
y_test <- test$quality
predictor_cols <- setdiff(names(test), "quality")
X_test_predictors <- test[, predictor_cols] # This now has 12 columns, as required

# 2. Call the function with ONLY the predictor data
# This will resolve the warnings and give a correct result.
X_test_matrix <- make_model_matrix(X_test_predictors)

# 3. Calculate the Mean Squared Error (MSE)
y_pred <- X_test_matrix %*% beta.hat
mse <- mean((y_test - y_pred)^2)

cat("\n✅ Test MSE:", mse, "\n")



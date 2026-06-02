# Load necessary libraries
library(MASS)
library(AICcmodavg)

# Set a seed for reproducibility
set.seed(123)

# ------------------ 1. Load and Prepare Data ------------------
wine <- read.csv("wine_quality_train.csv")
predictors <- setdiff(names(wine), "quality")

# Calculate the mean and standard deviation for each predictor.
means <- sapply(wine[predictors], mean)
sds <- sapply(wine[predictors], sd)

# Scale the entire training dataset
wine_scaled <- wine
wine_scaled[predictors] <- scale(wine[predictors])

# ------------------ 2. Build and Select the Model ------------------
formula_full <- as.formula(paste("quality ~ (", paste(predictors, collapse = " + "), ")^2"))
full_model <- lm(formula_full, data = wine_scaled)

# Define a function to perform backward variable selection using AICc
backward_aicc <- function(model) {
  best <- model
  best_aicc <- AICc(model)
  improved <- TRUE
  while (improved) {
    improved <- FALSE
    current_terms <- attr(terms(best), "term.labels")
    if (length(current_terms) == 0) break
    
    for (term in current_terms) {
      test_model <- update(best, as.formula(paste(". ~ . -", term)))
      new_aicc <- AICc(test_model)
      
      if (new_aicc < best_aicc) {
        best <- test_model
        best_aicc <- new_aicc
        improved <- TRUE
        break
      }
    }
  }
  return(best)
}

# Run the selection process to find the best model
best_model <- backward_aicc(full_model)
beta.hat <- coef(best_model)


create_model_matrix_function <- function(means_vec, sds_vec, final_model) {
  force(means_vec); force(sds_vec); force(final_model) 
  
  function(X.test) {
    new_scaled <- sweep(X.test, 2, means_vec, FUN = "-")
    new_scaled <- sweep(new_scaled, 2, sds_vec, FUN = "/")
    df <- as.data.frame(new_scaled)
    
    predictor_formula <- delete.response(terms(final_model))
    model.matrix(predictor_formula, data = df)
  }
}

# Now, call the factory ONE TIME to create our self-contained function
make_model_matrix <- create_model_matrix_function(means, sds, best_model)

save(make_model_matrix, beta.hat, file = "230443.Rdata")

cat("\n Final, self-contained submission file '230443.Rdata' created successfully!\n")

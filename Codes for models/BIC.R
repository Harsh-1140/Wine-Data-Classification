# ------------------ MTH441: Assignment 1 ------------------
# Uses BIC-based model pruning to minimize test MSE

library(MASS)
library(glmnet)
set.seed(123)

# ------------------ Load dataset ------------------
data <- read.csv("wine_quality_train.csv")
stopifnot("quality" %in% names(data))

scale_cols <- setdiff(names(data), "quality")

# ------------------ Build full quadratic formula ------------------
form_full <- as.formula(
  paste("quality ~ (", paste(scale_cols, collapse = " + "), ")^2")
)

# ------------------ Fit model and prune with stepAIC using BIC ------------------
fit_full <- lm(form_full, data = data)
fitt <- stepAIC(fit_full, trace = FALSE, k = log(nrow(data)))  # <-- BIC penalty
beta.hat <- coef(fitt)

# ------------------ Prepare RHS formula for prediction ------------------
rhs_formula <- formula(fitt)[-2]  # right-hand side only (drop 'quality ~')

# ------------------ Define make_model_matrix (self-contained) ------------------
make_model_matrix <- local({
  f <- rhs_formula  # capture formula inside environment
  function(X.test) {
    df <- as.data.frame(X.test)
    X_new <- model.matrix(f, data = df)
    return(X_new)
  }
})

save(make_model_matrix, beta.hat, file = "230443.Rdata")
cat("\n Successfully saved BIC-optimized self-contained 230443.Rdata\n")

summary(fitt)

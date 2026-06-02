# ------------------ MTH441: Assignment 1 ------------------

library(MASS)
library(glmnet)
set.seed(123)

# ------------------ Load dataset ------------------
data <- read.csv("wine_quality_train.csv")
stopifnot("quality" %in% names(data))
scale_cols <- setdiff(names(data), "quality")

# ------------------ Build full quadratic formula ------------------
# This includes main effects, squared terms, and all pairwise interactions
form_full <- as.formula(
  paste("quality ~ (", paste(scale_cols, collapse = " + "), ")^2")
)

# ------------------ Fit linear model and prune with stepAIC ------------------
fit_full <- lm(form_full, data = data)
fitt <- stepAIC(fit_full, trace = FALSE)
beta.hat <- coef(fitt)

rhs_formula <- formula(fitt)[-2]

make_model_matrix <- local({
  f <- rhs_formula
  function(X.new) {
    df <- as.data.frame(X.new)
    model.matrix(f, data = df)
  }
})

save(make_model_matrix, beta.hat, file = "230443.Rdata")

summary(fitt)



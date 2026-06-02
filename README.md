# Wine Quality Prediction using Linear Model Selection

This repository contains the implementation for **MTH441: Assignment 1**. The primary objective is to build a robust linear regression framework that minimizes the Mean Squared Prediction Error (MSPE) on an unobserved test dataset of wine profiles.

The project leverages statistical pruning strategies—specifically **Akaike Information Criterion (AIC)**, **Corrected AIC (AICc)**, and **Bayesian Information Criterion (BIC)**—applied to a full quadratic model space to systematically find an optimal balance between complexity and predictive power.

---

## Project Structure

* **`wine_quality_train.csv`**: The training dataset containing chemical properties of various tested wines and their subjective quality score.
* **`AIC.R`**: Scripts implementing backward elimination based on the standard Akaike Information Criterion.
* **`AICc.R`**: Scripts using the corrected AIC metric to handle small sample bias and avoid overfitting in rich parameter spaces.
* **`BIC.R`**: Scripts utilizing the Bayesian Information Criterion penalty ($\ln(n)$) to yield a more parsimonious model.
* **`Tester.R`**: A local validation script simulating the instructor's grading pipeline to ensure format compliance and error tracking.
* **`230443.Rdata`**: The final submission file containing the optimized regression coefficients (`beta.hat`) and a self-contained environment factory function (`make_model_matrix`).

---

## Methodology & Feature Engineering

To capture non-linear trends and synergistic interactions among the wine attributes without manually guessing structures, we initialize our modeling in a higher-dimensional space:

1. **Full Quadratic Expansion**: We construct a massive initial model space containing all $12$ main chemical features, their squared terms (quadratic curves), and all unique pairwise interactions:

$$\text{quality} \sim (\text{fixed acidity} + \text{volatile acidity} + \dots + \text{alcohol})^2$$


2. **Feature Standardization**: To prevent magnitude bias from dominating interactive terms, features are systematically scaled using training means and standard deviations.
3. **Stepwise Model Pruning**: Using step-down optimization algorithms (`MASS::stepAIC`), the full model is dynamically pruned. We contrasted three penalties to choose our final setup:
* **AIC**: Optimizes for predictive Kullback-Leibler distance ($k = 2$).
* **AICc**: Penalizes extra parameters heavier to account for finite sample sizes.
* **BIC**: Imposes a rigorous penalty ($k = \ln(n)$) favoring simpler, highly interpretable structures.



---

## Submission Architecture

Per the assignment criteria, the finalized deployment payload is saved into an isolated binary file (`230443.Rdata`) hosting exactly two elements:

1. **`beta.hat`**: A vector of the finalized regression coefficients.
2. **`make_model_matrix(X.test)`**: A clean, **self-contained evaluation function** wrapped within a localized environment context (`local()`). It accepts a raw $1000 \times 12$ matrix of test features, dynamically matches the exact transformations and interactions used during training, and outputs a formatted mathematical matrix ready for direct vector multiplication without leaking variables.

---

## Local Evaluation Pipeline

To verify model structure validity and calculate the Mean Squared Error locally prior to pushing updates to the leaderboard, execute the verification script:

```R
# Load compliance framework
source("Tester.R")

```

The underlying script structural logic mimics the grading engine:

```R
# 1. Load the compressed submission payload
load('230443.Rdata')

# 2. Extract predictors from a test holdout 
y_test <- test_data$quality
X_test_predictors <- test_data[, setdiff(names(test_data), "quality")]

# 3. Build compatible model matrix structures
X_test_matrix <- make_model_matrix(X_test_predictors)

# 4. Generate matrix predictions and assess performance
y_pred <- X_test_matrix %*% beta.hat
mse <- mean((y_test - y_pred)^2)

cat("Validation Pipeline Mean Squared Error:", mse, "\n")

```

---

The **AICc** model showed the best results with MSPE 0.501583321

# Regression: Overall Rating Prediction

Predicts `overall_rating` (47-91) from 28 skill attributes.

`reactions` is excluded on purpose. It correlates 0.86 with the target
because FIFA derives it from the overall rating, so including it would be
data leakage.

Pipeline: StandardScaler → PCA (15 components) → Ridge (alpha=1.0).

Step 7 retrains a one-feature Linear Regression on the same split so
Simple and Multiple can be compared fairly.

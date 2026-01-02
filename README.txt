# inspectML

*A lightweight evaluation and inspection toolkit for Machine Learning models.*

---

## Overview

**inspectML** is a simple and modular Python library designed to help you **evaluate** and **inspect** the performance of machine learning models.
It provides both **regression** and **classification (clustering)** metrics under a consistent, easy-to-use API.

---

## Features

### Regression Metrics (`inspectML.evaluation.regression`)

| Function     | Full Name                           | Description                                                         |
| ------------ | ----------------------------------- | ------------------------------------------------------------------- |
| `mse()`      | Mean Squared Error                  | Average of squared differences between predicted and actual values  |
| `mae()`      | Mean Absolute Error                 | Average of absolute differences between predicted and actual values |
| `rmse()`     | Root Mean Squared Error             | Square root of MSE                                                  |
| `evaluate()` | Evaluate All                        | Returns a dict containing MSE, MAE, and RMSE                        |
| `mape()`     | Mean Absolute Percentage Error      | Measures prediction accuracy as a percentage                        |
| `r2()`       | R-squared                           | Coefficient of determination                                        |
| `adj_r2()`   | Adjusted R-squared                  | Adjusted for number of predictors                                   |
| `msle()`     | Mean Squared Logarithmic Error      | Penalizes underestimation more than overestimation                  |
| `rmsle()`    | Root Mean Squared Logarithmic Error | Square root of MSLE                                                 |
| `medAE()`    | Median Absolute Error               | Median of absolute errors                                           |
| `evs()`      | Explained Variance Score            | Measures proportion of variance explained by model                  |

---

### Classification / Clustering Metrics (`inspectML.evaluation.classification`)

| Function  | Full Name                     | Description                                                               |
| --------- | ----------------------------- | ------------------------------------------------------------------------- |
| `SIL()`   | Silhouette Score              | Measures how similar a point is to its own cluster vs others              |
| `DBI()`   | Davies–Bouldin Index          | Lower values indicate better clustering separation                        |
| `CHI()`   | Calinski–Harabasz Index       | Higher values indicate better-defined clusters                            |
| `DI()`    | Dunn Index                    | Ratio of minimal inter-cluster distance to maximal intra-cluster distance |
| `inert()` | Inertia                       | Sum of squared distances to the nearest cluster center                    |
| `ARI()`   | Adjusted Rand Index           | Similarity between two clusterings, adjusted for chance                   |
| `NMI()`   | Normalized Mutual Information | Measures mutual dependence between label assignments                      |
| `HOM()`   | Homogeneity Score             | Each cluster contains only members of a single class                      |
| `COMP()`  | Completeness Score            | All members of a given class are assigned to the same cluster             |
| `VMEAS()` | V-Measure                     | Harmonic mean of Homogeneity and Completeness                             |
| `F1()`    | F1 Score                      | Harmonic mean of precision and recall (pairwise)                          |

---

## Installation
pip install inspectML

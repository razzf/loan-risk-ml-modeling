# Summary of 2_DecisionTree

[<< Go back](../README.md)


## Decision Tree
- **n_jobs**: -1
- **criterion**: entropy
- **max_depth**: 2
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
auc

## Training time

166.3 seconds

## Metric details
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.643405 |  nan        |
| auc       | 0.666149 |  nan        |
| f1        | 0.666625 |    0.317245 |
| accuracy  | 0.649811 |    0.462353 |
| precision | 0.695515 |    0.618957 |
| recall    | 1        |    0.317245 |
| mcc       | 0.302593 |    0.462353 |


## Metric details with threshold from accuracy metric
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.643405 |  nan        |
| auc       | 0.666149 |  nan        |
| f1        | 0.623416 |    0.462353 |
| accuracy  | 0.649811 |    0.462353 |
| precision | 0.674162 |    0.462353 |
| recall    | 0.579774 |    0.462353 |
| mcc       | 0.302593 |    0.462353 |


## Confusion matrix (at threshold=0.462353)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |         11068    |          4307.75 |
| Labeled as 1 |          6460.05 |          8912.77 |

## Learning curves
![Learning curves](learning_curves.png)

## Permutation-based Importance
![Permutation-based Importance](permutation_importance.png)
## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)


## Normalized Confusion Matrix

![Normalized Confusion Matrix](confusion_matrix_normalized.png)


## ROC Curve

![ROC Curve](roc_curve.png)


## Kolmogorov-Smirnov Statistic

![Kolmogorov-Smirnov Statistic](ks_statistic.png)


## Precision-Recall Curve

![Precision-Recall Curve](precision_recall_curve.png)


## Calibration Curve

![Calibration Curve](calibration_curve_curve.png)


## Cumulative Gains Curve

![Cumulative Gains Curve](cumulative_gains_curve.png)


## Lift Curve

![Lift Curve](lift_curve.png)



[<< Go back](../README.md)

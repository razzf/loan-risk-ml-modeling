# Summary of 1_DecisionTree

[<< Go back](../README.md)


## Decision Tree
- **n_jobs**: -1
- **criterion**: entropy
- **max_depth**: 4
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
auc

## Training time

203.4 seconds

## Metric details
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.630608 |  nan        |
| auc       | 0.697471 |  nan        |
| f1        | 0.688224 |    0.34978  |
| accuracy  | 0.645278 |    0.523929 |
| precision | 0.856496 |    0.762432 |
| recall    | 1        |    0.189036 |
| mcc       | 0.294387 |    0.523929 |


## Metric details with threshold from accuracy metric
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.630608 |  nan        |
| auc       | 0.697471 |  nan        |
| f1        | 0.614356 |    0.523929 |
| accuracy  | 0.645278 |    0.523929 |
| precision | 0.673101 |    0.523929 |
| recall    | 0.565042 |    0.523929 |
| mcc       | 0.294387 |    0.523929 |


## Confusion matrix (at threshold=0.523929)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |         11155.6  |          4220.19 |
| Labeled as 1 |          6689.06 |          8689.59 |

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

# Summary of 33_RandomForest

[<< Go back](../README.md)


## Random Forest
- **n_jobs**: -1
- **criterion**: entropy
- **max_features**: 0.8
- **min_samples_split**: 50
- **max_depth**: 7
- **eval_metric_name**: auc
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
auc

## Training time

2741.2 seconds

## Metric details
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.60445  | nan         |
| auc       | 0.736117 | nan         |
| f1        | 0.707902 |   0.33748   |
| accuracy  | 0.673855 |   0.499665  |
| precision | 0.886905 |   0.810118  |
| recall    | 1        |   0.0720153 |
| mcc       | 0.348922 |   0.506604  |


## Metric details with threshold from accuracy metric
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.60445  |  nan        |
| auc       | 0.736117 |  nan        |
| f1        | 0.662711 |    0.499665 |
| accuracy  | 0.673855 |    0.499665 |
| precision | 0.686223 |    0.499665 |
| recall    | 0.640757 |    0.499665 |
| mcc       | 0.348479 |    0.499665 |


## Confusion matrix (at threshold=0.499665)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |         10870    |          4505.74 |
| Labeled as 1 |          5524.67 |          9853.98 |

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

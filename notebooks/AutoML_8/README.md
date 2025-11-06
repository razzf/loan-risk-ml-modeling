# AutoML Leaderboard

| Best model   | name                                               | model_type    | metric_type   |   metric_value |   train_time |
|:-------------|:---------------------------------------------------|:--------------|:--------------|---------------:|-------------:|
|              | [1_DecisionTree](1_DecisionTree/README.md)         | Decision Tree | auc           |       0.697471 |       204.82 |
|              | [2_DecisionTree](2_DecisionTree/README.md)         | Decision Tree | auc           |       0.655017 |       166    |
|              | [3_DecisionTree](3_DecisionTree/README.md)         | Decision Tree | auc           |       0.655017 |       165.08 |
|              | [4_Default_LightGBM](4_Default_LightGBM/README.md) | LightGBM      | auc           |       0.782816 |       310.15 |
|              | [5_Default_Xgboost](5_Default_Xgboost/README.md)   | Xgboost       | auc           |       0.780709 |       257.61 |
|              | [15_LightGBM](15_LightGBM/README.md)               | LightGBM      | auc           |       0.781294 |       189.73 |
|              | [6_Xgboost](6_Xgboost/README.md)                   | Xgboost       | auc           |       0.779222 |       213.25 |
|              | [24_CatBoost](24_CatBoost/README.md)               | CatBoost      | auc           |       0.78206  |       498.67 |
|              | [33_RandomForest](33_RandomForest/README.md)       | Random Forest | auc           |       0.736117 |      2742.55 |
| **the best** | [Ensemble](Ensemble/README.md)                     | Ensemble      | auc           |       0.786388 |         5.53 |

### AutoML Performance
![AutoML Performance](ldb_performance.png)

### AutoML Performance Boxplot
![AutoML Performance Boxplot](ldb_performance_boxplot.png)

### Features Importance (Original Scale)
![features importance across models](features_heatmap.png)



### Scaled Features Importance (MinMax per Model)
![scaled features importance across models](features_heatmap_scaled.png)



### Spearman Correlation of Models
![models spearman correlation](correlation_heatmap.png)


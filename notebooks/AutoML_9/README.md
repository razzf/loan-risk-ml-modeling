# AutoML Leaderboard

| Best model   | name                                               | model_type    | metric_type   |   metric_value |   train_time |
|:-------------|:---------------------------------------------------|:--------------|:--------------|---------------:|-------------:|
|              | [1_DecisionTree](1_DecisionTree/README.md)         | Decision Tree | auc           |       0.706358 |       249.1  |
|              | [2_DecisionTree](2_DecisionTree/README.md)         | Decision Tree | auc           |       0.666149 |       226.81 |
|              | [3_DecisionTree](3_DecisionTree/README.md)         | Decision Tree | auc           |       0.666149 |       176.08 |
|              | [4_Default_LightGBM](4_Default_LightGBM/README.md) | LightGBM      | auc           |       0.78643  |       242.09 |
|              | [5_Default_Xgboost](5_Default_Xgboost/README.md)   | Xgboost       | auc           |       0.783768 |       255.24 |
|              | [15_LightGBM](15_LightGBM/README.md)               | LightGBM      | auc           |       0.785383 |       166.13 |
|              | [6_Xgboost](6_Xgboost/README.md)                   | Xgboost       | auc           |       0.781455 |       224.54 |
|              | [24_CatBoost](24_CatBoost/README.md)               | CatBoost      | auc           |       0.785909 |       366.46 |
|              | [33_RandomForest](33_RandomForest/README.md)       | Random Forest | auc           |       0.742849 |      2734.03 |
| **the best** | [Ensemble](Ensemble/README.md)                     | Ensemble      | auc           |       0.789519 |         6.71 |

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


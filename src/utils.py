import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr, ttest_ind, norm
from statsmodels.stats.proportion import proportion_confint
import plotly.graph_objects as go
from catboost import Pool, cv
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def plot_interactive_shap(data, response_dict):
    """
    Generates an interactive SHAP force plot from a response dictionary and original data.

    Args:
        data (dict): A dictionary containing the original feature values (e.g., data_n18).
        response_dict (dict): A dictionary containing 'prediction' and 'shap_values'.
    """
    if not isinstance(response_dict, dict) or "shap_values" not in response_dict:
        print("Error: The input 'response_dict' is not valid.")
        return

    if not isinstance(data, dict):
        print("Error: The input 'data' is not a valid dictionary.")
        return

    prediction = response_dict["prediction"]
    shap_values_dict = response_dict["shap_values"]

    base_value = prediction - sum(shap_values_dict.values())

    shap.initjs()

    explanation = shap.Explanation(
        values=np.array(list(shap_values_dict.values())),
        base_values=base_value,
        data=np.array(list(data.values())),
        feature_names=list(shap_values_dict.keys()),
    )

    shap.force_plot(explanation)


def plot_shap_values(response_dict):
    """
    Generates a horizontal bar chart of SHAP values from a response dictionary.

    Args:
        response_dict (dict): A dictionary containing 'shap_values'.
    """
    if not isinstance(response_dict, dict) or "shap_values" not in response_dict:
        print("Error: The input is not a valid response dictionary.")
        return

    shap_values = response_dict["shap_values"]

    df = pd.DataFrame(shap_values.items(), columns=["feature", "shap_value"])
    df_sorted = df.sort_values(by="shap_value", ascending=False)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_facecolor("white")
    ax.grid(False)

    colors = ["blue" if x < 0 else "red" for x in df_sorted["shap_value"]]

    ax.barh(df_sorted["feature"], df_sorted["shap_value"], color=colors)

    ax.set_title("All Feature Contributions to Model Prediction", fontsize=16)
    ax.set_xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
    ax.set_ylabel("")

    max_abs_shap = df_sorted["shap_value"].abs().max()
    x_buffer = max_abs_shap * 0.2
    ax.set_xlim(-(max_abs_shap + x_buffer), (max_abs_shap + x_buffer))

    text_offset = max_abs_shap * 0.015

    for index, value in enumerate(df_sorted["shap_value"]):
        if value > 0:
            ax.text(
                value + text_offset,
                index,
                f"{value:.4f}",
                va="center",
                ha="left",
                color="grey",
                fontsize=9,
            )
        else:
            ax.text(
                value - text_offset,
                index,
                f"{value:.4f}",
                va="center",
                ha="right",
                color="grey",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()


def create_feature_subset(model, X_data, n_features):
    """
    Creates a new dataset with only the top N most important features.

    Args:
        model: The trained CatBoost model.
        X_data (pd.DataFrame): The original feature data.
        n_features (int): The number of top features to keep.

    Returns:
        pd.DataFrame: The new X dataset containing only the top features.
    """
    importance_scores = model.get_feature_importance(type="PredictionValuesChange")
    all_features = X_data.columns.tolist()

    importance_series = pd.Series(importance_scores, index=all_features)
    top_features = (
        importance_series.sort_values(ascending=False).head(n_features).index.tolist()
    )

    X_subset = X_data[top_features]

    print(f"Created a subset with {n_features} features.")
    return X_subset


def run_cv_on_subset(X_data_subset, y_data, subset_name, categorical_features):
    """
    Runs cross-validation on a given feature subset.

    Args:
        X_data_subset (pd.DataFrame): The feature subset to use for training.
        y_data (pd.Series): The target variable.
        subset_name (str): A descriptive name for the subset.
    """
    print(f"\n--- Running Cross-Validation for Subset: {subset_name} ---")

    cat_features_subset = [
        col for col in X_data_subset.columns if col in categorical_features
    ]

    train_pool_subset = Pool(
        data=X_data_subset, label=y_data, cat_features=cat_features_subset
    )

    catboost_params = {
        "iterations": 1000,
        "learning_rate": 0.1,
        "early_stopping_rounds": 50,
        "eval_metric": "AUC",
        "verbose": 0,
        "auto_class_weights": "Balanced",
        "random_state": 42,
        "loss_function": "Logloss",
    }

    cv_results = cv(train_pool_subset, params=catboost_params, fold_count=5, plot=True)

    print(f"\nResults for {subset_name}:")
    print(f"Mean AUC: {cv_results['test-AUC-mean'].iloc[-1]:.4f}")
    print(f"Std AUC: {cv_results['test-AUC-std'].iloc[-1]:.4f}")


def get_confusion_matrix_df(y_true, y_pred):
    """Returns a confusion matrix as a labeled Pandas DataFrame."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        conf_matrix,
        index=["Actual Negative (0)", "Actual Positive (1)"],
        columns=["Predicted Negative (0)", "Predicted Positive (1)"],
    )


def plot_feature_selection_elbow_improved(
    feature_selection_summary, initial_features_count=None
):
    """
    Analyzes and plots the feature selection loss curve to find the elbow point
    using a more robust method that identifies where the curve flattens.

    This function takes a summary dictionary from a feature selection process,
    calculates the "elbow point" on the loss curve by finding the point of
    maximum curvature (inflection point), and generates an interactive Plotly graph.

    Args:
        feature_selection_summary (dict): A dictionary containing the results of
                                          a feature selection process, expected to
                                          have the following keys:
                                          - 'loss_graph': a dictionary with a key
                                            'loss_values' (list or array of floats).
                                          - 'eliminated_features': a list of
                                            eliminated feature indices.
        initial_features_count (int, optional): The total number of features at the
                                                start of the selection process. If None,
                                                it will be inferred from the summary.

    Returns:
        tuple: A tuple containing the number of features remaining at the elbow point
               and the corresponding loss value.
    """
    loss_values = np.array(feature_selection_summary["loss_graph"]["loss_values"])
    eliminated_features_indices = feature_selection_summary["eliminated_features"]

    if initial_features_count is None:
        print(
            "Warning: `initial_features_count` was not provided. Inferring from summary."
        )
        total_features = len(eliminated_features_indices) + 1
    else:
        total_features = initial_features_count

    num_features_removed = np.arange(len(loss_values)) + 1
    num_features_remaining = total_features - num_features_removed
    first_derivative = np.gradient(loss_values, num_features_remaining)

    second_derivative = np.gradient(first_derivative, num_features_remaining)

    elbow_point_index = np.argmax(second_derivative)

    elbow_num_features_remaining = num_features_remaining[elbow_point_index]
    elbow_loss_value = loss_values[elbow_point_index]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=num_features_remaining,
            y=loss_values,
            mode="lines+markers",
            name="Validation Loss",
            hoverinfo="text",
            text=[
                f"Removed: {i} features<br>Remaining: {rem_count} features<br>Loss: {loss:.4f}"
                for i, rem_count, loss in zip(
                    num_features_removed, num_features_remaining, loss_values
                )
            ],
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[elbow_num_features_remaining],
            y=[elbow_loss_value],
            mode="markers",
            marker=dict(color="orange", size=10, symbol="circle"),
            name="Elbow Point",
            hoverinfo="text",
            text=[f"Elbow Point at {elbow_num_features_remaining} features remaining"],
        )
    )

    fig.update_layout(
        title="Feature Selection Loss Curve (Inflection Point Method)",
        xaxis_title="Number of Remaining Features",
        yaxis_title="Validation Loss",
        showlegend=True,
        template="plotly_white",
        hovermode="closest",
        xaxis=dict(autorange="reversed"),
    )
    fig.show()

    print(
        f"The number of remaining features at the elbow point is: {elbow_num_features_remaining}"
    )
    print(f"The corresponding loss value is: {elbow_loss_value:.4f}")

    return elbow_num_features_remaining, elbow_loss_value


def create_feature_subset_excluding_list(X_data, list_of_cols_to_exclude):
    """
    Creates a new dataset by excluding a specific list of columns.

    Args:
        X_data (pd.DataFrame): The feature data.
        list_of_cols_to_exclude (list): A list of column names to exclude.

    Returns:
        pd.DataFrame: A new DataFrame excluding the specified columns.
    """
    features_to_keep = [
        col for col in X_data.columns if col not in list_of_cols_to_exclude
    ]

    X_subset = X_data[features_to_keep]

    print(
        f"Created a subset with {len(features_to_keep)} features, excluding: {list_of_cols_to_exclude}"
    )
    return X_subset


def adversarial_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    identifier_columns: str,
    test_size: float = 0.3,
    random_state: int = 42,
    n_estimators: int = 100,
):
    """
    Performs adversarial validation to assess the similarity between training and test datasets.
    This function trains a classifier to distinguish between samples from the training and test sets
    using their feature distributions. The resulting AUC score indicates how easily the classifier
    can differentiate between the two datasets; a high AUC suggests distributional differences.
    Args:
        train (pd.DataFrame): Training dataset containing features, target, and identifier columns.
        test (pd.DataFrame): Test dataset containing features and identifier columns.
        target_column (str): Name of the target column in the training dataset.
        identifier_columns (str): Name of the identifier column(s) to exclude from features.
        test_size (float, optional): Proportion of data to use for validation split. Default is 0.3.
        random_state (int, optional): Random seed for reproducibility. Default is 42.
        n_estimators (int, optional): Number of boosting rounds for the classifier. Default is 100.
    Returns:
        clf (LGBMClassifier): Trained LightGBM classifier distinguishing train/test samples.
        auc (float): ROC AUC score measuring separability between train and test sets.
        feature_list (List[str]): List of feature column names used for classification.
    Prints:
        Adversarial validation AUC score.
    """
    train_features = train.drop(columns=[target_column, identifier_columns]).copy()
    train_features["source"] = 1

    test_features = test.drop(columns=[identifier_columns]).copy()
    test_features["source"] = 0

    common_cols = list(set(train_features.columns) & set(test_features.columns))
    full_data = pd.concat(
        [train_features[common_cols], test_features[common_cols]],
        axis=0,
        ignore_index=True,
    )

    full_data = full_data.select_dtypes(include=[np.number])

    X = full_data.drop(columns=["source"])
    y = full_data["source"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    clf = LGBMClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    print(f"Adversarial validation AUC: {auc:.4f}")
    return clf, auc, X.columns.tolist()


def plot_partial_residuals(model, df, predictor):
    """
    Plot partial residuals to check linearity assumption for logistic regression.

    Parameters
    ----------
    model : statsmodels LogitResults object
        The fitted logistic regression model.
    df : pandas.DataFrame
        Original dataframe used for fitting.
    predictor : str
        Predictor variable to plot.
    """
    params = model.params.copy()
    if "const" in params:
        intercept = params.pop("const")
    else:
        intercept = 0
    linear_pred = df[predictor] * params[predictor]

    predicted = model.predict()
    residuals = df[model.model.endog_names] - predicted
    partial_residuals = linear_pred + residuals

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df[predictor], y=partial_residuals, alpha=0.6)
    sns.regplot(
        x=df[predictor], y=partial_residuals, lowess=True, scatter=False, color="red"
    )
    plt.xlabel(predictor)
    plt.ylabel("Partial Residuals")
    plt.title(f"Partial Residual Plot for {predictor}")
    plt.show()


def box_tidwell_test(df, y, predictors):
    """
    Performs Box-Tidwell test for linearity of continuous predictors.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the variables.
    y : str
        Name of the binary outcome variable.
    predictors : list of str
        List of continuous predictor variable names to test.

    Returns
    -------
    pd.DataFrame
        Summary table of coefficients and p-values for interaction terms.
    """
    df = df.copy()
    df = df.loc[:, [y] + predictors].dropna()

    for var in predictors:
        df[f"{var}_log_interaction"] = df[var] * np.log(df[var] + 1e-6)

    X = df[predictors + [f"{var}_log_interaction" for var in predictors]]
    X = sm.add_constant(X)
    model = sm.Logit(df[y], X).fit(disp=0)

    results = []
    for var in predictors:
        interaction_coef = model.params[f"{var}_log_interaction"]
        interaction_p = model.pvalues[f"{var}_log_interaction"]
        results.append(
            {
                "Predictor": var,
                "Interaction_Coefficient": interaction_coef,
                "p-value": interaction_p,
            }
        )

    return pd.DataFrame(results)


def plot_coef_with_ci(coef, lower_ci, upper_ci, feature_name="Feature", title=None):
    """
    Plots a single coefficient with 95% confidence interval as a vertical bar with error bars.

    Parameters
    ----------
    coef : float
        The coefficient estimate.
    lower_ci : float
        The lower bound of the 95% confidence interval.
    upper_ci : float
        The upper bound of the 95% confidence interval.
    feature_name : str, optional
        Name of the feature being plotted.
    title : str, optional
        Title for the plot.

    Returns
    -------
    None
        Displays the plot.
    """
    err_lower = coef - lower_ci
    err_upper = upper_ci - coef

    yerr = [[err_lower], [err_upper]]

    if coef > 0:
        direction = f"{feature_name} positive effect"
        color = "mediumseagreen"
    elif coef < 0:
        direction = f"{feature_name} negative effect"
        color = "salmon"
    else:
        direction = f"{feature_name} no effect"
        color = "gray"

    fig, ax = plt.subplots(figsize=(3, 3))

    ax.bar(
        [feature_name],
        [coef],
        color=color,
        yerr=yerr,
        capsize=8,
    )

    y_min = min(lower_ci - 0.1, coef - err_lower - 0.1, 0.01)
    y_max = max(upper_ci + 0.1, coef + err_upper + 0.1, 0.01)
    ax.set_ylim(y_min, y_max)

    offset = -0.05 * (y_max - y_min)

    ax.axhline(y=0, linestyle=":", color="gray", linewidth=1.5)
    ax.text(0.45, -0.02, "0", va="bottom", ha="left", fontsize=12, color="gray")
    ax.set_title(title or f"Coefficient for {feature_name} with 95% CI")
    ax.text(
        0,
        (
            coef + err_upper * 0.5 - offset
            if coef >= 0
            else coef - err_lower * 0.5 + offset
        ),
        f"{direction}\n{coef:.3f}\n[{lower_ci:.3f}, {upper_ci:.3f}]",
        ha="center",
        va="bottom" if coef >= 0 else "top",
        fontsize=10,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def plot_proportion_with_ci(ct, label_prefix="Group"):
    """
    Plots the proportion of successes with 95% confidence intervals for each group in a contingency table.
    Parameters
    ----------
    ct : pandas.DataFrame or pandas.Series
        Contingency table with group labels as index and counts of outcomes (e.g., 0/1) as columns.
        Assumes that the column '1' represents the count of successes for each group.
    label_prefix : str, optional
        Prefix for group labels in the plot (default is "Group").
    Returns
    -------
    None
        Displays a bar plot with proportions, confidence intervals, and annotations for each group.
    Notes
    -----
    - Uses the Wilson score interval for confidence intervals.
    - Bars are annotated with percentage and confidence interval.
    - The y-axis is hidden for a cleaner presentation.
    """
    props = []
    lower = []
    upper = []
    labels = []
    annotations = []

    for status in ct.index:
        success = ct.loc[status, 1]
        total = ct.loc[status].sum()
        prop = success / total
        ci_low, ci_upp = proportion_confint(success, total, alpha=0.05, method="wilson")

        props.append(prop)
        lower.append(prop - ci_low)
        upper.append(ci_upp - prop)
        labels.append(f"{label_prefix}={status}")

        percent = f"{prop*100:.1f}%"
        ci_text = f"[{ci_low*100:.2f}%, {ci_upp*100:.2f}%]"
        annotations.append(f"{percent}\n{ci_text}")

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(
        labels,
        props,
        yerr=[lower, upper],
        capsize=5,
        color=["skyblue", "salmon"][: len(labels)],
    )

    for bar, annotation in zip(bars, annotations):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            annotation,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("")
    ax.set_title(f"Default rate by {label_prefix} (with 95% CI)")
    ax.set_ylim(0, 0.5)

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.grid(False)
    plt.show()


def plot_difference_in_proportion(
    ct, feature_name="Group", title="Difference in Default Proportions with 95% CI"
):
    """
    Plots the absolute difference in proportions between two groups with a 95% confidence interval.
    Always shows the bar as positive; direction is indicated in the text label.

    Parameters
    ----------
    ct : pandas.DataFrame
        A 2x2 contingency table where rows represent groups (e.g., "F" and "M") and columns represent outcomes.
        Assumes ct.loc[1, 1] and ct.loc[0, 1] are the counts of 'success' for groups 1 and 0, respectively.
    feature_name : str, optional
        Name of the grouping variable. Will be used in the label like "feature_name = Group1 vs Group2".
    title : str, optional
        The title of the plot. Default is "Difference in Default Proportions with 95% CI".
    Returns
    -------
    None
        Displays a bar plot showing the absolute difference in proportions and its 95% confidence interval.
    """
    group_labels = ct.index.tolist()

    success1 = ct.loc[group_labels[0], 1]
    total1 = ct.loc[group_labels[0]].sum()
    p1 = success1 / total1

    success2 = ct.loc[group_labels[1], 1]
    total2 = ct.loc[group_labels[1]].sum()
    p2 = success2 / total2

    diff = p1 - p2
    abs_diff = abs(diff)
    se_diff = np.sqrt(p1 * (1 - p1) / total1 + p2 * (1 - p2) / total2)
    z = norm.ppf(0.975)

    ci_low = abs_diff - z * se_diff
    ci_upp = abs_diff + z * se_diff

    if diff > 0:
        direction = f"{group_labels[0]} higher"
    elif diff < 0:
        direction = f"{group_labels[1]} higher"
    else:
        direction = "no difference"

    label = f"{feature_name}={direction}"

    fig, ax = plt.subplots(figsize=(2.2, 2.5))

    ax.bar(
        ["Difference in Proportion"],
        [abs_diff],
        color="mediumseagreen",
        yerr=[[abs_diff - ci_low], [ci_upp - abs_diff]],
        capsize=8,
    )

    ax.set_ylim(0, max(0.2, ci_upp + 0.02))
    ax.set_title(title)
    ax.text(
        0,
        abs_diff + 0.015,
        f"{label}\n{abs_diff*100:.1f}%\n[{ci_low*100:.2f}%, {ci_upp*100:.2f}%]",
        ha="center",
        fontsize=10,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def plot_feature_distribution(
    df,
    feature_name,
    target_name=None,
    cat_threshold=10,
    max_categories=15,
    annotate=False,
    show_other=True,
    stacked_target_split=False,
    x_axis_rotation=0,
):
    """
    Plot the distribution of a feature, optionally split by a binary target.

    If stacked_target_split=True, a stacked bar chart is used for binary target.
    """
    feature = df[feature_name]
    if target_name:
        data = df[[feature_name, target_name]].dropna()
        feature = data[feature_name]
    else:
        data = df[[feature_name]].dropna()
        feature = data[feature_name]

    unique_vals = feature.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(feature)

    plt.figure(figsize=(5, 3))

    if (not is_numeric) or (unique_vals <= cat_threshold):
        top_categories = feature.value_counts().nlargest(max_categories).index
        if show_other:
            data[feature_name] = data[feature_name].where(
                data[feature_name].isin(top_categories), other="Other"
            )
        else:
            data = data[data[feature_name].isin(top_categories)]

        if target_name:
            if stacked_target_split:
                counts = (
                    pd.crosstab(
                        data[feature_name], data[target_name], normalize="index"
                    )
                    * 100
                )
                counts = counts[[0, 1]] if 0 in counts and 1 in counts else counts
                counts.plot(
                    kind="bar",
                    stacked=True,
                    color=["#a6bddb", "#ef8a62"],
                    figsize=(6, 4),
                )
                plt.ylabel("Percentage (%)")
                plt.title(f"Distribution of {target_name} by {feature_name}")
                plt.legend(title=target_name)
                plt.xticks(rotation=x_axis_rotation)
            else:
                means = data.groupby(feature_name)[target_name].mean().mul(100)
                means = means.sort_values(ascending=False)
                ax = sns.barplot(x=means.index, y=means.values)
                plt.ylabel("Percentage (%)")
                plt.title(f"Percentage of {target_name}=1 by {feature_name}")
                plt.xticks(rotation=x_axis_rotation)
                if annotate:
                    for i, val in enumerate(means.values):
                        ax.text(i, val + 1, f"{val:.1f}%", ha="center", va="bottom")
        else:
            value_counts = (
                feature.value_counts(normalize=True)
                .mul(100)
                .sort_values(ascending=False)
            )
            if len(value_counts) > max_categories:
                top = value_counts.iloc[:max_categories]
                if show_other:
                    other = value_counts.iloc[max_categories:].sum()
                    value_counts = pd.concat([top, pd.Series({"Other": other})])
                else:
                    value_counts = top

            ax = sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.ylabel("Percentage (%)")
            plt.title(f"Distribution of {feature_name} (percent of non-missing)")
            plt.xticks(rotation=x_axis_rotation)
            if annotate:
                for i, val in enumerate(value_counts.values):
                    ax.text(i, val + 1, f"{val:.1f}%", ha="center", va="bottom")

    else:
        if target_name:
            sns.boxplot(x=target_name, y=feature_name, data=data)
            plt.title(f"{feature_name} by {target_name}")
        else:
            sns.histplot(feature, kde=False, bins=30, stat="percent")
            plt.ylabel("Percentage (%)")
            plt.title(f"Distribution of {feature_name}")

    plt.tight_layout()
    plt.show()


def summarize_categorical_feature(df, feature_name):
    """
    Returns a summary table of counts and percentages for a categorical feature.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the feature.
    - feature_name (str): The name of the categorical feature.

    Returns:
    - pd.DataFrame: A table with counts and share (%) for each unique value.
    """
    return pd.concat(
        [
            df[feature_name].value_counts(dropna=False),
            df[feature_name]
            .value_counts(normalize=True, dropna=False)
            .mul(100)
            .round(2),
        ],
        axis=1,
        keys=["Count", "Share (%)"],
    )


def t_test_difference(df, feature, group_column, group_value_1, group_value_2):
    """
    Performs an independent t-test to compare the means of a specified feature between two groups.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    feature : str
        The name of the column (feature) for which the t-test is conducted.
    group_column : str
        The column name containing the group labels.
    group_value_1 : str or numeric
        The value in the `group_column` that represents the first group for comparison.
    group_value_2 : str or numeric
        The value in the `group_column` that represents the second group for comparison.

    Returns:
    -------
    None
        Prints the t-statistic, p-value, difference in means, and the 95% confidence interval
        for the difference in means between the two groups.

    Notes:
    -----
    The function performs an independent two-sample t-test assuming unequal variances.
    It also calculates the 95% confidence interval for the difference in means.
    """
    group_1 = df[df[group_column] == group_value_1][feature]
    group_2 = df[df[group_column] == group_value_2][feature]

    t_stat, p_value = ttest_ind(group_1, group_2, nan_policy="omit")

    mean_group_1 = group_1.mean()
    mean_group_2 = group_2.mean()

    std_group_1 = group_1.std(ddof=1)
    std_group_2 = group_2.std(ddof=1)

    n_group_1 = len(group_1)
    n_group_2 = len(group_2)

    se_diff = np.sqrt((std_group_1**2 / n_group_1) + (std_group_2**2 / n_group_2))

    z = 1.96
    lower_bound = (mean_group_1 - mean_group_2) - z * se_diff
    upper_bound = (mean_group_1 - mean_group_2) + z * se_diff

    difference_in_means = mean_group_1 - mean_group_2

    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(f"Difference in means: {difference_in_means:.3f}")
    print(
        f"95% Confidence Interval for the difference in means: ({lower_bound:.3f}, {upper_bound:.3f})"
    )


def custom_correlation_with_pvalues(df, feature_types):
    """
    Computes pairwise correlation coefficients and their corresponding p-values between all columns in a given DataFrame.
    The method used for calculating correlation depends on the data types of the two variables being compared.

    Args:
    df (pd.DataFrame): A DataFrame containing the data for which the correlations are calculated. The columns in this DataFrame
                       should correspond to variables that are categorized by type (continuous, binary, categorical, ordinal).
    feature_types (dict): A dictionary where keys are column names and values are the data types of those columns.
                           The types should be one of: "continuous", "binary", "categorical", "ordinal".

    Returns:
    tuple: A tuple containing two DataFrames:
           - corr_matrix (pd.DataFrame): A matrix of correlation coefficients for each pair of variables.
           - pval_matrix (pd.DataFrame): A matrix of p-values corresponding to the correlation coefficients.

    Correlation methods used:
    - Pearson correlation: Used for continuous vs continuous variables and binary vs continuous variables.
    - Cramér's V (chi-square test): Used for binary vs binary or categorical vs categorical variables.
    - Eta-squared (ANOVA): Used for categorical vs continuous variables, calculated as the proportion of variance explained.
    - Spearman rank correlation: Used for ordinal vs ordinal or ordinal vs. continuous variables.

    Notes:
    - Missing values are dropped pairwise for each correlation calculation.
    - Categorical variables are converted to numeric codes for the correlation calculation.
    - The function returns NaN for pairs of variables that don't have enough data or if no appropriate correlation method exists.
    """
    corr_matrix = pd.DataFrame(
        np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns
    )
    pval_matrix = pd.DataFrame(
        np.zeros((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns
    )

    df_encoded = df.copy()

    for col, col_type in feature_types.items():
        if col in df_encoded.columns:
            if col_type == "binary":
                unique_vals = df_encoded[col].dropna().unique()
                if len(unique_vals) == 2:
                    mapping = {unique_vals.min(): 0, unique_vals.max(): 1}
                    df_encoded[col] = df_encoded[col].map(mapping)
            elif col_type == "categorical":
                df_encoded[col] = df_encoded[col].astype("category").cat.codes

    for col1 in df_encoded.columns:
        for col2 in df_encoded.columns:
            if col1 == col2:
                corr = np.nan
                p_value = np.nan
            else:
                type1 = feature_types[col1]
                type2 = feature_types[col2]

                if type1 == "continuous" and type2 == "continuous":
                    valid_data = df_encoded[[col1, col2]].dropna()
                    if len(valid_data) < 2:
                        corr, p_value = np.nan, np.nan
                    else:
                        corr, p_value = pearsonr(valid_data[col1], valid_data[col2])

                elif (type1 in ["binary", "categorical"]) and (
                    type2 in ["binary", "categorical"]
                ):
                    contingency_table = pd.crosstab(df_encoded[col1], df_encoded[col2])
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    n = contingency_table.sum().sum()
                    min_dim = min(contingency_table.shape) - 1
                    if min_dim > 0:
                        corr = np.sqrt(chi2 / (n * min_dim))
                    else:
                        corr = np.nan

                elif (type1 == "binary" and type2 == "continuous") or (
                    type1 == "continuous" and type2 == "binary"
                ):
                    binary_col, cont_col = (
                        (col1, col2) if type1 == "binary" else (col2, col1)
                    )
                    valid_data = df_encoded[[binary_col, cont_col]].dropna()
                    if len(valid_data) < 2:
                        corr, p_value = np.nan, np.nan
                    else:
                        corr, p_value = pearsonr(
                            valid_data[binary_col], valid_data[cont_col]
                        )

                elif (type1 == "categorical" and type2 == "continuous") or (
                    type1 == "continuous" and type2 == "categorical"
                ):
                    cat_col, cont_col = (
                        (col1, col2) if type1 == "categorical" else (col2, col1)
                    )

                    valid_data = df_encoded[[cat_col, cont_col]].dropna()

                    groups = [
                        valid_data[cont_col][valid_data[cat_col] == cat_val]
                        for cat_val in valid_data[cat_col].unique()
                    ]

                    groups = [g for g in groups if len(g) > 0]

                    if len(groups) < 2 or valid_data[cont_col].var() == 0:
                        corr, p_value = np.nan, np.nan
                    else:
                        f_stat, p_value = f_oneway(*groups)
                        ss_between = sum(
                            len(g) * (g.mean() - valid_data[cont_col].mean()) ** 2
                            for g in groups
                        )
                        ss_total = sum(
                            (valid_data[cont_col] - valid_data[cont_col].mean()) ** 2
                        )
                        corr = ss_between / ss_total if ss_total > 0 else np.nan
                elif (
                    (type1 == "ordinal" or type2 == "ordinal")
                    or (type1 == "ordinal" and type2 == "continuous")
                    or (type1 == "continuous" and type2 == "ordinal")
                ):
                    valid_data = df_encoded[[col1, col2]].dropna()
                    if len(valid_data) < 2:
                        corr, p_value = np.nan, np.nan
                    else:
                        corr, p_value = spearmanr(valid_data[col1], valid_data[col2])

            corr_matrix.loc[col1, col2] = corr
            pval_matrix.loc[col1, col2] = p_value

    return corr_matrix, pval_matrix


def box_hist_plot(df, feature_name, ylim_min, ylim_max, xticks=None):
    """
    Creates a box plot and a histogram side by side for a specified feature in the DataFrame.

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    feature_name : str
        The name of the feature (column) in the DataFrame to plot.
    ylim_min : float
        The minimum value for the y-axis limits.
    ylim_max : float
        The maximum value for the y-axis limits.
    xticks : list or None, optional, default None
        A list of xticks for the histogram. If None, the default xticks will be used.

    Returns:
    -------
    None
        Displays the box plot and histogram side by side.

    Notes:
    -----
    - The box plot shows the distribution of the feature with its quartiles and potential outliers.
    - The histogram displays the distribution of the feature with kernel density estimation (KDE) overlaid.
    - Both plots share the same y-axis to facilitate comparison of the distribution.
    """
    f, (ax_box, ax_hist) = plt.subplots(
        1, 2, sharey=True, gridspec_kw={"width_ratios": (0.50, 0.25)}, figsize=(5, 5)
    )

    sns.boxplot(df[feature_name], ax=ax_box)
    sns.histplot(data=df, y=feature_name, ax=ax_hist, bins=20, kde=True)

    ax_box.set(xticks=[], xlabel="")
    ax_box.set_ylabel(feature_name, labelpad=10)
    ax_box.set_title(f"Box plot of $\\it{{{feature_name}}}$ (incl. histogram)", pad=35)
    plt.ylim(ylim_min, ylim_max)
    plt.ylim(ylim_min, ylim_max)
    if xticks is not None:
        ax_hist.set(xticks=xticks, xlabel="Respondents")
    else:
        ax_hist.set(xlabel="Respondents")
    sns.despine(ax=ax_box)
    sns.despine(ax=ax_hist, right=True, top=True)
    plt.tight_layout()
    plt.show()


def plot_features(df, feature_types, figsize=(8, 6), fixed_feature2=None):
    """
    Plots various types of feature comparisons based on their types (continuous, binary, categorical, ordinal).

    Parameters:
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    feature_types : dict
        A dictionary where keys are feature names and values are their corresponding types ("continuous", "binary", "categorical", "ordinal").
    figsize : tuple, optional, default (8, 6)
        The size of the plot.
    fixed_feature2 : str or None, optional, default None
        A specific feature to compare with all other features. If None, comparisons will be made between all pairs of features.

    Returns:
    -------
    None
        Displays the plots for each feature comparison.

    Notes:
    -----
    The function creates different types of plots depending on the types of features being compared. These include:
    - Scatter plot with regression for two continuous features
    - Stacked bar plot for two binary or categorical features
    - Violin plot for comparisons involving continuous and binary, ordinal, or categorical features.
    """
    plotted_pairs = set()
    feature2_list = [fixed_feature2] if fixed_feature2 else df.columns

    for feature1 in df.columns:
        for feature2 in feature2_list:
            if (
                feature1 == feature2
                or (pair := tuple(sorted([feature1, feature2]))) in plotted_pairs
            ):
                continue
            plotted_pairs.add(pair)

            type1, type2 = feature_types[feature1], feature_types[feature2]
            fig, ax = plt.subplots(figsize=figsize)

            if type1 == "continuous" and type2 == "continuous":
                sns.scatterplot(data=df, x=feature1, y=feature2, ax=ax)
                sns.regplot(
                    data=df,
                    x=feature1,
                    y=feature2,
                    ax=ax,
                    scatter=False,
                    color="red",
                    line_kws={"linewidth": 2},
                )
                title = f"Scatter with Regression: {feature1} vs {feature2}"
            elif type1 == type2 == "binary":
                pd.crosstab(df[feature1], df[feature2]).apply(
                    lambda x: x / x.sum(), axis=1
                ).plot(
                    kind="bar", stacked=True, ax=ax, colormap="coolwarm", legend=True
                )
                title = f"Stacked Bar: {feature1} vs {feature2}"
            elif "continuous" in (type1, type2) and "binary" in (type1, type2):
                sns.violinplot(
                    data=df,
                    x=feature1 if type1 == "binary" else feature2,
                    y=feature2 if type1 == "binary" else feature1,
                    ax=ax,
                )
                title = f"Violin: {feature1} vs {feature2}"
            elif "continuous" in (type1, type2) and "ordinal" in (type1, type2):
                sns.violinplot(
                    data=df,
                    x=feature1 if type1 != "continuous" else feature2,
                    y=feature2 if type1 != "continuous" else feature1,
                    ax=ax,
                )
                title = f"Violin: {feature1} vs {feature2}"
            elif "categorical" in (type1, type2) and "continuous" in (type1, type2):
                sns.violinplot(
                    data=df,
                    x=feature1 if type1 == "categorical" else feature2,
                    y=feature2 if type1 == "categorical" else feature1,
                    ax=ax,
                )
                title = f"Violin: {feature1} vs {feature2}"
            elif "categorical" in (type1, type2) or "ordinal" in (type1, type2):
                pd.crosstab(df[feature1], df[feature2]).apply(
                    lambda x: x / x.sum(), axis=1
                ).plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
                title = f"Stacked Bar: {feature1} vs {feature2}"

            ax.set_title(title, fontsize=10)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=35, ha="right", fontsize=10
            )

            if ax.get_legend():
                ax.get_legend().set_title(feature2)
                ax.get_legend().set_bbox_to_anchor((1.02, 0.5))
                ax.get_legend().set_frame_on(False)
                for text in ax.get_legend().get_texts():
                    text.set_fontsize(9)

            plt.tight_layout()
            plt.show()


def get_confusion_matrix_df(y_true, y_pred):
    """Returns a confusion matrix as a labeled Pandas DataFrame."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        conf_matrix,
        index=["Actual Negative (0)", "Actual Positive (1)"],
        columns=["Predicted Negative (0)", "Predicted Positive (1)"],
    )


def plot_pr_auc(model, X_test, y_test):
    """
    Function to compute and plot the Precision-Recall curve for a given model.

    Parameters:
    model : trained classifier (must have predict_proba method)
    X_test : test feature set
    y_test : true labels for test data
    """
    y_scores = model.predict_proba(X_test)

    pr_auc_pos = average_precision_score(y_test, y_scores[:, 1])
    pr_auc_neg = average_precision_score(1 - y_test, y_scores[:, 0])

    print(f"PR AUC Score (Class 1): {pr_auc_pos:.4f}")
    print(f"PR AUC Score (Class 0): {pr_auc_neg:.4f}")

    precision_pos, recall_pos, _ = precision_recall_curve(y_test, y_scores[:, 1])
    precision_neg, recall_neg, _ = precision_recall_curve(1 - y_test, y_scores[:, 0])

    plt.figure(figsize=(4, 3))
    plt.plot(
        recall_pos,
        precision_pos,
        marker=".",
        label=f"Class 1 (PR AUC = {pr_auc_pos:.4f})",
        color="blue",
    )
    plt.plot(
        recall_neg,
        precision_neg,
        marker=".",
        label=f"Class 0 (PR AUC = {pr_auc_neg:.4f})",
        color="red",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()


def plot_roc_auc(model, X_test, y_test):
    """
    Function to compute and plot the ROC curve for a given model.

    Parameters:
    model : trained classifier (must have predict_proba method)
    X_test : test feature set
    y_test : true labels for test data
    """
    y_scores = model.predict_proba(X_test)

    roc_auc_pos = roc_auc_score(y_test, y_scores[:, 1])
    roc_auc_neg = roc_auc_score(1 - y_test, y_scores[:, 0])

    print(f"ROC AUC Score (Class 1): {roc_auc_pos:.4f}")
    print(f"ROC AUC Score (Class 0): {roc_auc_neg:.4f}")

    fpr_pos, tpr_pos, _ = roc_curve(y_test, y_scores[:, 1])
    fpr_neg, tpr_neg, _ = roc_curve(1 - y_test, y_scores[:, 0])

    plt.figure(figsize=(4, 3))
    plt.plot(
        fpr_pos, tpr_pos, label=f"Class 1 (ROC AUC = {roc_auc_pos:.4f})", color="blue"
    )
    plt.plot(
        fpr_neg, tpr_neg, label=f"Class 0 (ROC AUC = {roc_auc_neg:.4f})", color="red"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

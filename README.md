# Machine Learning Capstone Project: Home Credit Default Risk Prediction (DS.v2.5.3.4.1)

## Table of Contents

- [Project Overview](#project-overview)
- [Project Objectives](#project-objectives)
- [Key Insights](#key-insights)
- [Model Architecture and Methodology](#model-architecture-and-methodology)
- [Models Developed](#models-developed)
- [Model Deployment and Access](#model-deployment-and-access)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Notebook Overview](#notebook-overview)


## Project Overview
This project delivers a robust, end-to-end solution for loan default risk prediction in the retail banking industry. The key achievements include:

- **Actionable Insights**: Extensive data consolidation, preprocessing, and feature engineering to transform raw data into a reliable source for analysis.

- **In-Depth Analysis**: Comprehensive feature selection analysis to ensure model accuracy, efficiency, and interpretability, addressing key regulatory requirements.

- **Production-Ready Models**: Development of two distinct machine learning models—a Interpretable Production Model and a Low-Dependency Model—for binary classification and probability prediction.

- **Automated Infrastructure**: Implementation of automated hyperparameter tuning and seamless Google Cloud deployment to ensure the model is ready for a production environment.
 
---

## Project Objectives

This project was built to address a critical challenge for retail banks: accurately and reliably predicting loan default risk. To achieve this, I set out to build a comprehensive machine learning solution that would not only deliver accurate predictions but also meet key business and regulatory requirements.

My primary objectives were:

* **To Develop a Production-Ready Service:** To create a scalable and robust machine learning service that could be easily deployed and integrated into a bank's existing infrastructure. This involved containerizing the application with **Docker** and deploying it on **Google Cloud**.
* **To Prioritize Interpretability:** To develop a model that is not a "black box." My goal was to ensure the model's predictions could be understood and explained to both bank stakeholders and regulatory bodies, providing a transparent view of the factors driving a loan applicant's risk score.
* **To Optimize for Efficiency and Performance:** : To build a lightweight model with a limited number of features that could produce accurate predictions quickly and efficiently. This project's ultimate aim was to achieve a performance level comparable to complex, state-of-the-art ensemble models and top-ranking Kaggle solutions, but with a much simpler model based on the CatBoost algorithm.
* **To Provide a Flexible Solution:** To develop two distinct models—a **Interpretable Production Model** and a **Low-Dependency Model** for environments with limited or inaccessible external data. This provides the flexibility to adapt to various business use cases.



## Key Insights

Through extensive exploratory data analysis (EDA), statistical inference, and model-based analysis, I uncovered several key insights that guided the project's direction and influenced the final model architecture.

### Data-Driven Findings
* **Loan Type:** Applicants with **cash loans** demonstrated a higher risk of defaulting compared to those with revolving loans.
* **Demographics:** There is a clear correlation between age and default risk, with **younger applicants** showing a higher propensity to default.
* **Life Events and Status:** Individuals on **maternity leave** or who are **unemployed** exhibited a significantly higher rate of default.
* **Socioeconomic Factors:** A clear inverse relationship exists between education level and default rate; the **higher the education level, the lower the risk** of defaulting. Similarly, certain occupations (e.g., low-skilled laborers) and employer organizations were associated with elevated default rates.
* **Prior Credit History:** An unexpected finding was that individuals with a previous loan from the same lender (**HomeCredit**) had a higher default rate than those with prior credit from another institution.

### Statistical Inference

I performed several hypothesis tests to validate key findings from the EDA, providing statistical confidence in the relationships between these features and default risk.

* **Social Circle:** The presence of defaulters in an applicant's social circle is **positively and significantly associated** with an increased risk of defaulting (p-value < 0.001).
* **Gender:** Being male was found to be a **significant predictor of increased default risk** (p-value < 0.001).
* **Education Level:** Higher education levels were confirmed to have a **significant protective effect**, corresponding to a lower risk of defaulting (p-value < 0.001).

### Model-Based Insights

The final **CatBoost** models, based on the gradient-boosting algorithm, identified a few key features that drove the majority of its predictions. The first model primarily relied on the external evaluation scores (`EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_1`), followed by demographic features like `AGE` and `GENDER` and financial features like `AMT_CREDIT` and `AMT_ANNUITY`. Interestingly, the total sum of payments from previous credit installments (`PREV_INST_AMT_PAYMENT_SUM_SUM`) was also a highly important feature, reflecting the model's ability to learn from historical payment behavior. In the second model, which avoids relying on external evaluation scores, `AGE` becomes the predominant feature in the prediction model, also followed by features related to the loan and loaned product price.

### A Note on Dataset Particulars

Adversarial validation revealed a significant difference in the data distributions between the training and test sets on the Kaggle platform. This was particularly pronounced in key financial features like `AMT_CREDIT` and `AMT_ANNUITY`. To mitigate the risk of overfitting and ensure robust performance, I implemented extensive measures to account for these distributional shifts throughout the modeling phase.


## Model Architecture and Methodology

### Data Preprocessing
I began the data preprocessing phase with a focus on simplicity and effectiveness. For both numerical and categorical features, I used a `SimpleImputer` with a constant fill value. This approach yielded the best results and provided a robust, straightforward method for handling missing data. Furthermore, I engineered new features to capture more complex patterns, including:

* **Financial Ratios:** A `DEBT_TO_INCOME_RATIO` was created to provide a normalized view of an applicant's financial burden.
* **Categorical Groupings:** Features like `ORGANIZATION_TYPE` and `OCCUPATION` were regrouped into meaningful clusters based on their income and loan history.
* **Behavioral Indicators:** I created flags for inconsistent credit cases (`flag_inconsistent_credit_cases`), recent phone changes (`recent_phone_change_flag`), and social circle default ratios (`social_circle_default_ratios`).
 

### Feature Selection
I implemented a multi-stage feature selection process to identify a minimal set of features that maintained high predictive power. This was a critical step in building an efficient and interpretable model.

1.  **Initial Importance Analysis:** I used a **CatBoost** model with default parameters to analyze feature importance using a combination of methods, including `PredictionValuesChange`, `LossFunctionChange`, `eli5`, and **SHAP** values.
2.  **Elbow Method:** I applied the **Elbow Method** to the cumulative feature importance curve to identify the point where the returns began to diminish. This heuristic helped me determine an optimal number of features for the final models.
3.  **Recursive Feature Elimination:** To further refine the feature set, I used **CatBoost's** built-in feature selection, which employs a **Recursive Feature Elimination** algorithm. By combining this with the `RecursiveByLossFunctionChange` method, I was able to efficiently reduce the number of columns without a significant loss of information.

This process ultimately allowed me to select the top features for the optimized models.
 

### Model Selection and Hyperparameter Tuning
To quickly explore the landscape of potential models and establish a performance benchmark, I first used **MLJar AutoML**. This allowed me to efficiently identify CatBoost and LightGBM as the top-performing models, providing a clear target for subsequent fine-tuning.

**CatBoost** offers several key advantages, most notably its native handling of categorical features. Unlike other algorithms that require manual preprocessing steps like one-hot encoding, CatBoost can directly incorporate categorical data, preserving its natural relationships and simplifying the data preparation workflow. Furthermore, CatBoost employs a unique "ordered boosting" technique to combat a common pitfall in gradient boosting: prediction shift. By training on a subset of the data and using an ordered permutation of the remaining data to compute the gradient, it avoids the bias that can lead to overfitting, resulting in a more robust and generalizable model. This is particularly important with the given dataset. Moreover, the algorithm is also highly efficient, with optimization for both CPU and GPU, making it well-suited for large-scale datasets and real-time applications. This combination of intelligent feature handling, robust overfitting prevention, and high performance often allows CatBoost to achieve superior results.
 
For the final models, I used **Optuna** for automated hyperparameter tuning to find a balance between model complexity and generalization. My tuning strategy was a two-stage process based on the assumption that tree parameters and boosting parameters are independent:

1.  **Stage 1: Tree Parameters:** I first optimized tree-specific parameters like `depth` and `l2_leaf_reg` by fixing the learning rate at a high value and using early stopping.
2.  **Stage 2: Boosting Parameters:** Once the optimal tree structure was found, I fine-tuned the learning rate to maximize performance, pushing the boosting parameters to their extreme as needed.

Due to the computational intensity of hyperparameter tuning, I subsampled the majority class to reduce the training data size, which allowed for a more efficient search. This approach to regularization, with a reduced `depth` and an increased `l2_leaf_reg`, encouraged the model to be simpler and more conservative, resulting in better generalization to unseen data. 

 

## Models Developed

I developed two distinct machine learning models to provide a flexible solution for different deployment scenarios, each with a unique set of trade-offs. Both models were rigorously tuned to counter overfitting, focusing on regularization and reduced complexity to ensure excellent generalization to unseen data.

### Interpretable Production Model

This model represents the ideal balance between **high performance**, **interpretability**, and **efficiency**. It is the primary candidate for production deployment in a typical banking environment.

* **Architecture:** A **CatBoost** gradient-boosting algorithm trained on a highly optimized feature set.
* **Features:** The model is comprised of a limited, but powerful, set of **$n=18$ features** selected through a rigorous multi-stage process.
* **Performance:** The model achieved a private ROC-AUC score of **$0.753$**. While this is slightly below the top-performing AutoML ensemble models (max. $0.783$) and the competition winners by private score ($0.806$), its simplicity and efficiency make it a highly practical and desirable solution for a production environment.

### Low-Dependency Model

This model was designed as a robust alternative for situations where access to complex or external data (such as external credit scores) is limited.

* **Architecture:** A **CatBoost** algorithm trained on an easily accessible feature set.
* **Features:** The model is comprised of **$n=36$ features** that do not require external data sources.
* **Performance:** It achieved a private ROC-AUC score of **$0.759$**, demonstrating strong predictive power even with simplified inputs. This makes it a valuable asset for scenarios where data availability is a key constraint.
 

## Model Deployment and Access

The two machine learning model pipelines were encapsulated within an API using FastAPI, enabling them to be served as a single, accessible service. This service was then containerized with Docker and deployed on a Google Cloud service endpoint. The endpoint can be accessed for real-time predictions. For detailed information on the deployment process, the API endpoints, and the testing procedures, please refer to Notebook 04: Model Deployment and Testing.


## Installation

To set up this project locally:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/TuringCollegeSubmissions/jwerne-DS.v2.5.3.4.1.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd jwerne-DS.v2.5.3.4.1
   ```
3. **Install required packages**:
   Ensure Python is installed and use the following command:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Open the notebook in Jupyter or JupyterLab to explore the analysis. Execute the cells sequentially to understand the workflow, from data exploration to model building and evaluation. For an in-depth exploration, refer to the notebook overview below.


## Data

This project uses the Home Credit Default Risk dataset.

You can download it from:
https://www.kaggle.com/c/home-credit-default-risk/data

Place the downloaded files into `data/raw/` before running the notebooks.


## Directory Structure


```
project-root/
├── README.md                        # Project overview, goals, and setup instructions
├── risk_evaluation_plan.md          # Detailed investigation and POC plan [v.03]
├── requirements.txt                 # Python dependencies
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA, data consolidation, aggregation, and cleaning
│   ├── 02_statistical_inference.ipynb   # Hypothesis testing
│   ├── 03_modeling_default.ipynb    # Creating two models for default risk prediction
│   └── 04_model_deployment.ipynb    # Testing the deployed models 
├── src/
│   ├── features.py                  # Feature engineering functions
│   └── utils.py                     # Helper functions (e.g., plotting, metrics, statistical testing)
├── data/
│   ├── raw/                         # Unprocessed Home Credit dataset
│   └── processed/                   # Cleaned/merged datasets
└── docs/
    └── POC_Presentation.ppt         # Summary slides for demo meetings
```


## Requirements

The `requirements.txt` file lists all Python dependencies. Install them using the command provided above.


## Notebook Overview

The notebooks include the following sections:

**Notebook 1: Data Preparation and EDA**
1. Introduction
2. Data Acquisition 
3. Adversarial Validation
4. Exploratory Data Analysis 


**Notebook 2: Statistical Inference**
1. Introduction  
2. Statistical Inference and Evaluation  


**Notebook 3: Machine Learning Modeling - Default prediction**
1. Importing libraries  
2. Loading data 
3. Target   
4. Metric definition
5. AutoML
6. Standalone model Training, Tuning, and Evaluation
7. Test Submission   


**Notebook 4: Machine Learning Modeling - Model deployment**
1. Introduction  
2. Testing deployed models  
3. Further improvements  
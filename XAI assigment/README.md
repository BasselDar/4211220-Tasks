# Explainable AI (XAI) Analysis on Adult Income Dataset

This project demonstrates the application of various Explainable AI (XAI) techniques to interpret a machine learning model's predictions on the Adult Income dataset. The goal is to understand how different features influence the model's predictions and to provide transparent explanations for the model's decisions.

## Dataset

The Adult Income dataset is a well-known dataset from the UCI Machine Learning Repository that contains information about individuals' demographic and employment characteristics. The target variable is whether an individual's income exceeds $50K per year.

### Features
- **Numerical Features:**
  - Age
  - Final weight (fnlwgt)
  - Education number
  - Capital gain
  - Capital loss
  - Hours per week

- **Categorical Features:**
  - Workclass
  - Education
  - Marital status
  - Occupation
  - Relationship
  - Race
  - Sex
  - Native country

## XAI Techniques Implemented

1. **SHAP (SHapley Additive exPlanations)**
   - Global feature importance
   - Individual prediction explanations
   - Feature interaction analysis
   - Dependence plots for top features

2. **LIME (Local Interpretable Model-agnostic Explanations)**
   - Local explanations for individual predictions
   - Feature importance visualization
   - Decision boundary analysis

3. **Feature Importance Analysis**
   - LightGBM built-in feature importance
   - Comparison between different importance methods
   - Feature type analysis (categorical vs numerical)

## Visualizations

The project includes several visualizations to help understand the data and model:

1. **Data Analysis Visualizations**
   - Target variable distribution
   - Feature correlation heatmap
   - Feature distributions by income class
   - Age and hours per week distributions
   - Education and marital status distributions

2. **Model Performance Metrics**
   - ROC Curve
   - Precision-Recall Curve
   - Feature importance comparison

3. **XAI Visualizations**
   - SHAP summary plots
   - LIME explanation plots
   - Feature importance comparison plots
   - SHAP dependence plots

## Requirements

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - shap
  - lime
  - lightgbm

## Usage

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook xai_adult_income.ipynb
   ```

## Results

The analysis provides insights into:
- Key factors influencing income prediction
- Feature importance rankings
- Model performance metrics
- Individual prediction explanations
- Feature interactions and dependencies


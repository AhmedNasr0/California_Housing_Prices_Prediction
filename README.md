# California Housing Price Prediction
This project predicts the median house value in California districts using machine learning models. The dataset used is the California Housing Prices dataset from Kaggle, which contains various features such as population, number of bedrooms, and location information.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Model Used](#model_used)
- [Visualization](#visualization)

# Project Overview
The project is designed to compare the performance of several machine learning models to predict house prices. The models evaluated include:

* Multiple Linear Regression
* Support Vector Regression (SVR)
* Decision Tree
* Random Forest
* XGBoost
The model with the lowest **Mean Absolute Error (MAE)** is selected as the best model.

## Installation
1. Clone this repository:

```
git clone https://github.com/your-username/california-housing-price-prediction.git
cd california-housing-price-prediction
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
## Usage
1. **Data Cleaning :**
   handle all Missing values in the total_bedrooms column are replaced by the mean
2. **Data Preprocessing:**
    The dataset is preprocessed by handling missing values and converting categorical features to numerical features using one-hot encoding.
    Categorical features, like ocean_proximity, are handled using one-hot encoding.
    Train-Test Split: The dataset is split into training and testing sets using an 80-20 split.

## Model Selection: 
The following models are trained and evaluated:

* Multiple Linear Regression
* Support Vector Regression (SVR)
* Decision Tree
* Random Forest
* XGBoost
The model with the lowest **Mean Absolute Error (MAE)** is selected as the best model.

## Model Evaluation: 
After training, the model is evaluated using:

* Mean Absolute Error (MAE)
* Residual Plot: To visualize prediction errors.


## Models Used
* **Multiple Linear Regression:** A linear approach to modeling the relationship between features and target variable.
* **Support Vector Regression (SVR):** A non-linear regression model based on Support Vector Machines.
* **Decision Tree:** A tree-like model used for regression tasks.
* **Random Forest:** An ensemble learning method based on averaging predictions from multiple decision trees.
* **XGBoost:** An optimized gradient boosting model that typically provides better performance with lower MAE.
## Visualization
Several visualizations are provided:

* **MAE Comparison:** A bar plot comparing the Mean Absolute Error of the models.
```
plt.bar(models, mae, color='red')
```
* **Predicted vs Actual Values:** A scatter plot showing the predicted vs actual house prices for XGBoost.
```
plt.scatter(y_test, xgb_y_prediction)
```
* **California Map:** A plot showing the locations of houses with their median house value overlaid on a map of California.
```
plt.imshow(californiaImg, extent=[-124.8,-113.8,32.45,42], alpha=0.4)
```
## Results
After testing multiple models, XGBoost provided the lowest Mean Absolute Error (MAE):

Multiple Linear Regression: 49,697.07
* SVR: 87,481.49
* Decision Tree: 43,848.08
* Random Forest: 31,278.15
* XGBoost: 29,696.11
**XGBoost was chosen as the best model based on these results.**

## Future Work
* Hyperparameter Tuning: Further optimization of models could improve accuracy.
* Feature Engineering: Additional features such as interaction terms could be added to improve model performance.
* Cross-Validation: Perform cross-validation for better generalization.
* Model Deployment: Build a web app for users to input data and receive house price predictions.

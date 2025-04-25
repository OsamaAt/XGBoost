California Housing Price Prediction using XGBoost
This project uses the California Housing dataset from Scikit-learn to predict median house prices using the XGBoost Regressor.
Features
- Dataset: fetch_california_housing
- Model: XGBoost Regressor
- Evaluation Metrics: MAE, MSE, RMSE
- Visualization of Feature Importance

Workflow
- Load Data: Using Scikit-learn’s built-in dataset.
- Data Preprocessing:- Split into training/testing.
- Scale the features using StandardScaler.

- Modeling:- Train using XGBoostRegressor with 200 trees, max_depth=4, and learning_rate=0.1.

- Evaluation:- Print model scores.
- Show actual vs predicted prices.
- Plot feature importances.


Results
- Achieved decent performance on both training and testing datasets.
- Visualized the impact of each feature on the predictions.

Requirements
- numpy
- pandas
- matplotlib
- scikit-learn
- xgboost

Run the code
python your_script.py

Author OsamaAt

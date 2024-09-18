import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap


class Modelling:
    def __init__(self, data):
        self.data = data


    def preprocess(self, num_cols: list)->pd.DataFrame:
        '''
        This funciton normalizes teh data using standard scaler

        Parameters:
            num_cols(list): list of numerical columsn to process

        Returns:
            pd.DataFrame
        '''
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data[num_cols])
        scaled_data.DataFrame(scaled_data, columns=num_cols)

        return scaled_data


    def split_data(self, X, y, test_size = 0.2):
        '''
        Splits the data to training and testing

        Parameters:
            X: Features
            y: label

        Returns: 
            X_train, X_test, y_train, y_test 
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        return X_train, X_test, y_train, y_test 


    def model_testing(self, model, X_test, y_test):
        '''
        This function calculates the accuracy of the model

        Parameters:
            mode: A model that is fitted
            X_test
            y_test
        '''
        y_pred = model.predict(X_test)

        # Measure accuracy
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Output the metrics
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("R-squared:", r2)


# plot prediction vs actual value for each model 
def plot_predictions_vs_actuals(model, X_test, y_premium_test):
    # Get predictions from the model
    y_pred = model.predict(X_test)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_premium_test, y_pred, alpha=0.6, color='b')
    
    # Add line for perfect prediction
    plt.plot([min(y_premium_test), max(y_premium_test)], [min(y_premium_test), max(y_premium_test)], color='red', linewidth=2)

    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model.__class__.__name__} Predictions vs Actual Values')

    plt.grid(True)
    plt.show()


# a single scatter plot for model predictions vs actual values
def plot_all_models_predictions(models_list, X_test, y_premium_test, model_names):
    plt.figure(figsize=(10, 8))

    # Different markers or colors for each model
    markers = ['o', 's', '^', 'D']
    colors = ['blue', 'green', 'orange', 'purple']
    
    # Loop through each model and plot its predictions
    for idx, (model, model_name) in enumerate(models_list):
        # Get predictions
        y_pred = model.predict(X_test)

        # Plot actual vs predicted values
        plt.scatter(y_premium_test, y_pred, alpha=0.6, marker=markers[idx], color=colors[idx], label=model_name)
    
    # Plot the perfect prediction line
    plt.plot([min(y_premium_test), max(y_premium_test)], [min(y_premium_test), max(y_premium_test)], color='red', linewidth=2, label='Perfect Prediction')

    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values for TotalPremium')

    # Show legend to differentiate models
    plt.legend()

    # Show grid for readability
    plt.grid(True)

    plt.show()
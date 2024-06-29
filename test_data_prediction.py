#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import joblib

# Load Data
def load_data(train_path, predict_path):
    train_data = pd.read_csv(train_path)
    predict_data = pd.read_csv(predict_path)
    return train_data, predict_data

# Exploratory Data Analysis
def explore_data(data):
    print("___________________________________________________________________________________________________________")
    print(data.info())

# Identify and handle missing values
def handle_missing_values(data):
    missing_values_summary = data.isnull().sum()
    columns_with_missing_values = missing_values_summary[missing_values_summary > 0]
    print(columns_with_missing_values)
    return columns_with_missing_values

# Impute missing values in 'Count' and bin non-whole values
def impute_and_bin_counts(data):
    non_whole_counts = data[~data['Count'].isnull() & ~data['Count'].apply(float.is_integer)]
    bins = [0, 1, 10, 50, 100, 500, 1000, 5000, 10000]
    labels = ['0-1', '1-10', '10-50', '50-100', '100-500', '500-1000', '1000-5000', '5000-10000']
    non_whole_counts['Count_Binned'] = pd.cut(non_whole_counts['Count'], bins=bins, labels=labels, include_lowest=True)
    non_whole_count_distribution = non_whole_counts['Count_Binned'].value_counts().sort_index()
    print(non_whole_count_distribution)

    data['Count'] = data['Count'].apply(lambda x: 0 if x % 1 != 0 else x)
    data['Count'].fillna(0, inplace=True)
    return data

# Predict missing 'Category' values using Decision Tree
def impute_missing_category(data):
    data_with_category = data[data['Category'].notnull()]
    data_without_category = data[data['Category'].isnull()]
    
    features = ['Retail_Price', 'Promo_Price', 'Count', 'Manufacturer']
    target = 'Category'
    
    data_with_category_encoded = pd.get_dummies(data_with_category[features])
    data_without_category_encoded = pd.get_dummies(data_without_category[features])
    
    data_with_category_encoded, data_without_category_encoded = data_with_category_encoded.align(data_without_category_encoded, join='left', axis=1, fill_value=0)
    
    X = data_with_category_encoded
    y = data_with_category[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    predicted_categories = clf.predict(data_without_category_encoded)
    accuracy = accuracy_score(y_val, clf.predict(X_val))
    print(f'Accuracy of the Category prediction model: {accuracy:.2f}')
    
    data.loc[data['Category'].isnull(), 'Category'] = predicted_categories
    return data

# Predict missing 'Promo_Price' values using Random Forest Regressor
def impute_missing_promo_price(data):
    data_with_promo_price = data[data['Promo_Price'].notnull()]
    data_without_promo_price = data[data['Promo_Price'].isnull()]
    
    features = ['Retail_Price', 'Count', 'Manufacturer', 'Category']
    target = 'Promo_Price'
    
    data_with_promo_price_encoded = pd.get_dummies(data_with_promo_price[features])
    
    X = data_with_promo_price_encoded
    y = data_with_promo_price[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train)
    
    y_pred_val = regressor.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    mse = mean_squared_error(y_val, y_pred_val)
    rmse = np.sqrt(mse)
    
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    
    data_without_promo_price_encoded = pd.get_dummies(data_without_promo_price[features])
    data_without_promo_price_encoded = data_without_promo_price_encoded.reindex(columns=data_with_promo_price_encoded.columns, fill_value=0)
    predicted_promo_prices = regressor.predict(data_without_promo_price_encoded)
    
    data.loc[data['Promo_Price'].isnull(), 'Promo_Price'] = predicted_promo_prices
    return data

# Train and evaluate model to predict 'List_Price'
def train_and_evaluate_model(data, predict_data):
    features = ['Retail_Price', 'Promo_Price', 'Count', 'Manufacturer', 'Category']
    target = 'List_Price'
    
    data_encoded = pd.get_dummies(data[features])
    X = data_encoded
    y = data[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train)
    
    y_pred_val = regressor.predict(X_val)
    
    mae = mean_absolute_error(y_val, y_pred_val)
    mse = mean_squared_error(y_val, y_pred_val)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred_val)
    
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'R-squared (R2): {r2:.2f}')
    
    joblib.dump(regressor, 'random_forest_model.joblib')

    validation_results = X_val.copy()
    validation_results['Actual_List_Price'] = y_val
    validation_results['Predicted_List_Price'] = y_pred_val
    validation_results.to_csv('validation_results.csv', index=False)
    
    # Predict on new data
    predict_data_encoded = pd.get_dummies(predict_data[features])
    predict_data_encoded = predict_data_encoded.reindex(columns=data_encoded.columns, fill_value=0)
    
    predicted_list_prices = regressor.predict(predict_data_encoded)
    predict_data['Predicted_List_Price'] = predicted_list_prices
    predict_data.to_csv('predicted_data.csv', index=False)

# Main function
def main():
    train_data_path = 'cvs_wg_pns_train.csv'
    predict_data_path = 'cvs_wg_pns_predict_on.csv'
    
    train_data, predict_data = load_data(train_data_path, predict_data_path)
    
    print("Train Data Info:")
    explore_data(train_data)
    
    print("Handling Missing Values in Train Data:")
    handle_missing_values(train_data)
    
    print("Imputing and Binning Counts:")
    train_data = impute_and_bin_counts(train_data)
    
    print("Imputing Missing Category Values:")
    train_data = impute_missing_category(train_data)
    
    print("Imputing Missing Promo_Price Values:")
    train_data = impute_missing_promo_price(train_data)
    
    print("Training and Evaluating Model for List_Price:")
    train_and_evaluate_model(train_data, predict_data)
    
    print("Predict Data Info:")
    explore_data(predict_data)

if __name__ == "__main__":
    main()

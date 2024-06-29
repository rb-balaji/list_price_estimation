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
import joblib


# Load Data
def load_data(predict_path):
    predict_data = pd.read_csv(predict_path)
    return predict_data


# Exploratory Data Analysis
def explore_data(data):
    print("___________________________________________________________________________________________________________")
    print(data.info())
    print(data.head())


# Impute missing values in 'Count' and set to zero where 'Count' has decimal values
def preprocess_count(data):
    data['Count'] = data['Count'].apply(lambda x: 0 if pd.isnull(x) or x % 1 != 0 else x)
    return data


# Predict missing 'Category' values using Decision Tree
def preprocess_category(data):
    data_with_category = data[data['Category'].notnull()]
    data_without_category = data[data['Category'].isnull()]

    features = ['Retail_Price', 'Promo_Price', 'Count', 'Manufacturer']
    target = 'Category'

    data_with_category_encoded = pd.get_dummies(data_with_category[features])
    data_without_category_encoded = pd.get_dummies(data_without_category[features])

    data_with_category_encoded, data_without_category_encoded = data_with_category_encoded.align(
        data_without_category_encoded, join='left', axis=1, fill_value=0)

    X = data_with_category_encoded
    y = data_with_category[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    predicted_categories = clf.predict(data_without_category_encoded)
    print(f'Accuracy of the Category prediction model: {clf.score(X_val, y_val):.2f}')

    data.loc[data['Category'].isnull(), 'Category'] = predicted_categories
    return data


# Predict missing 'Promo_Price' values using Random Forest Regressor
def preprocess_promo_price(data):
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

    data_without_promo_price_encoded = pd.get_dummies(data_without_promo_price[features])
    data_without_promo_price_encoded = data_without_promo_price_encoded.reindex(
        columns=data_with_promo_price_encoded.columns, fill_value=0)
    predicted_promo_prices = regressor.predict(data_without_promo_price_encoded)

    data.loc[data['Promo_Price'].isnull(), 'Promo_Price'] = predicted_promo_prices
    return data


# Train model and predict on new data
def train_model_and_predict(predict_data):
    features = ['Retail_Price', 'Promo_Price', 'Count', 'Manufacturer', 'Category']

    predict_data_encoded = pd.get_dummies(predict_data[features])

    # Load the pre-trained model
    model_path = 'random_forest_model.joblib'
    regressor = joblib.load(model_path)

    # Ensure the columns match between training and prediction datasets
    # Note: Adjust train_data_encoded_columns to match your training data encoding
    train_data_encoded_columns = regressor.feature_names_in_  # Use the feature names from the trained model
    predict_data_encoded = predict_data_encoded.reindex(columns=train_data_encoded_columns, fill_value=0)

    # Predict `List_Price`
    predicted_list_prices = regressor.predict(predict_data_encoded)
    predict_data['Predicted_List_Price'] = predicted_list_prices

    # Save the predictions to a CSV file
    output_path = 'predicted_data123.csv'
    predict_data.to_csv(output_path, index=False)

    print("Predictions saved to", output_path)


# Main function
def main():
    predict_data_path = 'cvs_wg_pns_predict_on.csv'

    predict_data = load_data(predict_data_path)

    print("Prediction Data Info:")
    explore_data(predict_data)

    print("Preprocessing Count:")
    predict_data = preprocess_count(predict_data)

    print("Preprocessing Category:")
    predict_data = preprocess_category(predict_data)

    print("Preprocessing Promo_Price:")
    predict_data = preprocess_promo_price(predict_data)

    print("Predicting List_Price for New Data:")
    train_model_and_predict(predict_data)

    print("Final Prediction Data Info:")
    explore_data(predict_data)


if __name__ == "__main__":
    main()

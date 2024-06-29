#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error,  r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib


# In[5]:


test_data = pd.read_csv('cvs_wg_pns_train.csv')
predict_data = pd.read_csv('cvs_wg_pns_predict_on.csv')


# In[6]:


# print("___________________________________________________________________________________________________________")
# print(test_data.head())
print("___________________________________________________________________________________________________________")
print(test_data.info())
# print("___________________________________________________________________________________________________________")
# print(test_data.describe())


# In[7]:


# Identify columns with missing values
missing_values_summary = test_data.isnull().sum()

# Filter out columns with no missing values
columns_with_missing_values = missing_values_summary[missing_values_summary > 0]

# Print the columns with missing values and their counts
print(columns_with_missing_values)


# In[90]:


# Creating bins for granular distribution
bins = [0, 1, 10, 50, 100, 500, 1000, 5000, 10000]
labels = ['0-1', '1-10', '10-50', '50-100', '100-500', '500-1000', '1000-5000', '5000-10000']

# Filtering non-whole number 'Count' values
non_whole_counts = test_data[~test_data['Count'].isnull() & ~test_data['Count'].apply(float.is_integer)]

# Binning the non-whole number 'Count' values
non_whole_counts['Count_Binned'] = pd.cut(non_whole_counts['Count'], bins=bins, labels=labels, include_lowest=True)

# Getting the distribution of non-whole number 'Count' values
non_whole_count_distribution = non_whole_counts['Count_Binned'].value_counts().sort_index()

# Display the distribution
print(non_whole_count_distribution)


# In[8]:


# Filter records where 'Count' is missing
missing_count_records = test_data[test_data['Count'].isnull()]

# Check for missing 'Category' and 'Multiplier' in the filtered records
missing_category_count = missing_count_records['Category'].isnull().sum()
missing_multiplier_count = missing_count_records['Multiplier'].isnull().sum()

# Get the total number of records with missing 'Count' value
total_missing_count_records = len(missing_count_records)

print(f'Total number of records with missing Count values: {total_missing_count_records}')
print(f'Number of records with missing Count and missing Category values: {missing_category_count}')
print(f'Number of records with missing Count and missing Multiplier values: {missing_multiplier_count}')

# Validate the hypothesis
if missing_category_count == total_missing_count_records and missing_multiplier_count == total_missing_count_records:
    print("Hypothesis is validated: All records with missing 'Count' also have missing 'Category' and 'Multiplier' values.")
else:
    print("Hypothesis is not validated: Some records with missing 'Count' have 'Category' or 'Multiplier' values.")


# In[9]:


# Impute missing values in 'Count' with 0
train = test_data['Count'].fillna(0, inplace=True)

# Verify the imputation
missing_count_abs = test_data['Count'].isnull().sum()
print(f'Number of records with missing Count values after imputation: {missing_count_abs}')


# In[12]:


# Impute 'Count' values: set to zero where 'Count' has decimal values
test_data['Count'] = test_data['Count'].apply(lambda x: 0 if x % 1 != 0 else x)


# In[93]:


# # Round off the 'Count' column values to the nearest whole number
# test_data['Count'] = test_data['Count'].round()

# # Verify the changes
# print(test_data['Count'].head())


# In[13]:


# Separate data into records with and without 'Category'
data_with_category = test_data[test_data['Category'].notnull()]
data_without_category = test_data[test_data['Category'].isnull()]

# Features and target for the prediction model
features = ['Retail_Price', 'Promo_Price', 'Count', 'Manufacturer']
target = 'Category'

# One-hot encode categorical features
data_with_category_encoded = pd.get_dummies(data_with_category[features])
data_without_category_encoded = pd.get_dummies(data_without_category[features])

# Ensure both datasets have the same dummy variables
data_with_category_encoded, data_without_category_encoded = data_with_category_encoded.align(data_without_category_encoded, join='left', axis=1, fill_value=0)

# Train a Decision Tree Classifier to predict 'Category'
X = data_with_category_encoded
y = data_with_category[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict missing 'Category' values
predicted_categories = clf.predict(data_without_category_encoded)

# Predict and evaluate on the validation set
y_pred_val = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_val)
print(f'Accuracy of the Category prediction model: {accuracy:.2f}')

# Impute the missing values
test_data.loc[test_data['Category'].isnull(), 'Category'] = predicted_categories

# # Save the imputed dataset to a new CSV file (optional)
# test_data.to_csv('cvs_wg_pns_train_category_imputed_final.csv', index=False)

# Verify the imputation
missing_category_count = test_data['Category'].isnull().sum()
print(f'Number of records with missing Category values after imputation: {missing_category_count}')


# In[14]:


# Calculate the number of records with empty or zero 'Count' values
empty_count = test_data['Count'].isnull().sum()
zero_count = (test_data['Count'] == 0).sum()
total_empty_or_zero = empty_count + zero_count

print(f'Number of records with empty Count values: {empty_count}')
print(f'Number of records with zero Count values: {zero_count}')
print(f'Total number of records with empty or zero Count values: {total_empty_or_zero}')


# In[15]:


# Impute missing (NaN) 'Count' values with 0
test_data['Count'].fillna(0, inplace=True)

# Ensure zero 'Count' values remain as 0 (already handled by the fillna step)
test_data.loc[test_data['Count'] == 0, 'Count'] = 0

# # Save the imputed dataset to a new CSV file (optional)
# test_data.to_csv('path_to_your_data/cvs_wg_pns_train_imputed.csv', index=False)

# Verify the imputation
empty_count = test_data['Count'].isnull().sum()
zero_count = (test_data['Count'] == 0).sum()
total_empty_or_zero = empty_count + zero_count

print(f'Number of records with empty Count values after imputation: {empty_count}')
print(f'Number of records with zero Count values after imputation: {zero_count}')
print(f'Total number of records with empty or zero Count values after imputation: {total_empty_or_zero}')


# In[97]:


# # Calculate the number of missing values for the Multiplier column before imputation
# missing_multiplier_before = test_data['Multiplier'].isnull().sum()

# # Calculate the Multiplier for records with missing Multiplier values
# test_data['Multiplier'] = np.where(test_data['Multiplier'].isnull(), 
#                                    test_data['Retail_Price'] * test_data['Count'], 
#                                    test_data['Multiplier'])

# # Calculate the number of missing values for the Multiplier column after imputation
# missing_multiplier_after = test_data['Multiplier'].isnull().sum()



# # # Save the modified dataset to a new CSV file (optional)
# # test_data.to_csv('path_to_your_data/cvs_wg_pns_train_with_interactions.csv', index=False)

# # Verify the new features
# print(test_data[['Retail_Price', 'Count', 'Multiplier']].head())

# # Output the number of missing values before and after imputation
# print(f'Missing Multiplier values before imputation: {missing_multiplier_before}')
# print(f'Missing Multiplier values after imputation: {missing_multiplier_after}')


# In[16]:


# # Load the training data
# test_data = pd.read_csv('path_to_your_data/cvs_wg_pns_train.csv')

# Separate data into records with and without 'Promo_Price'
data_with_promo_price = test_data[test_data['Promo_Price'].notnull()]
data_without_promo_price = test_data[test_data['Promo_Price'].isnull()]

# Features and target for the prediction model
features = ['Retail_Price', 'Count', 'Manufacturer', 'Category']
target = 'Promo_Price'

# One-hot encode categorical features
data_with_promo_price_encoded = pd.get_dummies(data_with_promo_price[features])

# Train a Random Forest Regressor to predict 'Promo_Price'
X = data_with_promo_price_encoded
y = data_with_promo_price[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Predict on the validation set
y_pred_val = regressor.predict(X_val)

# Calculate evaluation metrics
mae = mean_absolute_error(y_val, y_pred_val)
mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

# Predict missing 'Promo_Price' values
data_without_promo_price_encoded = pd.get_dummies(data_without_promo_price[features])
data_without_promo_price_encoded = data_without_promo_price_encoded.reindex(columns=data_with_promo_price_encoded.columns, fill_value=0)
predicted_promo_prices = regressor.predict(data_without_promo_price_encoded)

# Impute the missing values
test_data.loc[test_data['Promo_Price'].isnull(), 'Promo_Price'] = predicted_promo_prices

# # Save the imputed dataset to a new CSV file (optional)
# test_data.to_csv('path_to_your_data/cvs_wg_pns_train_promo_price_imputed_model.csv', index=False)

# Verify the imputation
missing_promo_price_count = test_data['Promo_Price'].isnull().sum()
print(f'Number of records with missing Promo_Price values after imputation: {missing_promo_price_count}')


# In[17]:


# print("___________________________________________________________________________________________________________")
# print(test_data.head())
print("___________________________________________________________________________________________________________")
print(test_data.info())
# print("___________________________________________________________________________________________________________")
# print(test_data.describe())


# In[18]:


# Verify the imputation
missing_category_count = test_data['Category'].isnull().sum()
print(f'Number of records with missing Category values after imputation: {missing_category_count}')


# In[19]:


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# # Load the training data
# test_data = pd.read_csv('path_to_your_data/cvs_wg_pns_train.csv')

# Check for missing values in 'Category'
missing_category_count = test_data['Category'].isnull().sum()
print(f'Number of records with missing Category values: {missing_category_count}')

if missing_category_count == 0:
    print("No missing Category values to impute.")
else:
    # Separate data into records with and without 'Category'
    data_with_category = test_data[test_data['Category'].notnull()]
    data_without_category = test_data[test_data['Category'].isnull()]

    # Features and target for the prediction model
    features = ['Retail_Price', 'Promo_Price', 'Count', 'Manufacturer']
    target = 'Category'

    # One-hot encode categorical features
    data_with_category_encoded = pd.get_dummies(data_with_category[features])
    data_without_category_encoded = pd.get_dummies(data_without_category[features])

    # Ensure both datasets have the same dummy variables
    data_with_category_encoded, data_without_category_encoded = data_with_category_encoded.align(data_without_category_encoded, join='left', axis=1, fill_value=0)

    # Debugging: Check the shapes of the encoded data
    print(f'Shape of data_with_category_encoded: {data_with_category_encoded.shape}')
    print(f'Shape of data_without_category_encoded: {data_without_category_encoded.shape}')

    # Check if data_without_category_encoded is empty
    if data_without_category_encoded.shape[0] == 0:
        print("No data to predict. Exiting.")
    else:
        # Train a Decision Tree Classifier to predict 'Category'
        X = data_with_category_encoded
        y = data_with_category[target]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Predict missing 'Category' values
        predicted_categories = clf.predict(data_without_category_encoded)

        # Predict and evaluate on the validation set
        y_pred_val = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        print(f'Accuracy of the Category prediction model: {accuracy:.2f}')

        # Impute the missing values
        test_data.loc[test_data['Category'].isnull(), 'Category'] = predicted_categories

        # Verify the imputation
        missing_category_count_after = test_data['Category'].isnull().sum()
        print(f'Number of records with missing Category values after imputation: {missing_category_count_after}')


# In[36]:


# # Calculate the mode of the 'Category' column
# mode_category = test_data['Category'].mode()[0]
# print(f'The mode of the Category column is: {mode_category}')

# # Impute missing values in 'Category' with the mode
# test_data['Category'].fillna(mode_category, inplace=True)

# # Verify the imputation
# missing_category_count = test_data['Category'].isnull().sum()
# print(f'Number of records with missing Category values after imputation: {missing_category_count}')

# # # Save the imputed dataset to a new CSV file (optional)
# # test_data.to_csv('path_to_your_data/cvs_wg_pns_train_category_imputed_mode.csv', index=False)


# In[103]:


# print("___________________________________________________________________________________________________________")
# print(test_data.head())
print("___________________________________________________________________________________________________________")
print(test_data.info())
# print("___________________________________________________________________________________________________________")
# print(test_data.describe())


# In[26]:


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# import joblib

# # Load the prepared data
# test_data = pd.read_csv('path_to_your_data/cvs_wg_pns_train.csv')

# Prepare the data
features = ['Retail_Price', 'Promo_Price', 'Count', 'Manufacturer', 'Category']
target = 'List_Price'

# One-hot encode categorical features
test_data_encoded = pd.get_dummies(test_data[features])

# Separate features and target variable
X = test_data_encoded
y = test_data[target]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Predict on the validation set
y_pred_val = regressor.predict(X_val)

# Calculate evaluation metrics
mae = mean_absolute_error(y_val, y_pred_val)
mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred_val)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# Optionally, save the trained model
joblib.dump(regressor, 'random_forest_model.joblib')

# Save the predictions along with validation data to a new CSV file (optional)
validation_results = X_val.copy()
validation_results['Actual_List_Price'] = y_val
validation_results['Predicted_List_Price'] = y_pred_val
# validation_results.to_csv('path_to_your_data/validation_results.csv', index=False)


# In[113]:


# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np

# # Load the datasets
# test_data = pd.read_csv('cvs_wg_pns_train.csv')
# predict_data = pd.read_csv('cvs_wg_pns_predict_on.csv')

# Define the feature columns
features = ['Retail_Price', 'Promo_Price', 'Count', 'Manufacturer', 'Category']

# Fill missing 'Count' values with 0 in the training and prediction datasets
test_data['Count'].fillna(0, inplace=True)
predict_data['Count'].fillna(0, inplace=True)

# Round 'Count' values to the nearest whole number
test_data['Count'] = test_data['Count'].round()
predict_data['Count'] = predict_data['Count'].round()

# Impute missing 'Multiplier' values
test_data['Multiplier'] = np.where(test_data['Multiplier'].isnull(), 
                                   test_data['Retail_Price'] * test_data['Count'], 
                                   test_data['Multiplier'])
predict_data['Multiplier'] = np.where(predict_data['Multiplier'].isnull(), 
                                      predict_data['Retail_Price'] * predict_data['Count'], 
                                      predict_data['Multiplier'])

# One-hot encode categorical features in the training and prediction datasets
test_data_encoded = pd.get_dummies(test_data[features])
predict_data_encoded = pd.get_dummies(predict_data[features])

# Align the new data with the training data to ensure it has the same columns
test_data_encoded, predict_data_encoded = test_data_encoded.align(predict_data_encoded, join='left', axis=1, fill_value=0)

# Define the target and features for training
X_train = test_data_encoded
y_train = test_data['List_Price']

# Train the Random Forest Regressor model
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Predict on the new dataset
predicted_list_prices = regressor.predict(predict_data_encoded)

# Add the predictions to the original data
predict_data['Predicted_List_Price'] = predicted_list_prices

# Save the prediction results for review (optional)
predict_data.to_csv('final_predicted_list_prices.csv', index=False)

# Output the first few rows of the prediction results
print(predict_data.head())

# If actual list prices are available in the prediction dataset, calculate accuracy
if 'List_Price' in predict_data.columns:
    actual_list_prices = predict_data['List_Price']
    mae = mean_absolute_error(actual_list_prices, predicted_list_prices)
    mse = mean_squared_error(actual_list_prices, predicted_list_prices)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_list_prices, predicted_list_prices)

    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'R-squared (R2): {r2:.2f}')
else:
    print("Actual list prices are not available in the prediction dataset for accuracy calculation.")


# In[111]:


print(predict_data.info())


# In[ ]:





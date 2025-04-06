# --------------------------------------------------------------
# Honor Code
#
# I, Shubam Khantwal, certify that this project is my original work.
# I have not copied or used unauthorized materials, resources, or
# third-party services in completing this project, except as permitted
# by the course instructor. All sources used have been appropriately
# cited in the project report and code documentation.

# --------------------------------------------------------------


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


file_path = "~/Retail_Transaction_Dataset.csv"
df = pd.read_csv(file_path)

print("Basic Statistical Summary:\n", df.describe())

print(df.head())

print(df.info())

# Data Cleaning
# Check for NaN Objects
nan_values = df.isna().sum()
print("Missing Values in Each Column:\n", nan_values)

duplicates = df.duplicated().sum()
print("Number of Duplicate Rows:", duplicates)

df = df.drop_duplicates()

# Feature Engineering
# Convert DiscountApplied(%) to an absolute value
df['DiscountApplied'] = (df['DiscountApplied(%)'] / 100) * df['Price']

# Drop the original DiscountApplied(%) column if no longer needed
df = df.drop(columns=['DiscountApplied(%)'])

df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['Year'] = df['TransactionDate'].dt.year
df['Month'] = df['TransactionDate'].dt.month
df['Day'] = df['TransactionDate'].dt.day
df = df.drop(columns=['TransactionDate'])

# Discretization & Binarization
le_store_location = LabelEncoder()
le_product_category = LabelEncoder()
le_payment_method = LabelEncoder()
le_product = LabelEncoder()

df['StoreLocation'] = le_product_category.fit_transform(df['StoreLocation'])
df['ProductCategory'] = le_product_category.fit_transform(df['ProductCategory'])
df['PaymentMethod'] = le_payment_method.fit_transform(df['PaymentMethod'])
df['ProductID'] = le_product.fit_transform(df['ProductID'])
print("Basic Statistical Summary:\n", df.describe())


plt.figure(figsize=(12, 6))
sns.histplot(df['TotalAmount'], kde=True, color='blue')
plt.title('Distribution of Total Amount')
plt.xlabel('Total Amount')
plt.ylabel('Frequency')
plt.show()

target_variable = df['TotalAmount']

# Visualizing Quantity vs. Total Amount
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quantity', y='TotalAmount', data=df)
plt.title('Quantity vs. Total Amount')
plt.xlabel('Quantity')
plt.ylabel('Total Amount')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Price', y='TotalAmount', data=df)
plt.title('Price vs. Total Amount')
plt.xlabel('Price')
plt.ylabel('Total Amount')
plt.show()

# Aggregate total sales and total quantity by ProductCategory
aggregated_data = df.groupby('ProductCategory').agg({'TotalAmount': 'sum', 'Quantity': 'sum'}).reset_index()
print("Aggregated Data (Total Sales and Quantity by Product Category):\n", aggregated_data)

# Visualizing Aggregated Data
g = sns.barplot(data=aggregated_data, x='ProductCategory', y='TotalAmount', palette='Blues_d')
plt.title('Total Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.show()

# Variable Transformation
scaler = StandardScaler()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Loop through each numerical feature for outlier detection and removal
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot for {feature} (Outlier Detection)')
    plt.xlabel(feature)
    plt.show()

    # Calculate IQR and bounds for each feature
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the rows outside of the bounds for this feature
    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]


for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot for {feature} (Outlier Detection)')
    plt.xlabel(feature)
    plt.show()


# Random Forest for Feature Importance
X = df.drop(columns=['TotalAmount'])
y = df['TotalAmount']
rf = RandomForestRegressor(n_estimators=100, random_state=5805)
rf.fit(X, y)
feature_importances = pd.Series(data=rf.feature_importances_, index=X.columns)

print("Feature Importances from Random Forest:\n",feature_importances)

# Step 5: Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Step 6: Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Step 7: Plotting the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='b')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance from Random Forest')
plt.grid(axis='x')
plt.show()

# Selecting top features based on importance (for demonstration, keeping top 5 features)
top_features = importance_df.head(5).index.tolist()
X_selected = X[top_features]

# Variance Inflation Factor (VIF) to check for multicollinearity
try:
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_selected.columns
    vif_data['VIF'] = [variance_inflation_factor(X_selected.values, i) for i in range(X_selected.shape[1])]
    print("Variance Inflation Factor (VIF) for Selected Features:\n", vif_data)
    # Removing features with high VIF (VIF > 10 indicates multicollinearity)
    filtered_features = vif_data[vif_data['VIF'] <= 10]['Feature']
    if filtered_features.empty:
        # If all features have high VIF, keep the ones with lowest VIF
        filtered_features = vif_data.nsmallest(2, 'VIF')['Feature']
    X_final = X_selected[filtered_features]
    print("Final Selected Features after VIF Analysis:\n", X_final.columns.tolist())

except Exception as e:
    print("Error calculating VIF:", e)
    X_final = X_selected  # Fallback to using selected features without VIF filtering

# Step 1.8: Covariance Matrix and Pearson Correlation
cov_matrix = np.cov(X_final, rowvar=False)
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Sample Covariance Matrix Heatmap (Scaled Data)')
plt.show()

# Sample Pearson Correlation Coefficients Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(X_final.corr(), annot=True, cmap='coolwarm')
plt.title('Pearson Correlation Coefficients Heatmap')
plt.show()


##Regression

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import f
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_final, df['TotalAmount'],
                                                    test_size=0.2, random_state=5805)

# Fit the model
model = sm.OLS(y_train, X_train).fit()

# Model summary
print("Full Model Summary:")
print(model.summary())

#T-Test results (t-statistic and p-values)
t_test_results = pd.DataFrame({
    # "Feature": X_train.columns,
    "T-Statistic": model.tvalues,
    "P-Value": model.pvalues
})
print("\nT-Test Results:")
print(t_test_results)


# Fit the full model with all features
full_model = sm.OLS(y_train, X_train).fit()
f_test_results = []

# Iterate over each feature to calculate F-test values
for feature in X_train.columns:
    # Remove the current feature
    reduced_X = X_train.drop(columns=[feature])

    # Fit the reduced model
    reduced_model = sm.OLS(y_train, reduced_X).fit()

    # Calculate Residual Sum of Squares (RSS)
    RSS_full = sum((y_train - full_model.fittedvalues) ** 2)
    RSS_reduced = sum((y_train - reduced_model.fittedvalues) ** 2)

    # Degrees of freedom
    df1 = 1  # Degrees of freedom for the feature
    df2 = len(y_train) - len(X_train.columns)  # Residual degrees of freedom

    # Calculate F-statistic
    f_stat = ((RSS_reduced - RSS_full) / df1) / (RSS_full / df2)

    # Calculate p-value using scipy.stats
    p_value = f.sf(f_stat, df1, df2)

    # Append the results to the list
    f_test_results.append({
        'Feature': feature,
        'F-Statistic': f_stat,
        'P-Value': p_value
    })

# Convert the results to a DataFrame
f_test_results_df = pd.DataFrame(f_test_results)

# Display the results
print("\n F-Test Results:")
print(f_test_results_df)

#F-Test results (overall model significance)
f_statistic = model.fvalue
f_pvalue = model.f_pvalue
print(f"\nF-Test Statistic: {f_statistic: .3f}")
print(f"F-Test P-Value: {f_pvalue}")

# Predict on train and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# Plotting Train, Test, and Predicted Data
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Test Data", color='blue')
plt.plot(y_test_pred.values, label="Predicted Test Data", color='red')
plt.plot(y_train_pred.values, label="Predicted Train Data", color='green', alpha=0.7)
plt.title("Train vs Test vs Predicted Data")
plt.legend()
plt.xlabel("Observations")
plt.ylabel("Total Amount")
plt.show()

# Plot actual vs predicted values for data
plt.figure(figsize=(10, 6))
plt.scatter(y_test.values, y_test_pred.values, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Data')
plt.ylabel('Predicted Data')
plt.title('Actual vs Predicted Data')
plt.legend()
plt.grid()
plt.show()

# Calculate and display performance metrics
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

r_squared = model.rsquared
adj_r_squared = model.rsquared_adj
aic = model.aic
bic = model.bic

# Confidence Intervals for coefficients
conf_intervals = model.conf_int()
conf_intervals.columns = ['0', '1']
print("\nConfidence Intervals:")
print(conf_intervals)

def stepwise_selection(X, y, threshold_out=0.001):
    included = list(X.columns)
    removed_features = []

    while True:
        changed = False
        # Backward step: remove multiple features with p-values > threshold_out
        model = sm.OLS(y, pd.DataFrame(X[included])).fit()
        pvalues = model.pvalues
        features_to_remove = pvalues[pvalues > threshold_out].index.tolist()

        if features_to_remove:
            # Remove features with p-values > threshold_out
            for feature in features_to_remove:
                included.remove(feature)
                removed_features.append(feature)
                print(f"Removing feature: {feature} (p={pvalues[feature]})")
            changed = True

        if not changed:
            break

    return included, removed_features

# stepwise regression
selected_features, removed_features = stepwise_selection(X_train, y_train)
print("\nFinal Selected Features:")
print(selected_features)
print("\nRemoved Features:")
print(removed_features)

# Final model with stepwise-selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

final_model = sm.OLS(y_train, X_train_selected).fit()
print("\nFinal Model Summary:")
print(final_model.summary())

# Predict and evaluate the final model
y_train_pred_final = final_model.predict(X_train_selected)
y_test_pred_final = final_model.predict(X_test_selected)

final_mse_test = mean_squared_error(y_test, y_test_pred_final)



# Calculate MAE
final_mae_test = mean_absolute_error(y_test, y_test_pred_final)

# Updated results
final_results_df = pd.DataFrame({
    'Metric': ['R-Squared', 'Adjusted R-Squared', 'AIC', 'BIC', 'MSE', 'MAE'],
    'Value': [final_model.rsquared, final_model.rsquared_adj, final_model.aic, final_model.bic,
              final_mse_test, final_mae_test]
})
print("\nFinal Model Performance Metrics:")
print(final_results_df)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.combine import SMOTETomek


# Load the dataset
df = pd.read_csv('~/Retail_Transaction_Dataset.csv')

# Preprocessing Step
# Convert TransactionDate to datetime and extract features
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['Year'] = df['TransactionDate'].dt.year
df['Month'] = df['TransactionDate'].dt.month
df['Day'] = df['TransactionDate'].dt.day
df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek
df['Hour'] = df['TransactionDate'].dt.hour

# Add new features: is_weekend and is_holiday
# Assuming weekends are Saturday (5) and Sunday (6)
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [5, 6] else 0)
# Assuming a placeholder function for holidays
def is_holiday(date):
    holidays = ['2023-12-25', '2024-01-01']  # Add more holidays as necessary
    return 1 if date.strftime('%Y-%m-%d') in holidays else 0

df['IsHoliday'] = df['TransactionDate'].apply(is_holiday)

# Drop original TransactionDate column
df.drop('TransactionDate', axis=1, inplace=True)

# Encode categorical features
label_encoders = {}
categorical_columns = ['ProductID', 'PaymentMethod', 'StoreLocation']

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Encode target variable ProductCategory
target_encoder = LabelEncoder()
df['ProductCategory'] = target_encoder.fit_transform(df['ProductCategory'])

# Features and target selection
X = df.drop(['CustomerID', 'ProductCategory'], axis=1)
y = df['ProductCategory']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTETomek (combination of SMOTE and Tomek links)
smotetomek = SMOTETomek(random_state=42)
X_train, y_train = smotetomek.fit_resample(X_train, y_train)

# Normalize numerical features using Min-Max Scaler
scaler = MinMaxScaler()
numerical_features = ['Quantity', 'Price', 'DiscountApplied(%)', 'TotalAmount', 'Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'IsWeekend', 'IsHoliday']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Data Augmentation and Feature Engineering to Improve Model Performance
# Create interaction terms and polynomial features for numerical columns
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train[numerical_features])
X_test_poly = poly.transform(X_test[numerical_features])

# Concatenate the polynomial features with the rest of the dataset
X_train = np.concatenate([X_train.drop(columns=numerical_features).values, X_train_poly], axis=1)
X_test = np.concatenate([X_test.drop(columns=numerical_features).values, X_test_poly], axis=1)

# Perform Grid Search with Cross Validation to find the best parameters for Decision Tree with pre-pruning
param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.01, 0.05, 0.1]
}

grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, n_jobs=-1, verbose=2)
grid_search_dt.fit(X_train, y_train)

# Get the best parameters and the best estimator for Decision Tree
best_params_dt = grid_search_dt.best_params_
best_estimator_dt = grid_search_dt.best_estimator_

# Make predictions using the best Decision Tree estimator
y_pred_best_dt = best_estimator_dt.predict(X_test)

# Evaluation of the best Decision Tree model
classification_rep_best_dt = classification_report(y_test, y_pred_best_dt, target_names=target_encoder.classes_)
conf_matrix_best_dt = confusion_matrix(y_test, y_pred_best_dt)

print("Best Parameters (Decision Tree):", best_params_dt)
print("\nClassification Report (Decision Tree):\n", classification_rep_best_dt)
print("\nConfusion Matrix (Decision Tree):\n", conf_matrix_best_dt)

# Optimize Cost Complexity function with post-pruning for Decision Tree
path = best_estimator_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # Exclude the maximum value as it prunes all nodes

trees = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha, **{k: v for k, v in best_params_dt.items() if k != 'ccp_alpha'})
    tree.fit(X_train, y_train)
    trees.append(tree)

# Evaluate each pruned tree and pick the one with the best cross-validation score
from sklearn.model_selection import cross_val_score

best_score = 0
best_tree = None
for tree in trees:
    scores = cross_val_score(tree, X_train, y_train, cv=5)
    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_tree = tree

# Make predictions using the best pruned Decision Tree estimator
y_pred_best_pruned_dt = best_tree.predict(X_test)

# Evaluation of the best pruned Decision Tree model
classification_rep_best_pruned_dt = classification_report(y_test, y_pred_best_pruned_dt, target_names=target_encoder.classes_)
conf_matrix_best_pruned_dt = confusion_matrix(y_test, y_pred_best_pruned_dt)

print("\nClassification Report (Pruned Decision Tree):\n", classification_rep_best_pruned_dt)
print("\nConfusion Matrix (Pruned Decision Tree):\n", conf_matrix_best_pruned_dt)

# Perform Grid Search with Cross Validation to find the best parameters for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

# Get the best parameters and the best estimator for Random Forest
best_params_rf = grid_search_rf.best_params_
best_estimator_rf = grid_search_rf.best_estimator_

# Make predictions using the best Random Forest estimator
y_pred_best_rf = best_estimator_rf.predict(X_test)

# Evaluation of the best Random Forest model
classification_rep_best_rf = classification_report(y_test, y_pred_best_rf, target_names=target_encoder.classes_)
conf_matrix_best_rf = confusion_matrix(y_test, y_pred_best_rf)

print("Best Parameters (Random Forest):", best_params_rf)
print("\nClassification Report (Random Forest):\n", classification_rep_best_rf)
print("\nConfusion Matrix (Random Forest):\n", conf_matrix_best_rf)

# Perform Grid Search with Cross Validation to find the best parameters for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=5, n_jobs=-1, verbose=2)
grid_search_gb.fit(X_train, y_train)

# Get the best parameters and the best estimator for Gradient Boosting
best_params_gb = grid_search_gb.best_params_
best_estimator_gb = grid_search_gb.best_estimator_

# Make predictions using the best Gradient Boosting estimator
y_pred_best_gb = best_estimator_gb.predict(X_test)

# Evaluation of the best Gradient Boosting model
classification_rep_best_gb = classification_report(y_test, y_pred_best_gb, target_names=target_encoder.classes_)
conf_matrix_best_gb = confusion_matrix(y_test, y_pred_best_gb)

print("Best Parameters (Gradient Boosting):", best_params_gb)
print("\nClassification Report (Gradient Boosting):\n", classification_rep_best_gb)
print("\nConfusion Matrix (Gradient Boosting):\n", conf_matrix_best_gb)

# //////////////////////////////////////
#


from sklearn.model_selection import train_test_split

# # Assuming X_final contains the final features and data['TotalAmount'] is the target
# X_train, X_test, y_train, y_test = train_test_split(X_final, data['TotalAmount'], test_size=0.2, random_state=5805)

import statsmodels.api as sm

# Adding a constant for the intercept
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit the model
model = sm.OLS(y_train, X_train_const).fit()

# Model summary
print(model.summary())

import matplotlib.pyplot as plt

# Predict on train and test sets
y_train_pred = model.predict(X_train_const)
y_test_pred = model.predict(X_test_const)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Test Data", color='blue')
plt.plot(y_test_pred.values, label="Predicted Test Data", color='red')
plt.plot(y_train_pred.values, label="Predicted Train Data", color='green', alpha=0.7)
plt.title("Train vs Test vs Predicted Data")
plt.legend()
plt.xlabel("Observations")
plt.ylabel("Total Amount")
plt.show()


from sklearn.metrics import mean_squared_error

# Calculate MSE
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

# Extract R-squared, Adjusted R-squared, AIC, and BIC
r_squared = model.rsquared
adj_r_squared = model.rsquared_adj
aic = model.aic
bic = model.bic

# Results Table
import pandas as pd

results_df = pd.DataFrame({
    'Metric': ['R-Squared', 'Adjusted R-Squared', 'AIC', 'BIC', 'MSE (Train)', 'MSE (Test)'],
    'Value': [r_squared, adj_r_squared, aic, bic, mse_train, mse_test]
})
print(results_df)

# Confidence Intervals
conf_intervals = model.conf_int()
conf_intervals.columns = ['Lower Bound', 'Upper Bound']
conf_intervals.index.name = 'Predictor'
print(conf_intervals)


# Stepwise selection without constant and feature removal one by one
def stepwise_selection(X, y, threshold_in=0.01, threshold_out=0.05):
    included = list(X.columns)
    while True:
        changed = False
        # Backward step: remove one feature at a time
        model = sm.OLS(y, pd.DataFrame(X[included])).fit()
        pvalues = model.pvalues
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            excluded_feature = pvalues.idxmax()
            included.remove(excluded_feature)
            print(f"Removing feature: {excluded_feature} (p={worst_pval})")

        # Forward step: add one feature at a time
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, pd.DataFrame(X[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            print(f"Adding feature: {best_feature} (p={best_pval})")

        if not changed:
            break

    return included


# Apply stepwise regression
selected_features = stepwise_selection(X_train, y_train)
print("Selected Features:", selected_features)

# Final model with stepwise-selected features
X_train_selected = sm.add_constant(X_train[selected_features])
X_test_selected = sm.add_constant(X_test[selected_features])

final_model = sm.OLS(y_train, X_train_selected).fit()
print(final_model.summary())


# results_df = pd.DataFrame({
#     'Metric': ['R-Squared', 'Adjusted R-Squared', 'AIC', 'BIC', 'MSE (Train)', 'MSE (Test)'],
#     'Value': [r_squared, adj_r_squared, aic, bic, mse_train, mse_test]
# })
# print(results_df)
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
#
# # Load the data
# df = pd.read_csv("~/Retail_Transaction_Dataset.csv")
#
# # Data Preprocessing
# df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
# df['TotalAmount'] = df['Quantity'] * df['Price']  # Calculate Total Amount if needed
#
# # Remove NaN values
# df = df.dropna()
#
# # Feature Engineering
# df['Year'] = df['TransactionDate'].dt.year
# df['Month'] = df['TransactionDate'].dt.month
#
# # Calculate average price per product
# average_price_per_product = df.groupby('ProductID')['Price'].mean().reset_index()
# average_price_per_product.rename(columns={'Price': 'AveragePrice'}, inplace=True)
#
# # Track sales frequency
# sales_frequency = df['ProductID'].value_counts().reset_index()
# sales_frequency.columns = ['ProductID', 'SalesFrequency']
#
# # Merging Average Price and Sales Frequency
# product_analysis = pd.merge(average_price_per_product, sales_frequency, on='ProductID')
#
# # Display the product analysis
# print("Product Analysis:")
# print(product_analysis)
#
# # Preparing data for Multiple Linear Regression
# features = df[['CustomerID', 'ProductID', 'Quantity', 'Price', 'Year', 'Month']]
# X = pd.get_dummies(features, drop_first=True)  # Convert categorical features to dummy variables
# y = df['TotalAmount']
#
# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Predictions
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"\nMean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")
#
# # Visualize predictions versus actual sales
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test, y=y_pred)
# plt.xlabel('Actual Total Amount')
# plt.ylabel('Predicted Total Amount')
# plt.title('Actual vs Predicted Total Amount')
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line for reference
# plt.show()
#
# # Additional Analysis: Best and Worst Selling Products
# best_selling = sales_frequency.nlargest(5, 'SalesFrequency')
# worst_selling = sales_frequency.nsmallest(5, 'SalesFrequency')
#
# print("\nBest Selling Products:")
# print(best_selling)
#
# print("\nWorst Selling Products:")
# print(worst_selling)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # Load the data
# df = pd.read_csv("~/Retail_Transaction_Dataset.csv")
#
# # Data Preprocessing
# df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
# df['TotalAmount'] = df['Quantity'] * df['Price']  # Calculate Total Amount if needed
#
# # Remove NaN values
# df = df.dropna()
#
# # Feature Engineering
# df['Year'] = df['TransactionDate'].dt.year
# df['Month'] = df['TransactionDate'].dt.month
#
# # Calculate average price per product
# average_price_per_product = df.groupby('ProductID')['Price'].mean().reset_index()
# average_price_per_product.rename(columns={'Price': 'AveragePrice'}, inplace=True)
#
# # Track sales frequency
# sales_frequency = df['ProductID'].value_counts().reset_index()
# sales_frequency.columns = ['ProductID', 'SalesFrequency']
#
# # Merging Average Price and Sales Frequency
# product_analysis = pd.merge(average_price_per_product, sales_frequency, on='ProductID')
#
# # Display the product analysis
# print("Product Analysis:")
# print(product_analysis)
#
# # Preparing data for Random Forest Regression
# features = df[['CustomerID', 'ProductID', 'Quantity', 'Price', 'Year', 'Month']]
# X = pd.get_dummies(features, drop_first=True)  # Convert categorical features to dummy variables
# y = df['TotalAmount']
#
# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train the Random Forest Model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# # Predictions
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"\nMean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")
#
# # Visualize predictions versus actual sales
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test, y=y_pred)
# plt.xlabel('Actual Total Amount')
# plt.ylabel('Predicted Total Amount')
# plt.title('Actual vs Predicted Total Amount with Random Forest')
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line for reference
# plt.show()
#
# # Additional Analysis: Best and Worst Selling Products
# best_selling = sales_frequency.nlargest(5, 'SalesFrequency')
# worst_selling = sales_frequency.nsmallest(5, 'SalesFrequency')
#
# print("\nBest Selling Products:")
# print(best_selling)
#
# print("\nWorst Selling Products:")
# print(worst_selling)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # Load the data
# df = pd.read_csv("~/Retail_Transaction_Dataset.csv")
#
#
# # Data Preprocessing
# df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
# df['TotalAmount'] = df['Quantity'] * df['Price']  # Calculate Total Amount
#
# # Resample to daily data and calculate total revenue
# df.set_index('TransactionDate', inplace=True)
# daily_revenue = df['TotalAmount'].resample('D').sum()
#
# # Analyze Sales Revenue and Trends
# plt.figure(figsize=(12, 6))
# plt.plot(daily_revenue, label='Daily Revenue', color='blue')
# plt.title('Daily Revenue Over Time')
# plt.xlabel('Date')
# plt.ylabel('Total Revenue')
# plt.legend()
# plt.grid()
# plt.show()
#
# # Calculate Growth Rates
# daily_growth_rate = daily_revenue.pct_change().fillna(0)
# plt.figure(figsize=(12, 6))
# plt.plot(daily_growth_rate, label='Daily Growth Rate', color='green')
# plt.title('Daily Growth Rate Over Time')
# plt.xlabel('Date')
# plt.ylabel('Growth Rate')
# plt.legend()
# plt.grid()
# plt.show()
#
# # Monitor Transaction Volumes
# daily_transaction_volume = df['Quantity'].resample('D').sum()
# plt.figure(figsize=(12, 6))
# plt.plot(daily_transaction_volume, label='Daily Transaction Volume', color='orange')
# plt.title('Daily Transaction Volume Over Time')
# plt.xlabel('Date')
# plt.ylabel('Total Transactions')
# plt.legend()
# plt.grid()
# plt.show()
#
# # Analyze Seasonal Patterns
# monthly_revenue = df['TotalAmount'].resample('M').sum()
# plt.figure(figsize=(12, 6))
# monthly_revenue.plot(kind='bar', label='Monthly Revenue', color='purple')
# plt.title('Monthly Revenue Analysis')
# plt.xlabel('Month')
# plt.ylabel('Total Revenue')
# plt.legend()
# plt.grid()
# plt.show()
#
# # Preparing data for Random Forest Regression
# features = df[['CustomerID', 'ProductID', 'Quantity', 'Price']]
# X = pd.get_dummies(features, drop_first=True)  # Convert categorical features to dummy variables
# y = df['TotalAmount']
#
# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train the Random Forest Model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# # Predictions
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"\nMean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")
#
# # Make Future Predictions (e.g., for the next month)
# future_dates = pd.date_range(start=daily_revenue.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
# future_data = pd.DataFrame(index=future_dates)
#
# # Assuming average transactions, you'll need to create a dummy feature set
# # Replace with actual data or estimation for better predictions
# dummy_features = {
#     'quantity': [1] * 30,
#     'customer_id': [0] * 30,
#     'product_id': [0] * 30,
# }
#
# future_df = pd.DataFrame(dummy_features)
# future_df = pd.get_dummies(future_df, drop_first=True)
#
# # Predict future sales
# future_predictions = model.predict(future_df)
#
# # Store predictions in a DataFrame
# future_revenue_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['PredictedRevenue'])
#
# # Plot Future Predictions
# plt.figure(figsize=(12, 6))
# plt.plot(daily_revenue, label='Historical Revenue', color='blue')
# plt.plot(future_revenue_df, label='Predicted Future Revenue', color='red')
# plt.title('Historical and Predicted Revenue')
# plt.xlabel('Date')
# plt.ylabel('Revenue')
# plt.legend()
# plt.grid()
# plt


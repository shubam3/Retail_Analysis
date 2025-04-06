import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# Initialize PrettyTable for Complete Classifier Comparison
final_comparison_table = PrettyTable()
final_comparison_table.title = "Complete Classifier Comparison"
final_comparison_table.field_names = [
    "Model", "Precision", "Recall", "Specificity", "F-score", "AUC", "Confusion Matrix"
]

# Load the dataset
file_path = "~/Retail_Transaction_Dataset.csv"
df = pd.read_csv(file_path)

# Convert TransactionDate to datetime
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
# Extract time-based features
df['TransactionMonth'] = df['TransactionDate'].dt.month
df['TransactionDay'] = df['TransactionDate'].dt.day
df['TransactionHour'] = df['TransactionDate'].dt.hour
df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek

# Encode categorical variables
label_encoders = {}

# Create binary target variable
df['TransactionType'] = pd.qcut(df['TotalAmount'], q=2, labels=['Low', 'High'])
df['TransactionType'] = LabelEncoder().fit_transform(df['TransactionType'])

class_distribution = df['TransactionType'].value_counts()
print(class_distribution)

class_distribution.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Define feature matrix and target variable
features = ['Quantity', 'Price', 'DiscountApplied(%)','TransactionMonth','TransactionDay','TransactionHour','DayOfWeek']
X = df[features]
y = df['TransactionType']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=5805)
rf_model.fit(X_train, y_train)

# Evaluate feature importance
feature_importances = rf_model.feature_importances_
important_features = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Predict and evaluate the model
y_pred = rf_model.predict(X_test)
classification_report_result = classification_report(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.barh(important_features['Feature'], important_features['Importance'], color='b')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance from Random Forest')
plt.grid(axis='x')
plt.show()

top_features = important_features.head(3)['Feature'].tolist()
X_selected = X[top_features]

# Splitting the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=5805,stratify=y)

# Grid Search for Pre-Pruning with Additional Parameters
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.001, 0.01, 0.1, 0.2]  # Cost complexity for post-pruning

}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=5805),
    param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Fit Grid Search for Pre-Pruning
grid_search.fit(X_train, y_train)

# Retrieve the best pre-pruned model
best_pre_pruned_model = grid_search.best_estimator_

# Post-Pruning: Optimize Cost Complexity Parameter (ccp_alpha)
path = best_pre_pruned_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Evaluate models with different ccp_alpha values
pruned_scores = []
for alpha in ccp_alphas:
    dt_model_pruned = DecisionTreeClassifier(random_state=5805, ccp_alpha=alpha)
    dt_model_pruned.fit(X_train, y_train)
    pruned_scores.append(dt_model_pruned.score(X_test, y_test))

# Select the best ccp_alpha
best_ccp_alpha = ccp_alphas[np.argmax(pruned_scores)]
best_post_pruned_model = DecisionTreeClassifier(random_state=5805, ccp_alpha=best_ccp_alpha)
best_post_pruned_model.fit(X_train, y_train)


# Function to plot Decision Tree
def plot_decision_tree(model, feature_names, title):
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=["Low", "High"], filled=True, rounded=True)
    plt.title(title)
    plt.show()

train_acc_pre = best_pre_pruned_model.score(X_train, y_train)
test_acc_pre = best_pre_pruned_model.score(X_test, y_test)

print("Pre-Pruning Results:")
print(f"Accuracy: {test_acc_pre:.4f}")
plot_decision_tree(best_pre_pruned_model, features, title="Pre-Pruned Decision Tree")

# Initialize PrettyTable for Pre-Pruned classifier comparison
pre_pruned_table = PrettyTable()
pre_pruned_table.title = "Classifier Comparison"
pre_pruned_table.field_names = ["Model", "Precision", "Recall", "Specificity", "F-score","Auc", "Confusion Matrix"]

y_pred_pre = best_pre_pruned_model.predict(X_test)

# Performance metrics for Pre-Pruned Model
conf_matrix_pre = confusion_matrix(y_test, y_pred_pre)
report_pre = classification_report(y_test, y_pred_pre, output_dict=True)
tn, fp, fn, tp = conf_matrix_pre.ravel()
specificity_pre = round(tn / (tn + fp), 4)
precision_pre = round(report_pre["1"]["precision"], 4)
recall_pre = round(report_pre["1"]["recall"], 4)
f1_score_pre = round(report_pre["1"]["f1-score"], 4)
fpr, tpr, _ = roc_curve(y_test, y_pred_pre)
auc_score = round(auc(fpr, tpr),4)
sns.heatmap(conf_matrix_pre, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Append results to PrettyTable for Pre-Pruned Model
pre_pruned_table.add_row([
    "Pre-Pruned Decision Tree",
    precision_pre,
    recall_pre,
    specificity_pre,
    f1_score_pre,
    auc_score,
    conf_matrix_pre.tolist(),

])
print(pre_pruned_table)
final_comparison_table.add_row(["Pre-Pruned Decision Tree", precision_pre, recall_pre,
                                specificity_pre, f1_score_pre, auc_score, conf_matrix_pre.tolist()])

# Post-Pruning (Cost Complexity Optimization)
path = best_pre_pruned_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# Evaluate models with different ccp_alpha values
pruned_models = []
train_acc_post = []
test_acc_post = []

for alpha in ccp_alphas:
    model = DecisionTreeClassifier(random_state=5805, ccp_alpha=alpha)
    model.fit(X_train, y_train)
    pruned_models.append(model)
    train_acc_post.append(model.score(X_train, y_train))
    test_acc_post.append(model.score(X_test, y_test))

# Select the best post-pruned model
best_ccp_alpha_idx = np.argmax(test_acc_post)
best_post_pruned_model = pruned_models[best_ccp_alpha_idx]

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_acc_post, label='Train Accuracy', marker='o')
plt.plot(ccp_alphas, test_acc_post, label='Test Accuracy', marker='o')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Alpha for Pruned Decision Tree')
plt.legend()
plt.grid()
plt.show()

print("\nPost-Pruning Results:")
print(f"Best ccp_alpha: {ccp_alphas[best_ccp_alpha_idx]:.4f}")
print(f"Accuracy: {test_acc_post[best_ccp_alpha_idx]:.4f}")
plot_decision_tree(best_post_pruned_model, features, title="Post-Pruned Decision Tree")

# Initialize PrettyTable for Post-Pruned classifier comparison
post_pruned_table = PrettyTable()
post_pruned_table.title = "Classifier Comparison"
post_pruned_table.field_names = ["Model", "Precision", "Recall",
                                 "Specificity", "F-score","Auc" ,"Confusion Matrix"]

# Evaluate Final Post-Pruned Model
y_pred_post = best_post_pruned_model.predict(X_test)

# Performance metrics
conf_matrix_post = confusion_matrix(y_test, y_pred_post)
report_post = classification_report(y_test, y_pred_post, output_dict=True)
tn, fp, fn, tp = conf_matrix_post.ravel()
specificity_post = round(tn / (tn + fp), 4)
precision_post = round(report_post["1"]["precision"], 4)
recall_post = round(report_post["1"]["recall"], 4)
f1_score_post = round(report_post["1"]["f1-score"], 4)
fpr, tpr, _ = roc_curve(y_test, y_pred_post)
auc_score = round(auc(fpr, tpr),4)
sns.heatmap(conf_matrix_post, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'],
            yticklabels=['Low', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Display Final Results
print("\nFinal Post-Pruned Results:")

# Append results to PrettyTable for Post-Pruned Model
post_pruned_table.add_row([
    "Post-Pruned Decision Tree",
    precision_post,
    recall_post,
    specificity_post,
    f1_score_post,
    auc_score,
    conf_matrix_post.tolist()
])

# Print Post-Pruned Classifier Comparison Table
print(post_pruned_table)
final_comparison_table.add_row(["Post-Pruned Decision Tree", precision_post, recall_post,
                                specificity_post, f1_score_post, auc_score, conf_matrix_post.tolist()])

# Stratified K-Fold Cross Validation for Final Model
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)
cv_scores_pre = cross_val_score(best_pre_pruned_model, X, y, cv=stratified_kfold, scoring='accuracy')
cv_scores_post = cross_val_score(best_post_pruned_model, X, y, cv=stratified_kfold, scoring='accuracy')

# Initialize PrettyTable for Stratified K-fold results
kfold_table_pre = PrettyTable()
kfold_table_pre.title = "Stratified K-Fold Comparison"
kfold_table_pre.field_names = ["Model", "Accuracies", "Max Accuracy", "Min Accuracy", "Avg Accuracy"]

# Append K-fold results to PrettyTable ]
kfold_table_pre.add_row([
    "Pre-Pruned Decision Tree",
    list(cv_scores_pre),
    f"{cv_scores_pre.max() * 100:.2f}",
    f"{cv_scores_pre.min() * 100:.2f}",
    f"{cv_scores_pre.mean() * 100:.2f}"
])

# Print Stratified K-Fold Results Table
print(kfold_table_pre)

# Initialize PrettyTable for Stratified K-fold results
kfold_table_post = PrettyTable()
kfold_table_post.title = "Stratified K-Fold Comparison"
kfold_table_post.field_names = ["Model", "Accuracies", "Max Accuracy", "Min Accuracy", "Avg Accuracy"]

# Append K-fold results to PrettyTable for Post-Pruned Model
kfold_table_post.add_row([
    "Post-Pruned Decision Tree",
    list(cv_scores_post),
    f"{cv_scores_post.max() * 100:.2f}",
    f"{cv_scores_post.min() * 100:.2f}",
    f"{cv_scores_post.mean() * 100:.2f}"
])

# Print Stratified K-Fold Results Table for Post-Pruned Model
print(kfold_table_post)
plt.figure(figsize=(10, 8))
# Pre-Pruned Model
y_pred_prob_pre = best_pre_pruned_model.predict_proba(X_test)[:, 1]
fpr_pre, tpr_pre, _ = roc_curve(y_test, y_pred_prob_pre)
roc_auc_pre = auc(fpr_pre, tpr_pre)
plt.plot(fpr_pre, tpr_pre, label=f"Pre-Pruned (AUC = {roc_auc_pre:.2f})", linestyle='--')

# Post-Pruned Model
y_pred_prob_post = best_post_pruned_model.predict_proba(X_test)[:, 1]
fpr_post, tpr_post, _ = roc_curve(y_test, y_pred_prob_post)
roc_auc_post = auc(fpr_post, tpr_post)
plt.plot(fpr_post, tpr_post, label=f"Post-Pruned (AUC = {roc_auc_post:.2f})", linestyle='-.')

# Cost Complexity Optimized Model
best_ccp_alpha_idx = np.argmax(test_acc_post)  # Index of the best alpha from pruning
ccp_alpha_model = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alphas[best_ccp_alpha_idx])
ccp_alpha_model.fit(X_train, y_train)
y_pred_prob_ccp = ccp_alpha_model.predict_proba(X_test)[:, 1]
fpr_ccp, tpr_ccp, _ = roc_curve(y_test, y_pred_prob_ccp)
roc_auc_ccp = auc(fpr_ccp, tpr_ccp)
plt.plot(fpr_ccp, tpr_ccp, label=f"Cost Complexity Optimized (AUC = {roc_auc_ccp:.2f})", linestyle='-')

# Add plot formatting
plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison for Pre-Pruning, Post-Pruning, and Cost Complexity Optimization")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


#Logistic regression
from sklearn.linear_model import LogisticRegression

# Define logistic regression and hyperparameter grid
logreg = LogisticRegression(random_state=5805, solver='liblinear')
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2']         # L1 and L2 regularization
}

# Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=logreg,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, y_train)

# Retrieve the best model
best_model = grid_search.best_estimator_
print("Best Hyperparameters for logistic regression:", grid_search.best_params_)

y_pred = best_model.predict(X_test)
# Confusion Matrix
l_conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(l_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Compute metrics
report = classification_report(y_test, y_pred, output_dict=True)
l_precision = round(report["1"]["precision"],4)
l_recall = round(report["1"]["recall"],4)
l_f1_score = round(report["1"]["f1-score"],4)
l_specificity = round(l_conf_matrix[0, 0] / (l_conf_matrix[0, 0] + l_conf_matrix[0, 1]) ,4) # TN / (TN + FP)
l_accuracy = round((l_conf_matrix[0, 0] + l_conf_matrix[1, 1]) / l_conf_matrix.sum(),4)

# Predict probabilities for ROC and AUC
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
l_roc_auc = round(auc(fpr, tpr),4)

logistic_table = PrettyTable()
logistic_table.title = "Classifier Comparison"
logistic_table.field_names = ["Model", "Precision", "Recall", "Specificity", "F-score","Auc", "Confusion Matrix"]

# Append results to PrettyTable for Pre-Pruned Model
logistic_table.add_row([
    "Logistic Regression",
    l_precision,
    l_recall,
    l_specificity,
    l_f1_score,
    l_roc_auc,
    l_conf_matrix.tolist(),

])
print("Accuracy:", l_accuracy)
print(logistic_table)
final_comparison_table.add_row(["Logistic Regression", l_precision, l_recall,
                                l_specificity, l_f1_score, l_roc_auc, l_conf_matrix.tolist()])

stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)
cv_scores_l = cross_val_score(best_model, X, y, cv=stratified_kfold, scoring='accuracy')

kfold_table_l = PrettyTable()
kfold_table_l.title = "Stratified K-Fold Comparison"
kfold_table_l.field_names = ["Model", "Accuracies", "Max Accuracy", "Min Accuracy", "Avg Accuracy"]

# Append K-fold results to PrettyTable for logistic k-fold
kfold_table_l.add_row([
    "Logistic Regression",
    list(cv_scores_l),
    f"{cv_scores_l.max() * 100:.2f}",
    f"{cv_scores_l.min() * 100:.2f}",
    f"{cv_scores_l.mean() * 100:.2f}"
])
# Print Stratified K-Fold Results Table  for logistic regression
print(kfold_table_l)
# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {l_roc_auc:.2f})", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
print(f"Stratified K-Fold Accuracy: {cv_scores.mean():.4f}")

#KNN
from sklearn.neighbors import KNeighborsClassifier
# Define the range of k values to test
k_values = range(1, 31)  # Test k from 1 to 30
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)
# Track accuracy for each k
mean_accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
    mean_accuracies.append(scores.mean())

# Select the optimal k
optimal_k = k_values[np.argmax(mean_accuracies)]
print(f"Optimal k: {optimal_k}")

# Define the range of k values to test
k_values = range(1, 31)  # Test k from 1 to 30
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)

# Track accuracy for each k
mean_accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
    mean_accuracies.append(scores.mean())

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, mean_accuracies, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Cross-Validated Accuracy')
plt.grid()
plt.show()

# Select the optimal k
optimal_k = k_values[np.argmax(mean_accuracies)]
print(f"Optimal k: {optimal_k}")
# Train the model with optimal k
knn_model = KNeighborsClassifier(n_neighbors=optimal_k)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

# Confusion Matrix
k_conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(k_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Compute metrics
report = classification_report(y_test, y_pred, output_dict=True)
k_precision = round(report["1"]["precision"],4)
k_recall = round(report["1"]["recall"],4)
k_f1_score = round(report["1"]["f1-score"],4)
k_accuracy = round(knn_model.score(X_test, y_test),4)
k_specificity = round(k_conf_matrix[0, 0] / (k_conf_matrix[0, 0] + k_conf_matrix[0, 1]),4)  # TN / (TN + FP)

# Predict probabilities for ROC and AUC
y_pred_prob = knn_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
k_roc_auc = round(auc(fpr, tpr),4)

KNN_table = PrettyTable()
KNN_table.title = "Classifier Comparison"
KNN_table.field_names = ["Model", "Precision", "Recall", "Specificity", "F-score","Auc", "Confusion Matrix"]

# Append results to PrettyTable for KNN Model
KNN_table.add_row([
    "KNN",
    k_precision,
    k_recall,
    k_specificity,
    k_f1_score,
    k_roc_auc,
    k_conf_matrix.tolist(),

])
print("Accuracy:", k_accuracy)
print(KNN_table)
final_comparison_table.add_row(["KNN", k_precision, k_recall,
                                k_specificity, k_f1_score, k_roc_auc, k_conf_matrix.tolist()])

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"KNN (AUC = {k_roc_auc:.2f})", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

k_cv_scores = cross_val_score(knn_model, X, y, cv=cv, scoring='accuracy')

k_table = PrettyTable()
k_table.title = "Stratified K-Fold Comparison"
k_table.field_names = ["Model", "Accuracies", "Max Accuracy", "Min Accuracy", "Avg Accuracy"]

# Append K-fold results to PrettyTable for KNN Model
k_table.add_row([
    "KNN",
    list(k_cv_scores),
    f"{k_cv_scores.max() * 100:.2f}",
    f"{k_cv_scores.min() * 100:.2f}",
    f"{k_cv_scores.mean() * 100:.2f}"
])
print(k_table)

#SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np

# Ensure binary target labels for confusion matrix visualization
target_labels = sorted(y_train.unique())

# Sampling smaller subsets
X_train_sample = X_train.sample(frac=0.001, random_state=5805)
y_train_sample = y_train.loc[X_train_sample.index]

X_test_sample = X_test.sample(frac=0.001, random_state=5805)
y_test_sample = y_test.loc[X_test_sample.index]

# Linear Kernel
print("\nTraining SVM with linear kernel...\n")
param_grid_linear = {'C': [0.1, 1], 'gamma': ['scale', 'auto']}
grid_search_linear = GridSearchCV(
    SVC(kernel="linear", probability=True, random_state=5805),
    param_grid=param_grid_linear,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=5805),
    n_jobs=-1,
    verbose=1
)
grid_search_linear.fit(X_train_sample, y_train_sample)
svm_best_linear = grid_search_linear.best_estimator_
y_pred_linear = svm_best_linear.predict(X_test_sample)
svm_conf_matrix_linear = confusion_matrix(y_test_sample, y_pred_linear, labels=target_labels)
report_linear = classification_report(y_test_sample, y_pred_linear, output_dict=True, labels=target_labels)
print("Best hyperparameters for linear kernel:", svm_best_linear)
# Polynomial Kernel
print("\nTraining SVM with polynomial kernel...\n")
param_grid_poly = {
    'C': [0.1, 1],
    'degree': [2, 3, 4],
    'coef0': [0, 1],
    'gamma': ['scale', 'auto']
}
grid_search_poly = GridSearchCV(
    SVC(kernel="poly", probability=True, random_state=5805),
    param_grid=param_grid_poly,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=5805),
    n_jobs=-1,
    verbose=1
)
grid_search_poly.fit(X_train_sample, y_train_sample)
svm_best_poly = grid_search_poly.best_estimator_
y_pred_poly = svm_best_poly.predict(X_test_sample)
svm_conf_matrix_poly = confusion_matrix(y_test_sample, y_pred_poly, labels=target_labels)
report_poly = classification_report(y_test_sample, y_pred_poly, output_dict=True, labels=target_labels)
print("Best hyperparameters for poly kernel:", svm_best_poly)

# RBF Kernel
print("\nTraining SVM with RBF kernel...\n")
param_grid_rbf = {'C': [0.1, 1], 'gamma': ['scale', 'auto']}
grid_search_rbf = GridSearchCV(
    SVC(kernel="rbf", probability=True, random_state=5805),
    param_grid=param_grid_rbf,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=5805),
    n_jobs=-1,
    verbose=1
)
grid_search_rbf.fit(X_train_sample, y_train_sample)
svm_best_rbf = grid_search_rbf.best_estimator_
y_pred_rbf = svm_best_rbf.predict(X_test_sample)
svm_conf_matrix_rbf = confusion_matrix(y_test_sample, y_pred_rbf, labels=target_labels)
report_rbf = classification_report(y_test_sample, y_pred_rbf, output_dict=True, labels=target_labels)
print("Best hyperparameters for rbf kernel:", svm_best_rbf)
# Evaluation
for kernel, best_model, y_pred, conf_matrix, report, param_grid in [
    ("linear", svm_best_linear, y_pred_linear, svm_conf_matrix_linear, report_linear, param_grid_linear),
    ("polynomial", svm_best_poly, y_pred_poly, svm_conf_matrix_poly, report_poly, param_grid_poly),
    ("rbf", svm_best_rbf, y_pred_rbf, svm_conf_matrix_rbf, report_rbf, param_grid_rbf)
]:
    svm_precision = round(report[str(target_labels[1])]["precision"], 4)
    svm_recall = round(report[str(target_labels[1])]["recall"], 4)
    svm_f1_score = round(report[str(target_labels[1])]["f1-score"], 4)
    svm_specificity = round(conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if len(target_labels) > 1 else None, 4)
    y_pred_prob = best_model.predict_proba(X_test_sample)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_sample, y_pred_prob)
    svm_roc_auc = round(auc(fpr, tpr), 4)

    # PrettyTable for results
    svm_table = PrettyTable()
    svm_table.title = f"SVM ({kernel} Kernel)"
    svm_table.field_names = ["Model", "Precision", "Recall", "Specificity", "F-score", "AUC", "Confusion Matrix"]
    svm_table.add_row([
        kernel,
        svm_precision,
        svm_recall,
        svm_specificity,
        svm_f1_score,
        svm_roc_auc,
        np.array(conf_matrix).tolist(),
    ])
    print(svm_table)
    final_comparison_table.add_row(
        [f"SVM ({kernel} Kernel)", svm_precision, svm_recall, svm_specificity, svm_f1_score, svm_roc_auc,
         conf_matrix.tolist()])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'],
                yticklabels=['Low', 'High'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    # Cross-validation
    svm_cv_scores = cross_val_score(best_model, X, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                                    scoring='accuracy')
    svm_table_kfold = PrettyTable()
    svm_table_kfold.title = f"Stratified K-Fold Comparison ({kernel} Kernel)"
    svm_table_kfold.field_names = ["Model", "Accuracies", "Max Accuracy", "Min Accuracy", "Avg Accuracy"]
    svm_table_kfold.add_row([
        kernel,
        list(svm_cv_scores),
        f"{svm_cv_scores.max() * 100:.2f}",
        f"{svm_cv_scores.min() * 100:.2f}",
        f"{svm_cv_scores.mean() * 100:.2f}"
    ])
    print(svm_table_kfold)

    if len(target_labels) == 2:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"SVM (Kernel: {kernel}, AUC = {svm_roc_auc:.2f})", linestyle='-')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f'ROC Curve ({kernel} Kernel)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

#Navie bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np

# Ensure binary target labels for confusion matrix visualization
target_labels = sorted(y_train.unique())
# Sampling smaller subsets
X_train_sample = X_train.sample(frac=0.025, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]
X_test_sample = X_test.sample(frac=0.025, random_state=42)
y_test_sample = y_test.loc[X_test_sample.index]

#GaussianNB model
gnb = GaussianNB()
# hyperparameter grid
param_grid = {'var_smoothing': np.logspace(-9, -3, 7)}

# Perform GridSearchCV
print("\nPerforming GridSearchCV for Naive Bayes...\n")
grid_search = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_sample, y_train_sample)

# Best Model from Grid Search
best_model = grid_search.best_estimator_
print(f"\nBest Parameters from GridSearchCV: {grid_search.best_params_}")

# Train the best model
best_model.fit(X_train_sample, y_train_sample)
# Make predictions
y_pred = best_model.predict(X_test_sample)

# Confusion Matrix
nb_conf_matrix = confusion_matrix(y_test_sample, y_pred, labels=target_labels)
sns.heatmap(nb_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_labels, yticklabels=target_labels)
plt.title('Confusion Matrix (Best GaussianNB)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Calculate Metrics
report = classification_report(y_test_sample, y_pred, output_dict=True, labels=target_labels)
nb_accuracy = round(best_model.score(X_test_sample, y_test_sample), 4)
nb_precision = round(report[str(target_labels[1])]["precision"], 4)
nb_recall = round(report[str(target_labels[1])]["recall"], 4)
nb_f1_score = round(report[str(target_labels[1])]["f1-score"], 4)
nb_specificity = round(nb_conf_matrix[0, 0] / (nb_conf_matrix[0, 0] + nb_conf_matrix[0, 1]), 4)

# AUC Calculation
y_pred_prob = best_model.predict_proba(X_test_sample)[:, 1]
fpr, tpr, _ = roc_curve(y_test_sample, y_pred_prob)
nb_roc_auc = round(auc(fpr, tpr), 4)

# Create PrettyTable for Evaluation
nb_table = PrettyTable()
nb_table.title = "Classifier Comparison (Naive Bayes)"
nb_table.field_names = ["Model", "Precision", "Recall", "Specificity", "F-score", "AUC", "Confusion Matrix"]

# Add metrics to the table
nb_table.add_row([
    "Naive Bayes",
    nb_precision,
    nb_recall,
    nb_specificity,
    nb_f1_score,
    nb_roc_auc,
    nb_conf_matrix.tolist()
])
print(nb_table)
final_comparison_table.add_row(["Naive Bayes", nb_precision, nb_recall,
                                nb_specificity, nb_f1_score, nb_roc_auc, nb_conf_matrix.tolist()])

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Naive Bayes (AUC = {nb_roc_auc:.2f})", linestyle='-')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve (Naive Bayes)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

print(f"AUC: {nb_roc_auc:.4f}")

# Cross-validated Accuracy
nb_cv_scores = cross_val_score(best_model, X_train_sample, y_train_sample,
                               cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                               scoring='accuracy')

# Create PrettyTable for Cross-validation
nb_cv_table = PrettyTable()
nb_cv_table.title = "Stratified K-Fold Comparison (Naive Bayes)"
nb_cv_table.field_names = ["Model", "Accuracies", "Max Accuracy", "Min Accuracy", "Avg Accuracy"]

# Add CV metrics to the table
nb_cv_table.add_row([
    "Naive Bayes",
    list(np.round(nb_cv_scores, 4)),
    f"{nb_cv_scores.max() * 100:.2f}",
    f"{nb_cv_scores.min() * 100:.2f}",
    f"{nb_cv_scores.mean() * 100:.2f}"
])
print(nb_cv_table)

# #Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np

# Sampling smaller subsets
X_train_sample = X_train
y_train_sample = y_train.loc[X_train_sample.index]
X_test_sample = X_test
y_test_sample = y_test.loc[X_test_sample.index]
target_labels = sorted(y_train.unique())

# Define Random Forest with GridSearchCV
rf_params = {
    'n_estimators': [50, 100],  # Number of trees
    'max_depth': [None, 10],    # Tree depth
    'min_samples_split': [2, 5] # Minimum samples per split
}
rf_model = GridSearchCV(
    RandomForestClassifier(random_state=5805),
    param_grid=rf_params,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=1
)

# Fit the model
print("\nTraining and Evaluating Random Forest...\n")
rf_model.fit(X_train_sample, y_train_sample)

# Predictions
y_pred = rf_model.predict(X_test_sample)

# Confusion Matrix
r_conf_matrix = confusion_matrix(y_test_sample, y_pred, labels=target_labels)
sns.heatmap(r_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_labels, yticklabels=target_labels)
plt.title('Confusion Matrix (Random Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report and Metrics
report = classification_report(y_test_sample, y_pred, output_dict=True, labels=target_labels)
r_accuracy = round(rf_model.score(X_test_sample, y_test_sample), 4)
r_precision = round(report[str(target_labels[1])]["precision"], 4)
r_recall = round(report[str(target_labels[1])]["recall"], 4)
r_f1_score = round(report[str(target_labels[1])]["f1-score"], 4)
r_specificity = round(r_conf_matrix[0, 0] / (r_conf_matrix[0, 0] + r_conf_matrix[0, 1]), 4)

# Calculate AUC
y_pred_prob = rf_model.predict_proba(X_test_sample)[:, 1]
fpr, tpr, _ = roc_curve(y_test_sample, y_pred_prob)
r_roc_auc = round(auc(fpr, tpr), 4)

# Create PrettyTable for evaluation
r_table = PrettyTable()
r_table.title = "Classifier Comparison (Random Forest)"
r_table.field_names = ["Model", "Precision", "Recall", "Specificity", "F-score", "AUC", "Confusion Matrix"]

r_table.add_row([
    "Random Forest",
    r_precision,
    r_recall,
    r_specificity,
    r_f1_score,
    r_roc_auc,
    r_conf_matrix.tolist()
])

print(r_table)
final_comparison_table.add_row(["Random Forest", r_precision, r_recall,
                                r_specificity, r_f1_score, r_roc_auc, r_conf_matrix.tolist()])

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {r_roc_auc:.2f})", linestyle='-')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve (Random Forest)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

print(f"AUC: {r_roc_auc:.4f}")

# Cross-validated accuracy
r_cv_scores = cross_val_score(rf_model, X_train_sample, y_train_sample,
                              cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=5805), scoring='accuracy')

# Create PrettyTable for cross-validation results
r_cv_table = PrettyTable()
r_cv_table.title = "Stratified K-Fold Comparison (Random Forest)"
r_cv_table.field_names = ["Model", "Accuracies", "Max Accuracy", "Min Accuracy", "Avg Accuracy"]

r_cv_table.add_row([
    "Random Forest",
    list(np.round(r_cv_scores, 4)),
    f"{r_cv_scores.max() * 100:.2f}",
    f"{r_cv_scores.min() * 100:.2f}",
    f"{r_cv_scores.mean() * 100:.2f}"
])
print(r_cv_table)

# Neural network
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# Check class balance
print("Class distribution in y_train_sample:")
print(y_train_sample.value_counts())
print("Class distribution in y_test_sample:")
print(y_test_sample.value_counts())
# Apply SMOTE for balancing
smote = SMOTE(random_state=5805)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_sample, y_train_sample)

# Scale the data
scaler = StandardScaler()
X_train_sample_scaled = scaler.fit_transform(X_train_balanced)
X_test_sample_scaled = scaler.transform(X_test_sample)

# Define hyperparameter grid
param_grid = {
    'hidden_layer_sizes': [(50,), (100,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001],
    'learning_rate': ['adaptive'],
    'learning_rate_init': [0.001]
}

# GridSearchCV
grid_search = GridSearchCV(
    MLPClassifier(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10),
    param_grid=param_grid,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=5805),
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train_sample_scaled, y_train_balanced)

# Best model and parameters
best_model = grid_search.best_estimator_
print("Best Parameters (MLP):", grid_search.best_params_)
# Predictions and probabilities
y_pred = best_model.predict(X_test_sample_scaled)
y_pred_prob = best_model.predict_proba(X_test_sample_scaled)[:, 1]

# Precision-recall adjustment
precision, recall, thresholds = precision_recall_curve(y_test_sample, y_pred_prob)
optimal_idx = np.argmax(2 * precision * recall / (precision + recall))
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.4f}")

# Apply optimal threshold
y_pred_adjusted = (y_pred_prob >= optimal_threshold).astype(int)
# Confusion Matrix
nn_conf_matrix = confusion_matrix(y_test_sample, y_pred_adjusted)
sns.heatmap(nn_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y_train.unique()), yticklabels=sorted(y_train.unique()))
plt.title('Confusion Matrix (MLP)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Metrics calculation
report = classification_report(y_test_sample, y_pred_adjusted, output_dict=True)
nn_accuracy = round(best_model.score(X_test_sample_scaled, y_test_sample), 4)
nn_precision_score = round(report['1']["precision"] if "1" in report else None, 4)
nn_recall_score = round(report['1']["recall"] if "1" in report else None, 4)
nn_f1_score = round(report['1']["f1-score"] if "1" in report else None, 4)
nn_specificity = round(nn_conf_matrix[0, 0] / (nn_conf_matrix[0, 0] + nn_conf_matrix[0, 1]), 4)
fpr, tpr, _ = roc_curve(y_test_sample, y_pred_prob)
nn_roc_auc = round(auc(fpr, tpr), 4)
print("Accuracy: ", nn_accuracy)

# PrettyTable for Classifier Comparison
nn_table = PrettyTable()
nn_table.title = "Classifier Comparison"
nn_table.field_names = ["Model", "Precision", "Recall", "Specificity", "F-score", "AUC", "Confusion Matrix"]

nn_table.add_row([
    "Neural Network", nn_precision_score, nn_recall_score,
    nn_specificity, nn_f1_score, nn_roc_auc, nn_conf_matrix.tolist()
])
print("Accuracy:", nn_accuracy)
print(nn_table)
final_comparison_table.add_row(["Neural Network", nn_precision_score,
                                nn_recall_score, nn_specificity, nn_f1_score, nn_roc_auc, nn_conf_matrix.tolist()])

# ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"MLP (AUC = {nn_roc_auc:.2f})", linestyle='-')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve (MLP)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Cross-validated accuracy
nn_cv_scores = cross_val_score(best_model, X_train_sample_scaled,
                               y_train_balanced, cv=StratifiedKFold(n_splits=3,
                                                shuffle=True, random_state=5805), scoring='accuracy')

# Cross-Validation PrettyTable
nn_table_cv = PrettyTable()
nn_table_cv.title = "Stratified K-Fold Comparison"
nn_table_cv.field_names = ["Model", "Accuracies", "Max Accuracy", "Min Accuracy", "Avg Accuracy"]

nn_table_cv.add_row([
    "Neural Network", list(nn_cv_scores), f"{nn_cv_scores.max() * 100:.2f}",
    f"{nn_cv_scores.min() * 100:.2f}", f"{nn_cv_scores.mean() * 100:.2f}"
])
print(nn_table_cv)
# Print the final table
print(final_comparison_table)

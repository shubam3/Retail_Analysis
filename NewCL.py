import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from prettytable import PrettyTable


# Load the dataset
file_path = "~/Retail_Transaction_Dataset.csv"
df = pd.read_csv(file_path)

# Feature Engineering: Extracting time-based features
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['TransactionMonth'] = df['TransactionDate'].dt.month
df['TransactionDay'] = df['TransactionDate'].dt.day
df['TransactionHour'] = df['TransactionDate'].dt.hour
df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek

# Encoding categorical variables
label_encoders = {}
categorical_columns = ['ProductID', 'PaymentMethod', 'StoreLocation', 'ProductCategory']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Creating a binary target variable: High vs. Low transactions
df['TransactionType'] = pd.qcut(df['TotalAmount'], q=2, labels=['Low', 'High'])
df['TransactionType'] = LabelEncoder().fit_transform(df['TransactionType'])

class_distribution = df['TransactionType'].value_counts()
print(class_distribution)

class_distribution.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

imbalance_ratio = class_distribution.max() / class_distribution.min()
print(f'Imbalance Ratio: {imbalance_ratio}')
# Feature matrix (X) and target variable (y)
features = [
    'Quantity', 'Price', 'DiscountApplied(%)',
    'TransactionMonth', 'TransactionDay', 'TransactionHour', 'DayOfWeek'
]
X = df[features]
y = df['TransactionType']

# Splitting the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

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

# Import necessary libraries
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

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
pre_pruned_table.field_names = ["Model", "Precision", "Recall", "Specificity", "F-score", "Confusion Matrix"]


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

# Evaluate Final Post-Pruned Model
y_pred = best_post_pruned_model.predict(X_test)

# Performance metrics
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)
precision = report["1"]["precision"]
recall = report["1"]["recall"]
f1_score = report["1"]["f1-score"]
y_pred_prob = best_post_pruned_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Display Final Results
print("\nFinal Post-Pruned Results:")
print(f"Specificity: {specificity:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Stratified K-Fold Cross Validation for Final Model
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5805)
cv_scores = cross_val_score(best_post_pruned_model, X, y, cv=stratified_kfold, scoring='accuracy')

print(f"Stratified K-Fold Accuracy: {cv_scores.mean():.4f}")

# Initialize the plot
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
plt.plot([0, 1], [0, 1], "k--", label="Random Guessing")  # Diagonal line
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
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Compute metrics
report = classification_report(y_test, y_pred, output_dict=True)
precision = report["1"]["precision"]
recall = report["1"]["recall"]
f1_score = report["1"]["f1-score"]
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # TN / (TN + FP)
accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Predict probabilities for ROC and AUC
y_pred_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

print("AUC:", roc_auc)


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


# Predictions
y_pred = knn_model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Compute metrics
report = classification_report(y_test, y_pred, output_dict=True)
precision = report["1"]["precision"]
recall = report["1"]["recall"]
f1_score = report["1"]["f1-score"]
accuracy = knn_model.score(X_test, y_test)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])  # TN / (TN + FP)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Predict probabilities for ROC and AUC
y_pred_prob = knn_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"KNN (AUC = {roc_auc:.2f})", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

print(f"AUC: {roc_auc:.4f}")

cv_scores = cross_val_score(knn_model, X, y, cv=cv, scoring='accuracy')
print(f"Stratified K-Fold Accuracy: {cv_scores.mean():.4f}")

#SVM

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure binary target labels for confusion matrix visualization
target_labels = sorted(y_train.unique())

# Sampling smaller subsets
X_train_sample = X_train.sample(frac=0.001, random_state=5805)  # 0.1% of the data
y_train_sample = y_train.loc[X_train_sample.index]

X_test_sample = X_test.sample(frac=0.001, random_state=5805)    # 0.1% of the test data
y_test_sample = y_test.loc[X_test_sample.index]

# Function to train and evaluate SVM for a given kernel
def svm_classifier(kernel_type, param_grid):
    print(f"\nTraining SVM with {kernel_type} kernel...\n")



    # Define SVM with GridSearchCV
    grid_search = GridSearchCV(
        SVC(kernel=kernel_type, probability=True, random_state=5805),
        param_grid=param_grid,
        scoring='accuracy',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=5805),  # Reduced to 3 splits for faster computation
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    grid_search.fit(X_train_sample, y_train_sample)

    # Retrieve the best model
    best_model = grid_search.best_estimator_
    print(f"Best Parameters ({kernel_type} Kernel):", grid_search.best_params_)

    # Predict on the test set
    y_pred = best_model.predict(X_test_sample)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_sample, y_pred, labels=target_labels)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_labels, yticklabels=target_labels)
    plt.title(f'Confusion Matrix ({kernel_type} Kernel)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Compute Metrics
    report = classification_report(y_test_sample, y_pred, output_dict=True, labels=target_labels)
    accuracy = grid_search.best_score_
    precision = report[str(target_labels[1])]["precision"]
    recall = report[str(target_labels[1])]["recall"]
    f1_score = report[str(target_labels[1])]["f1-score"]
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if len(target_labels) > 1 else None

    # Print Metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    if specificity:
        print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    # Predict probabilities for ROC and AUC (if binary classification)
    if len(target_labels) == 2:
        y_pred_prob = best_model.predict_proba(X_test_sample)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_sample, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"SVM (Kernel: {kernel_type}, AUC = {roc_auc:.2f})", linestyle='-')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f'ROC Curve ({kernel_type} Kernel)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

        print(f"AUC: {roc_auc:.4f}")


    return best_model

# Linear Kernel
param_grid_linear = {'C': [0.1, 1]}
best_model_linear = svm_classifier('linear', param_grid_linear)

param_grid_poly = {
    'C': [0.1, 1],
    'degree': [2, 3, 4],  # Polynomial degree
    'coef0': [0, 1],      # Independent term in kernel function
    'gamma': ['scale', 'auto']
}
best_model_poly = svm_classifier('poly', param_grid_poly)

# Radial Basis Function (RBF) Kernel
param_grid_rbf = {
    'C': [0.1, 1],
    'gamma': ['scale', 'auto']
}

best_model_rbf = svm_classifier('rbf', param_grid_rbf)


# Navie bayes
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# Ensure binary target labels for confusion matrix visualization
target_labels = sorted(y_train.unique())

# Sampling smaller subsets
X_train_sample = X_train
y_train_sample = y_train.loc[X_train_sample.index]

X_test_sample = X_test    # 0.1% of the test data
y_test_sample = y_test.loc[X_test_sample.index]
# X_train_sample = X_train.sample(frac=0.001, random_state=42)  # 0.1% of the data
# y_train_sample = y_train.loc[X_train_sample.index]
#
# X_test_sample = X_test.sample(frac=0.001, random_state=42)    # 0.1% of the test data
# y_test_sample = y_test.loc[X_test_sample.index]

# Function to train and evaluate Naïve Bayes
def naive_bayes_classifier(nb_type):
    print(f"\nTraining Naïve Bayes Classifier ({nb_type})...\n")

    # Timing the process
    start_time = time.time()

    # Select classifier type
    if nb_type == 'GaussianNB':
        model = GaussianNB()
    elif nb_type == 'MultinomialNB':
        model = MultinomialNB()
    elif nb_type == 'BernoulliNB':
        model = BernoulliNB()
    else:
        raise ValueError("Invalid Naïve Bayes type. Choose 'GaussianNB', 'MultinomialNB', or 'BernoulliNB'.")

    # Fit the model
    model.fit(X_train_sample, y_train_sample)

    # Predict on the test set
    y_pred = model.predict(X_test_sample)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_sample, y_pred, labels=target_labels)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_labels, yticklabels=target_labels)
    plt.title(f'Confusion Matrix ({nb_type})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Compute Metrics
    report = classification_report(y_test_sample, y_pred, output_dict=True, labels=target_labels)
    accuracy = model.score(X_test_sample, y_test_sample)
    precision = report[str(target_labels[1])]["precision"]
    recall = report[str(target_labels[1])]["recall"]
    f1_score = report[str(target_labels[1])]["f1-score"]
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if len(target_labels) > 1 else None

    # Print Metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    if specificity:
        print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    # Predict probabilities for ROC and AUC (if binary classification)
    if len(target_labels) == 2:
        y_pred_prob = model.predict_proba(X_test_sample)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_sample, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"Naïve Bayes ({nb_type}, AUC = {roc_auc:.2f})", linestyle='-')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f'ROC Curve ({nb_type})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

        print(f"AUC: {roc_auc:.4f}")

    # Cross-validated accuracy
    cv_scores = cross_val_score(model, X_train_sample, y_train_sample, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='accuracy')
    print(f"Stratified K-Fold Accuracy: {cv_scores.mean():.4f}")

    return model


# Train Gaussian Naïve Bayes
best_model_gnb = naive_bayes_classifier('GaussianNB')

# Train Multinomial Naïve Bayes (if suitable for dataset)
# Uncomment the following lines if your features are non-negative counts/frequencies
# best_model_mnb = naive_bayes_classifier('MultinomialNB')

# Train Bernoulli Naïve Bayes (if suitable for dataset)
# Uncomment the following lines if your features are binary
# best_model_bnb = naive_bayes_classifier('BernoulliNB')


# • Random Forest

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# Sampling smaller subsets
X_train_sample = X_train
y_train_sample = y_train.loc[X_train_sample.index]

X_test_sample = X_test
y_test_sample = y_test.loc[X_test_sample.index]

target_labels = sorted(y_train.unique())


def model_evaluator(model, model_name):
    print(f"\nTraining and Evaluating {model_name}...\n")

    start_time = time.time()

    # Fit the model
    model.fit(X_train_sample, y_train_sample)

    # Predict on the test set
    y_pred = model.predict(X_test_sample)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_sample, y_pred, labels=target_labels)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_labels, yticklabels=target_labels)
    plt.title(f'Confusion Matrix ({model_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Classification Report and Metrics
    report = classification_report(y_test_sample, y_pred, output_dict=True, labels=target_labels)
    accuracy = model.score(X_test_sample, y_test_sample)
    precision = report[str(target_labels[1])]["precision"]
    recall = report[str(target_labels[1])]["recall"]
    f1_score = report[str(target_labels[1])]["f1-score"]
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if len(target_labels) > 1 else None

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    if specificity:
        print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    # Predict probabilities for ROC and AUC (if binary classification)
    if len(target_labels) == 2:
        y_pred_prob = model.predict_proba(X_test_sample)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_sample, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})", linestyle='-')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f'ROC Curve ({model_name})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

        print(f"AUC: {roc_auc:.4f}")

    # Cross-validated accuracy
    cv_scores = cross_val_score(model, X_train_sample, y_train_sample,
                                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=5805), scoring='accuracy')
    print(f"Stratified K-Fold Accuracy: {cv_scores.mean():.4f}")
    print(f"Execution Time: {time.time() - start_time:.2f} seconds\n")

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

model_evaluator(rf_model, "Random Forest")


# Define Bagging Classifier
bagging_model = BaggingClassifier(
    estimator=RandomForestClassifier(random_state=5805),  # Corrected from base_estimator to estimator
    n_estimators=10,                                    # Number of base learners
    max_samples=0.8,                                    # Fraction of samples for each learner
    random_state=5805
)

model_evaluator(bagging_model, "Bagging")


from sklearn.linear_model import LogisticRegression

# Define Stacking Classifier
stacking_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=5805)),
        ('gb', GradientBoostingClassifier(random_state=5805))
    ],
    final_estimator=LogisticRegression(),
    cv=3
)

model_evaluator(stacking_model, "Stacking")

# Corrected AdaBoost Implementation
adaboost_model = AdaBoostClassifier(
    estimator=RandomForestClassifier(random_state=5805),  # Corrected from base_estimator to estimator
    n_estimators=50,                                    # Number of boosting rounds
    learning_rate=1.0,
    random_state=5805
)

model_evaluator(adaboost_model, "AdaBoost")


gboost_model = GradientBoostingClassifier(
    n_estimators=50,           # Number of boosting rounds
    learning_rate=0.1,
    max_depth=3,
    random_state=5805
)

model_evaluator(gboost_model, "Gradient Boosting")


# Neural network
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from sklearn.preprocessing import StandardScaler

# Sampling smaller subsets
X_train_sample = X_train # 0.1% of the data
y_train_sample = y_train.loc[X_train_sample.index]

X_test_sample = X_test    # 0.1% of the test data
y_test_sample = y_test.loc[X_test_sample.index]

# Multi-layer Perceptron with GridSearchCV
def neural_network_classifier():
    print("\nTraining Multi-Layer Perceptron (MLP)...\n")


    # Scale the data
    scaler = StandardScaler()
    X_train_sample_scaled = scaler.fit_transform(X_train_sample)
    X_test_sample_scaled = scaler.transform(X_test_sample)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],  # Simplified architecture
        'activation': ['relu'],  # Focused on ReLU
        'solver': ['adam'],  # Efficient solver
        'alpha': [0.0001],  # Regularization
        'learning_rate': ['adaptive'],  # Adjust learning rate
        'learning_rate_init': [0.001]  # Smaller learning rate
    }

    grid_search = GridSearchCV(
        MLPClassifier(random_state=5805, max_iter=1000, early_stopping=True, n_iter_no_change=10, verbose=0),
        param_grid=param_grid,
        scoring='accuracy',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=5805),
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    grid_search.fit(X_train_sample_scaled, y_train_sample)




    # Retrieve the best model
    best_model = grid_search.best_estimator_
    print("Best Parameters (MLP):", grid_search.best_params_)



    # Predict on the test set
    y_pred = best_model.predict(X_test_sample)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_sample, y_pred, labels=sorted(y_train.unique()))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y_train.unique()), yticklabels=sorted(y_train.unique()))
    plt.title('Confusion Matrix (MLP)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Compute Metrics
    report = classification_report(y_test_sample, y_pred, output_dict=True)
    accuracy = best_model.score(X_test_sample, y_test_sample)
    precision = report['1']["precision"] if "1" in report else None
    recall = report['1']["recall"] if "1" in report else None
    f1_score = report['1']["f1-score"] if "1" in report else None
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if len(conf_matrix) > 1 else None

    # Print Metrics
    print(f"Accuracy: {accuracy:.4f}")
    if precision is not None:
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
    if specificity is not None:
        print(f"Specificity: {specificity:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    # Predict probabilities for ROC and AUC (if binary classification)
    if len(sorted(y_train.unique())) == 2:
        y_pred_prob = best_model.predict_proba(X_test_sample)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_sample, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"MLP (AUC = {roc_auc:.2f})", linestyle='-')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title('ROC Curve (MLP)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

        print(f"AUC: {roc_auc:.4f}")

    # Cross-validated accuracy
    cv_scores = cross_val_score(best_model, X_train_sample, y_train_sample, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='accuracy')
    print(f"Stratified K-Fold Accuracy: {cv_scores.mean():.4f}")


    return best_model

# Train and evaluate MLP
best_mlp_model = neural_network_classifier()

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Sampling smaller subsets
X_train_sample = X_train  # Full training data
y_train_sample = y_train.loc[X_train_sample.index]

X_test_sample = X_test  # Full test data
y_test_sample = y_test.loc[X_test_sample.index]

# Multi-layer Perceptron with GridSearchCV
def neural_network_classifier():
    print("\nTraining Multi-Layer Perceptron (MLP)...\n")

    # Check class balance
    print("Class distribution in y_train_sample:")
    print(y_train_sample.value_counts())
    print("Class distribution in y_test_sample:")
    print(y_test_sample.value_counts())

    smote = SMOTE(random_state=5805)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_sample, y_train_sample)

    # Scale the data
    scaler = StandardScaler()
    X_train_sample_scaled = scaler.fit_transform(X_train_balanced)
    X_test_sample_scaled = scaler.transform(X_test_sample)

    # Define hyperparameter grid
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],  # Simplified architecture
        'activation': ['relu'],                # Focused on ReLU
        'solver': ['adam'],                    # Efficient solver
        'alpha': [0.0001],                     # Regularization
        'learning_rate': ['adaptive'],         # Adjust learning rate
        'learning_rate_init': [0.001]          # Smaller learning rate
    }

    # Define GridSearchCV
    grid_search = GridSearchCV(
        MLPClassifier(random_state=42, max_iter=1000, early_stopping=True, n_iter_no_change=10, verbose=0),
        param_grid=param_grid,
        scoring='accuracy',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=5805),
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    grid_search.fit(X_train_sample_scaled, y_train_balanced)

    # Retrieve the best model
    best_model = grid_search.best_estimator_
    print("Best Parameters (MLP):", grid_search.best_params_)

    # Predict on the test set
    y_pred = best_model.predict(X_test_sample_scaled)
    y_pred_prob = best_model.predict_proba(X_test_sample_scaled)[:, 1]

    # Adjust decision threshold for precision-recall tradeoff
    precision, recall, thresholds = precision_recall_curve(y_test_sample, y_pred_prob)
    optimal_idx = np.argmax(2 * precision * recall / (precision + recall))  # Maximize F1-score
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    # Use the new threshold
    y_pred_adjusted = (y_pred_prob >= optimal_threshold).astype(int)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_sample, y_pred_adjusted)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y_train.unique()), yticklabels=sorted(y_train.unique()))
    plt.title('Confusion Matrix (MLP)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Compute Metrics
    report = classification_report(y_test_sample, y_pred_adjusted, output_dict=True)
    accuracy = best_model.score(X_test_sample_scaled, y_test_sample)
    precision_score = report['1']["precision"] if "1" in report else None
    recall_score = report['1']["recall"] if "1" in report else None
    f1_score = report['1']["f1-score"] if "1" in report else None
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if len(conf_matrix) > 1 else None

    # Print Metrics
    print(f"Accuracy: {accuracy:.4f}")
    if precision_score is not None:
        print(f"Precision: {precision_score:.4f}")
        print(f"Recall (Sensitivity): {recall_score:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
    if specificity is not None:
        print(f"Specificity: {specificity:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test_sample, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"MLP (AUC = {roc_auc:.2f})", linestyle='-')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curve (MLP)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    print(f"AUC: {roc_auc:.4f}")

    # Cross-validated accuracy
    cv_scores = cross_val_score(best_model, X_train_sample_scaled, y_train_balanced, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=5805), scoring='accuracy')
    print(f"Stratified K-Fold Accuracy: {cv_scores.mean():.4f}")

    return best_model

# Train and evaluate MLP
best_mlp_model = neural_network_classifier()

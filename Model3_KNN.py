import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, classification_report
import timeit

# Read the dataset
df = pd.read_csv('data/dermatology.csv', delimiter='\t')
df.replace('?', np.nan, inplace=True)
df = df.astype(float)

# handle missing values with mean
df.fillna(df.mean(), inplace=True)

# Selecting clinical and histopathological attributes
clinical_attributes = df.iloc[:, :11]  # Clinical attributes (1 to 11)
histopathological_attributes = df.iloc[:, 11:33]  # Histopathological attributes (12 to 33)

# Concatenate clinical and histopathological attributes
X = pd.concat([clinical_attributes, histopathological_attributes], axis=1)

# Target variable (disease type)
y = df.iloc[:, -1]

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Train the kNN model
k = 6  # Number of neighbors
#//code adapted from Shafi, 2023
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
#//end of adapted code

# Predict disease types
y_pred = knn.predict(X_test)
print("the predicted disease labels are",y_pred)

# Plot predicted vs true labels
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
plt.figure(figsize=(10, 8))
for i in range(1, 7):  # Loop through each disease type
    true_indices = y_test.reset_index(drop=True) == i  # Reset index to ensure correct alignment
    pred_indices = y_pred == i
    if np.any(true_indices) and np.any(pred_indices):  # Check if both true and predicted labels contain instances of this disease type
        common_indices = np.intersect1d(np.where(true_indices)[0],
        np.where(pred_indices)[0])  # Find common indices
        plt.scatter(y_test.iloc[common_indices], y_pred[common_indices],
        alpha=0.5, color=colors[i-1], label=f'Disease Type {i}')
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.title('Predicted vs True Labels')
plt.grid(True)
plt.legend()
plt.show()

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# Display confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the kNN model:", accuracy)
precision, recall, f1score,_ = precision_recall_fscore_support(
y_test, y_pred, average='weighted')
# Display evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1score)
# Compute single fold cross-validation score
cross_val_accuracy = cross_val_score(knn, X, y, cv=5).mean()
print("Single Fold Cross-Validation Accuracy:", cross_val_accuracy)
# Measure execution time
def time_knn():
    knn.fit(X_train, y_train)
execution_time = timeit.timeit(time_knn, number=1)
print(f"Execution Time: {execution_time} seconds")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#//code adapted from Shafi, 2023
#finding accuracy for different values of k = 1 to 30
k_values = [i for i in range (1,31)]
scores = []
scaler = StandardScaler()
X = scaler.fit_transform(X)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))
sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.show()
#//end of adapted code

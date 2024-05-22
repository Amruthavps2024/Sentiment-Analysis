import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import confusion_matrix
import timeit
# Read the data set and handle missing values
df = pd.read_csv(r'data/dermatology.csv', delimiter='\t')
df.head()
df.info()
df.isnull()
df.replace('?', np.nan, inplace=True)
# feature extraction of both clinical attributes and histopathological attributes
X = df.iloc[:, :-2].values
y = df.iloc[:, -1].values
print(X,y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                    y, test_size = 0.25, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Random Forest classifier
classifier = RandomForestClassifier(n_estimators=10, random_state=0)
# Train the classifier
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("The predicted disease labels are",y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
precision, recall, f1score,_ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted')
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1score)
# Compute single fold cross-validation score
cross_val_accuracy = cross_val_score(classifier, X, y, cv=5).mean()
print("Single Fold Cross-Validation Accuracy:", cross_val_accuracy)
# Measure execution time
execution_time = timeit.timeit(RandomForestClassifier, number=1)
print(f"Execution Time: {execution_time} seconds")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix 
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(
    y_test, y_pred)  # Compute confusion matrix
sns.heatmap(conf_matrix, annot=True,
            cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix of random forest classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

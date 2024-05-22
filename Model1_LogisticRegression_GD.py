import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import timeit
from sklearn.metrics import precision_score, recall_score, f1_score

class GD_LogisticRegression:
    #//code adapted from Koushik,2023
    def __init__(self, learning_rate=0.01, n_iterations=500):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Compute predicted values
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    #////end of adapted code
    def score(self, X, y):
        # Make predictions
        y_pred = self.predict(X)
        # Compute accuracy
        return accuracy_score(y, y_pred)
     
     #//code adapted from Dementiy,2017       
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred_prob = self.sigmoid(linear_model)
        y_pred = np.where(y_pred_prob >= 0.5, 1, 0)  # Convert probabilities to class labels
        return y_pred
    
    def get_params(self, deep=True):
        return {"learning_rate": self.learning_rate, "n_iterations": self.n_iterations}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
#//end of adapted code

# Read the dataset
df = pd.read_csv('data/dermatology.csv', delimiter='\t')
df.replace('?', np.nan, inplace=True)
df = df.astype(float)

# Handling missing values with mean
df.fillna(df.mean(), inplace=True)

# Selecting feature (age) and target variable (type of disease)
X = df['Age'].values.reshape(-1, 1)
y = df['Disease']

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# plotting the actual data
plt.figure()
plt.plot(X, y, '+', markersize=5,
label='Original Data', color='green')
plt.title('Actual Data')
plt.xlabel('Age')
plt.ylabel('Disease label')
plt.legend()
plt.show()

print(X.shape)
print(y.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  the logistic regression model based on gradient descent 
model = GD_LogisticRegression()
# Train the model
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
print("predicted class labels are ",y_pred)

# plotting the prediction 
plt.scatter(X[:180], y[:180], color='b',
            label='Actual')
plt.plot(X_test, y_pred, color='r', marker='x',
         label='Predicted') 
plt.xlabel('Age')
plt.ylabel('Disease')
plt.title('Logistic Regression: Age Vs Disease')
plt.legend()
plt.show()

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
# Print evaluation metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("MAE:", mae)
print("R-squared :", r_squared)
print("MSE:", mse)
print("RMSE:", rmse)
cross_val_accuracy = cross_val_score(model,
                                     X, y, cv=5).mean()
print("Single Fold Cross-Validation Accuracy:", 
      cross_val_accuracy)

# Display confusion matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.title('Confusion Matrix of Logistic Regression model using GDS')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Measure execution time
execution_time = timeit.timeit(
    GD_LogisticRegression, number=1)
print(f"Execution Time: {execution_time} seconds")
results = {}
results['Logistic Regression'] = [precision_score
                        (y_test, y_pred, average='weighted'),
            recall_score(y_test, y_pred, average='weighted'),
                f1_score(y_test, y_pred, average='weighted'),]
print("Precision,Recall and F1 score of the model is ")
print(results)

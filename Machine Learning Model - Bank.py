import pandas as pd

# Loading the dataset "bank-additional-full"
file_path=r"C:\Users\ADAM\Downloads\bank+marketing\bank-additional\bank-additional\bank-additional-full.csv"
data=pd.read_csv(file_path, sep=';')

# Display the first few rows
print(data.head())

# Checking dataset info
print(data.info())

# Checking for missing values
print(data.isnull().sum())
print("c/c: The dataset has no missing values")

# Data Preprocessing

from sklearn.preprocessing import LabelEncoder

# Separate features and target
X=data.drop('y',axis=1) # Features
y=data['y']             # Target

# Convert target to binary (yes=1, no=0)
y= LabelEncoder().fit_transform(y)

# Perform one-hot encoding for categorical features
X=pd.get_dummies(X,drop_first=True)

# Checking the preprocessed data
print(X.head())

# Spliting the data

from sklearn.model_selection import train_test_split

# Spliting the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:",X_train.shape)
print("Testing set shape:",X_test.shape)

# Building the decision tree classifier

from sklearn.tree import DecisionTreeClassifier

# Initialize and train the decision tree
model=DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Print feature importance
importance=pd.DataFrame({'Feature':X.columns,'Importance':model.feature_importances_})
print(importance.sort_values(by='Importance',ascending=False))

# Evaluating the model

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Making predictions on the test set
y_pred=model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm= confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True, fmt='d', cmap='Blues',xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualizing the decision tree

from sklearn.tree import plot_tree

# Limiting the depth of the tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)  # Set a max depth
model.fit(X_train, y_train)

# Plotting the decision tree
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, fontsize=8)
plt.title("Decision Tree Visualization")
plt.show()

# Hyperparameter tuning : Optimizing the decision tree parameters using GridSearchCV

from sklearn.model_selection import GridSearchCV 

# Define hyperparameters
param_grid={
    'criterion':['gini', 'entropy'],
    'max_depth':[5,10,15,20,None],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,5]
}

# Performing Grid Search
grid_search=GridSearchCV(DecisionTreeClassifier(random_state=42),param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters", grid_search.best_params_)

# Evaluating the best model
best_model=grid_search.best_estimator_
y_pred_best=best_model.predict(X_test)
print("\nBest Model Accuracy", accuracy_score(y_test,y_pred_best))

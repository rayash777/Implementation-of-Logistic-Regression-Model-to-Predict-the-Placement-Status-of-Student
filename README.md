# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start the program and import the required libraries.
2.Create a dataset containing student CGPA, aptitude scores, and placement status.
3.Split the dataset into training and testing sets.
4.Train the Logistic Regression model using the training data and predict placement status.
5.Evaluate the model using accuracy and classification report, then predict the placement status of a new student.
```
## Program:
```
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Dataset: CGPA and Aptitude Score
X = np.array([
    [5.5, 50],
    [6.0, 55],
    [6.2, 60],
    [6.5, 65],
    [7.0, 70],
    [7.2, 75],
    [7.5, 80],
    [8.0, 85],
    [8.5, 90],
    [9.0, 95]
])

# Target: Placement Status (0 = Not Placed, 1 = Placed)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict placement status
y_pred = model.predict(X_test)

print("Actual Placement Status:", y_test)
print("Predicted Placement Status:", y_pred)
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Predict placement for a new student
new_student = np.array([[7.8, 82]])
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("New Student Placement Status: Placed")
else:
    print("New Student Placement Status: Not Placed")
```

## Output:
<img width="666" height="312" alt="image" src="https://github.com/user-attachments/assets/292d8e58-1175-4b04-b18c-65bfaa1ff915" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

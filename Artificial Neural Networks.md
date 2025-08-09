# 🏦 Bank Customer Churn Prediction with ANN

## 🧩 Problem Statement

A bank maintains customer data containing:

- **CustomerId**
- **Surname**
- **CreditScore**
- **Geography** (Country/Region)
- **Gender**
- **Age**
- **Tenure** (years with bank)
- **Balance**
- **NumOfProducts** (number of bank products used)
- **HasCrCard** (1 = has credit card, 0 = no)
- **IsActiveMember** (1 = active, 0 = inactive)
- **EstimatedSalary**
- **Exited** (1 = left the bank, 0 = stayed)

The bank wants to **predict the probability that a customer will leave** (customer churn) using an **Artificial Neural Network (ANN)** model.

---

## 🎯 Objective

> Train an ANN to predict whether a customer will **exit** the bank based on their profile, in order to:
> - Identify high-risk customers
> - Take retention measures before they leave

---

## 📂 Dataset Example

| CustomerId | Surname | CreditScore | Geography | Gender | Age | Tenure | Balance  | NumOfProducts | HasCrCard | IsActiveMember | EstimatedSalary | Exited |
|------------|---------|-------------|-----------|--------|-----|--------|----------|---------------|-----------|----------------|-----------------|--------|
| 15634602   | Hargrave| 619         | France    | Female | 42  | 2      | 0.00     | 1             | 1         | 1              | 101348.88       | 1      |
| 15647311   | Hill    | 608         | Spain     | Female | 41  | 1      | 83807.86 | 1             | 0         | 1              | 112542.58       | 0      |
| 15619304   | Onio    | 502         | France    | Female | 42  | 8      | 159660.80| 3             | 1         | 0              | 113931.57       | 1      |

---

## 🧠 ANN Model Approach

### **1. Data Preprocessing**
- Drop **CustomerId** and **Surname** (irrelevant to churn)
- Encode categorical variables:
  - `Geography` → One-Hot Encoding
  - `Gender` → Binary Encoding
- Scale numerical features for faster ANN convergence

---

### **2. ANN Architecture**
- **Input layer**: One neuron for each feature after preprocessing  
- **Hidden layers**:
  - Dense layer with ReLU activation
  - Dropout layers for regularization
- **Output layer**:
  - 1 neuron with **sigmoid activation** for binary classification

---

### **3. Model Training**
- Loss function: **Binary Crossentropy**
- Optimizer: **Adam**
- Metric: **Accuracy**
- Training on **train set**, validation on **test set**

---

## 💻 Example Code

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = pd.read_csv('Churn_Modelling.csv')

# Preprocess
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# ANN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=6, activation='relu'))
model.add(tf.keras.layers.Dense(units=6, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
```

---

## 📊 Expected Output

- **Churn Probability** for each customer
- **Overall Accuracy, Precision, Recall, F1 Score**
- Identify **high-risk customers** for targeted retention

---

## ✅ Benefits

- **Predictive insight** into customer behavior
- **Data-driven retention strategies**
- **Improved customer lifetime value (CLV)**

---

## 🔍 Notes

- Additional features like **transaction history** or **complaints** can improve predictions
- Can be extended to **multi-class classification** if predicting churn reasons


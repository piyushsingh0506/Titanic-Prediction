import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load Dataset
df = pd.read_csv("train.csv")

# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Features and Target
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# Train Model
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    random_state=42
)

model.fit(X, y)

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("Model Saved Successfully")
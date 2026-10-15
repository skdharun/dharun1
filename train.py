import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Generate sample student data
np.random.seed(42)

n = 1000

study_hours = np.random.uniform(0, 10, n)
attendance = np.random.uniform(30, 100, n)
internal_marks = np.random.uniform(0, 100, n)

# 2. Generate Pass/Fail labels using a sample rule
score = (
    3.5 * study_hours
    + 0.25 * attendance
    + 0.25 * internal_marks
)

result = (score >= 42).astype(int)

# 3. Create a dataset
data = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "internal_marks": internal_marks,
    "result": result
})

# 4. Separate input features and target
X = data[[
    "study_hours",
    "attendance",
    "internal_marks"
]]

y = data["result"]

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Train the machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 7. Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Student Result Prediction Model")
print("--------------------------------")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Model accuracy: {accuracy:.2%}")

# 8. Save the trained model
joblib.dump(model, "model.pkl")

print("Model successfully saved as model.pkl")

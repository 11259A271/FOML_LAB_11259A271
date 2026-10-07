
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Load the CSV
data = pd.read_csv("heart.csv")

# Remove accidental spaces from column names
data.columns = data.columns.str.strip()

# Convert categorical columns into numbers
data = pd.get_dummies(data, drop_first=True)

# Inputs and target
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1
)

# Create the Gradient Boosting model
model = GradientBoostingClassifier(random_state=1)

# Train
model.fit(X_train, y_train)

# Test
print("Accuracy:", round(model.score(X_test, y_test), 3))
new_patient = [X_test.iloc[0]]                           # one new patient's test results
result = model.predict(new_patient)[0]                   # predict (1 or 0)
print("Heart disease? (1=yes, 0=no):", result) 


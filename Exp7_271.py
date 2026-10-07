from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=1
)
tree = DecisionTreeClassifier(random_state=1)
tree.fit(X_train, y_train)
forest = RandomForestClassifier(
    n_estimators=100,
    random_state=1
)
forest.fit(X_train, y_train)
print("one tree accuracy: ", round(tree.score(X_test, y_test), 3))
print("Forest accuracy: ", round(forest.score(X_test, y_test), 3))
names = data.target_names
new_tumour = X_test[0]
result = forest.predict(new_tumour.reshape(1, -1))[0]
print("Diagnosis: ", names[result])
import matplotlib.pyplot as plt          # the charting library

importances = model.feature_importances_       # how useful each feature was
names = data.feature_names
top = sorted(zip(importances, names), reverse=True)[:5]  # keep the top 5
vals = [t[0] for t in top]; labs = [t[1] for t in top]
plt.barh(labs[::-1], vals[::-1], color="#2F49D1")  # horizontal bars
plt.xlabel("importance"); plt.title("Top 5 features")
plt.show()


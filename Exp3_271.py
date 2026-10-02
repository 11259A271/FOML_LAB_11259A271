import seaborn as sns                                    # gives us the car data
from sklearn.model_selection import train_test_split     # tool that splits data
from sklearn.linear_model import LinearRegression        # the straight-line model

data = sns.load_dataset("mpg").dropna()                  # load cars, drop rows with blanks
X = data[["cylinders","displacement","horsepower","weight","acceleration","model_year"]]  # inputs
y = data["mpg"]                                          # the number we want to predict

X_train, X_test, y_train, y_test = train_test_split(     # split learn + exam
    X, y, test_size=0.2, random_state=1)

model = LinearRegression()                               # create the model
model.fit(X_train, y_train)                              # learn from the training cars

print("R2 score:", round(model.score(X_test, y_test), 3))
new_car = [[4, 120, 90, 2500, 16, 80]]                   # a small, light, modern car
mpg = model.predict(new_car)[0]                          # predict its mileage
print("Predicted mileage:", round(mpg, 1), "mpg")        
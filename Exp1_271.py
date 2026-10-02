import pandas as pd
import seaborn as sns
data = sns.load_dataset('titanic')
data = data[["survived","pclass","sex","age","fare","embarked"]]
print(data.isnull().sum())
data["age"] = data["age"].fillna(data["age"].median())
data["embarked"] = data["embarked"].fillna("s")
data = pd.get_dummies(data,columns=["embarked","sex"],dtype=int)
print(data.head())
print("Any blanks left?",data.isnull().values.any())
new = pd.DataFrame([{"pclass":2,"sex":"female","age":27,"fare":30,"embarked":"C"}])
new = pd.get_dummies(new, columns =["sex","embarked"], dtype=int)
new = new.reindex(columns=data.drop("survived",axis=1).columns,fill_value=0)
print(new)
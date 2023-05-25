import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression

%matplotlib inline

df = pd.read_csv("insurance_data.csv")
df.head()

plt.scatter(df.age,df.bought_insurance,marker='+',color='red')

tts(df[['age']],df.bought_insurance,train_size=0.9)
X_train, X_test, y_train, y_test = tts(df[['age']],df.bought_insurance,test_size=0.1)

model = LogisticRegression()
model.fit(X_train,y_train)

model.predict(X_test)
model.score(X_test,y_test)
model.predict_proba(X_test)

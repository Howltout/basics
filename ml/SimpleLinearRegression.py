import pandas as pd
import numpy as np
import matplotlib.pynet as plt
from sklearn import linear_model

df = pd.read_csv("homeprices.csv")

%matplotlib inline
plt.scatter(df.area,df.price,color='red')
plt.xlabel('area')
plt.ylabel('price')

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

reg.predict([[3300]])

reg.coef_
reg.intercept_

%matplotlib inline
plt.scatter(df.area,df.price,color='red')
plt.xlabel('area')
plt.ylabel('price')
plt.plot(df.area,reg.predict(df[['area']],color='blue')

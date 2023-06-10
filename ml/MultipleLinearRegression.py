import pandas as pd
import numpy as np
import math
from sklearn import linear_model

df = pd.read_csv('homeprices.csv')
bed_median = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.filna(bed_median, inplace = True)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

reg.coef_
reg.intercept_

reg.predict([4000,3,35]

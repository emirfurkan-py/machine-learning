import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
import numpy as np

df=pd.read_csv("polynomial_regression_dataset.csv",sep=";")

y=df.araba_max_hiz.values.reshape(-1,1)
x=df.araba_fiyat.values.reshape(-1,1)

lr=LinearRegression()

lr.fit(x,y)
y_head=lr.predict(x)

plt.scatter(x,y)
plt.xlabel("araba_max_hiz")
plt.ylabel("araba_fiyat")
plt.plot(x,y_head,color="red",label="linear")


polynomial_regression=PolynomialFeatures(degree=4)
x_polynomial=polynomial_regression.fit_transform(x)

lr2=LinearRegression()
lr2.fit(x_polynomial,y)

y_head2=lr2.predict(x_polynomial)
plt.plot(x,y_head2,color="green",label="poly")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
import numpy as np

df=pd.read_csv("multiple_linear_regression_dataset.csv",sep=";")

x=df.iloc[:,[0,2]].values
y=df.maas.values.reshape(-1,1)

mullinearregression=LinearRegression()
mullinearregression.fit(x,y)

print("bo=",mullinearregression.intercept_)
print(mullinearregression.predict([[10,35],[5,35]]))


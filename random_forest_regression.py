import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv("random_forest_regression_dataset.csv",sep=";",header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

rf=RandomForestRegressor(n_estimators=100,random_state=42)   #estimator=kaç ağaç kullanıcak

rf.fit(x,y)

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=rf.predict(x_)
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribunlevet")
plt.ylabel("ucret")
plt.show()

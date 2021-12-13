import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

df=pd.read_csv("decision_tree_regression_dataset.csv",sep=";",header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

tree_reg=DecisionTreeRegressor()
tree_reg.fit(x,y)

print(tree_reg.predict([[6]]))

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=tree_reg.predict(x_)

plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()






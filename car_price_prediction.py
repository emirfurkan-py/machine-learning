import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data=pd.read_csv("car_dataset.csv")

# print(data.head())
# print(data.info())

data.isnull().sum()

# print(data.fuel.value_counts())
# print(data.seller_type.value_counts())
# print(data.transmission.value_counts())


data.replace({'fuel':{'Petrol':0,'Diesel':1,'CNG':2,"LPG":3,"Electric":4}},inplace=True)


data.replace({'seller_type':{'Dealer':0,'Individual':1,"Trustmark Dealer":2}},inplace=True)


data.replace({'transmission':{'Manual':0,'Automatic':1}},inplace=True)

data.replace({'owner':{'First Owner':0,'Second Owner':1,"Third Owner":2,"Fourth & Above Owner":3,"Test Drive Car":4}},inplace=True)

#print(data.head())

x=data.drop(["name","selling_price"],axis=1)
y=data["selling_price"]

#print(x)
#print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2)
linear_reg_model=LinearRegression()
linear_reg_model.fit(x_train,y_train)
training_data_prediction=linear_reg_model.predict(x_train)
error_score=metrics.r2_score(y_train,training_data_prediction)
print("Rsquare error=",error_score)

plt.scatter(y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()

test_data_prediction=linear_reg_model.predict(x_test)
error_score = metrics.r2_score(y_test, test_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
print(linear_reg_model.predict(x))
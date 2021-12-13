import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

data=pd.read_csv("gold_price_dataset.csv")

data.isnull.sum()

data.describe()

correlation=data.corr()




import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')


df = pd.read_csv('netflix.csv')
df.head(6)

df.shape

plt.figure(figsize=(16,8))
plt.title('Google')
plt.xlabel('Days')
plt.ylabel('Close price USD ($)')
plt.plot(df['Close'])
plt.show()

df = df[['Close']]
df.head(4)

future_days = 25
# Create a new coloum(target) shifted 'x' unit/days up
df['Prediction'] = df[['Close']].shift(-future_days)
df.head(4)


X = np.array(df.drop(['Prediction'],1))[:-future_days]
print(X)

y = np.array(df['Prediction'])[:-future_days]
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)

#Create the decison tree regressor model
tree = DecisionTreeRegressor().fit(X_train,y_train)
#Create the linear regression model 
lr = LinearRegression().fit(X_train,y_train)

#Getting the last x rows of the feature dataset
x_future = df.drop(['Prediction'],1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future    

tree_prediction = tree.predict(x_future)
print(tree_prediction)
print()
#Show the model linear perdiction 
lr_prediction = lr.predict(x_future)
print(lr_prediction)

predictions = tree_prediction

valid = df[X.shape[0]:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close price USD ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Org', 'Val','Predicted'])
plt.show()




# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program and Output:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: RAJATHURAI K
RegisterNumber: 25016579
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('drive/MyDrive/Dataset-ML/student_scores.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/ce69963c-9e81-4531-9936-8ed0667213ad)
```
df.tail()
```
![image](https://github.com/user-attachments/assets/e8ef2d36-7987-4704-b7fc-3c532d29f7be)
```
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
X,Y
```
![image](https://github.com/user-attachments/assets/9ed3f8f2-f940-4706-8545-a1e8fd9ef941)
```
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
Y_pred,Y_test
```
![image](https://github.com/user-attachments/assets/41a501c9-f572-4f83-9666-d9c481dd37f0)
```
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('Hours VS Scores (Training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```
![image](https://github.com/user-attachments/assets/5c9a63c0-1115-4a56-9598-d4c616ad40a7)
```
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,regressor.predict(X_test),color='yellow')
plt.title('Hours VS Scores (Test set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```
![image](https://github.com/user-attachments/assets/decc794b-aa1f-48a4-9370-593ccf6721d8)
```
mse=mean_squared_error(Y_test,Y_pred)
mae=mean_absolute_error(Y_test,Y_pred)
rmse=np.sqrt(mse)
print("MSE = ",mse)
print('MAE = ',mae)
print('RMSE = ',rmse)
```
![image](https://github.com/user-attachments/assets/c82833da-b6f2-43c4-9916-68c248d3baf6)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

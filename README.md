# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: vinodhini k
RegisterNumber:  212223230245
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
````
```
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.tail())
`````
![image](https://github.com/user-attachments/assets/d8e3e0ce-6676-4e0d-bf17-ee655e12f96c)

```
df.info()
````
![image](https://github.com/user-attachments/assets/c029e34b-97ee-4a6e-8a97-1b83d491f0a0)
```
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)
`````
![image](https://github.com/user-attachments/assets/a42856cf-d358-4aa7-a7aa-0e5acacc12df)
````
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
`````
````
`x_train.shape
````````````
![image](https://github.com/user-attachments/assets/57ad5df2-58d4-4066-9e80-6eb6c82acb6a)
```

x_test.shape
`````

![image](https://github.com/user-attachments/assets/ac087b13-5dd8-44b8-b122-01e5e38ffa29)

```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
`````

![image](https://github.com/user-attachments/assets/cc27a63c-330b-497e-baaf-7ffe601cc796)
```
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
`````

![image](https://github.com/user-attachments/assets/f0f0048a-bab6-47b9-a6d9-a4be3fdc203c)

``````
`plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,reg.predict(x_train),color="blue")
plt.title("Training set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
## Output:
![simple linear regression model for predicting the marks scored](sam.png)
```````

![image](https://github.com/user-attachments/assets/cd1a9277-0c0b-456c-83b3-930a49e8d583)

```````
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg.predict(x_test),color="black")
plt.title("Test set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()`
`````


![image](https://github.com/user-attachments/assets/7e03f5d5-36bf-442a-b1b0-c0c26ac2759d)


````
mse=mean_squared_error(y_test,y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
`````````

![image](https://github.com/user-attachments/assets/0ae54bc4-6913-46b8-b803-63488fc5c7e0)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

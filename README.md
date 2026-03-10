# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.
2.Analyse the data.
3.Use modelselection and Countvectorizer to preditct the values.
4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: ASHWIN H
RegisterNumber: 212225230024 
*/
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
<img width="962" height="545" alt="image" src="https://github.com/user-attachments/assets/1cb103c1-27ae-4d7a-9735-894cdab9d3dc" />
<img width="360" height="38" alt="image" src="https://github.com/user-attachments/assets/8967f813-37e9-4bc7-a5af-6a95609b3497" />
<img width="1044" height="179" alt="image" src="https://github.com/user-attachments/assets/ddf219af-1c73-4b81-a47a-4066756b3835" />
<img width="952" height="57" alt="image" src="https://github.com/user-attachments/assets/87c086c4-b6d5-4e49-b2cb-2479755d17ce" />
<img width="907" height="241" alt="image" src="https://github.com/user-attachments/assets/6f797b11-c169-4750-a2a3-f28ad6d8fe39" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

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
<img width="963" height="547" alt="image" src="https://github.com/user-attachments/assets/7ff706fb-1978-470b-b05d-199e30d3f9da" />
<img width="137" height="34" alt="image" src="https://github.com/user-attachments/assets/b725e775-17e6-4159-8780-059bca585766" />
<img width="94" height="40" alt="image" src="https://github.com/user-attachments/assets/16a2442d-54bd-42b9-abca-904050567fb5" />
<img width="95" height="30" alt="image" src="https://github.com/user-attachments/assets/98ba4455-fcd7-4d03-900e-350bc90c0eb3" />
<img width="1043" height="164" alt="image" src="https://github.com/user-attachments/assets/d13bb4ef-fa1e-473f-9d16-642568000396" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

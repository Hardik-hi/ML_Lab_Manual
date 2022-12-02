 
#importing libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as LR 
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sms

 
#reading data
ds=np.array(pd.read_csv("logreg.csv"))
x=ds[:,:-1]
y=ds[:,-1]

x_train,x_test,y_train,y_test=tts(x,y,test_size=0.3,random_state=0)

ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.transform(x_test)

classifier=LR(random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)

print(cm,acc)

 
sms.regplot(x=x_test[:,0],y=y_pred)

 




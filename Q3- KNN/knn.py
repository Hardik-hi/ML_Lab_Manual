 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts

df=pd.read_csv("iris.csv")
data=np.array(df)

 
#splitting the test and training data
attr=data[:,:-1]
target=data[:,-1]
attr_train,attr_test,target_train,target_test=tts(attr,target,shuffle=True,random_state=0,test_size=53)


 
#function to evaluate class of a given point

def knn_classifier(i,k):
    l=[]
    #for every training data calculate the distance from the testing data
    for j,attr_row in enumerate(attr_train):
        dis=np.sqrt(sum(np.square(val1-val2) for val1,val2 in zip(attr_row,i)))
        l.append([dis,target_train[j]])

    #pick top k smallest values in the temporary list
    sorted_l=sorted(l)[:k]
    class_list=[i.pop(1) for i in sorted_l]
    return max(set(class_list),key=class_list.count)


 
#calling the function in a loop for the testing data

predicted_class=[]

for x in attr_test:
    predicted_class.append(knn_classifier(x,3))

acc=0

for i,val in enumerate(target_test):
    if val==predicted_class[i]:
        acc+=1

print(acc/len(target_test))
print(predicted_class)

 




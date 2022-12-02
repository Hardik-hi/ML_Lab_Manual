 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
import math

 
#reading the dataset
dataset=np.array(pd.read_csv("naive.csv",header=None))

 
#splitting the dataset into training and testing
t_size=int(len(dataset)*0.33)
train,test=tts(dataset,test_size=t_size,shuffle=True,random_state=0)

 
#function to summarize mean and sd of a given attribute set
def summarize(ds):
    #creates dictionary of tuples having mean and sd for the attr
    summaries=[(np.mean(attr),np.std(attr)) for attr in zip(*ds)]
    #remove target values
    del summaries[-1]
    return summaries

 
#function to separate dataset by classes
def separatebyclass(ds):
    separated={}
    for vec in ds:
        if(vec[-1] not in separated):
            separated[vec[-1]]=[]
        separated[vec[-1]].append(vec)
    return separated

 
#function to summarise the whole dataset by classes
def summarize_dataset(ds):
    separated=separatebyclass(ds)
    summaries={}
    for class_label,instances in separated.items():
        summaries[class_label]=summarize(instances) #get mean, sd of every attr row
    return summaries

 
#function to get the Gaussian probability of a given value
def gaussian_prob(x,mean,sd):
    expo=math.exp(-(math.pow(x-mean,2)/(2*sd*sd)))
    return expo/((math.sqrt(2*math.pi))*sd)

 
#function to get the probability for each class and predict the one with highest
def predict(summaries,input_row):
    #get probabilities for each class
    probabilities={}
    for class_label,summary in summaries.items():
        probabilities[class_label]=1
        for i in range(len(summary)):
            mean,sd=summary[i]
            probabilities[class_label]*=gaussian_prob(input_row[i],mean,sd)
    
    #to get the highest probability
    best_label,best_prob=None,-1
    for label,prob in probabilities.items():
        if best_label is None or prob>best_prob:
            best_label=label
            best_prob=prob
    
    return best_label

 
#function to predict labels for whole testset
def getpredictions(summaries,testset):
    predictions=[]
    for row in testset:
        predictions.append(predict(summaries,row))
    return predictions

 
#function to get preds!
def finalfun():
    #summarise training set
    summaries=summarize_dataset(train)
    predictions=getpredictions(summaries,test)

    print(predictions)

    match=0
    for i in range(len(test)):
        if predictions[i]==test[i][-1]:
            match+=1
    
    print("Accuracy: ",match/len(test)*100)

 
finalfun()

 




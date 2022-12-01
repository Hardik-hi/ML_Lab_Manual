import pandas as pd
import numpy as np
 
#to read the data in the csv file
data = pd.read_csv("find_s_dataset.csv")
 
#making an array of all the attributes
d = np.array(data)[:,:-1]
print("\n The attributes are: ",d)
 
#segragating the target that has positive and negative examples
target = np.array(data)[:,-1]
print("\n The target is: ",target)
 
#training function to implement find-s algorithm
def train(c,t):
    for i, val in enumerate(t):
        if val == "Yes":
            specific_hypothesis = c[i].copy()
            break
             
    for i, val in enumerate(c):
        if t[i] == "Yes":
            for x in range(len(specific_hypothesis)):
                if val[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
                 
    return specific_hypothesis
 
#obtaining the final hypothesis
print("\n The final hypothesis is:",train(d,target))
 
import numpy as np
import matplotlib.pyplot as plt

 
#finding coefficients
def coeff(x,y):
    num=np.sum((x-np.mean(x))*(y-np.mean(y)))
    den=np.sum(np.square(x-np.mean(x)))

    m=num/den
    c=(np.mean(y)-np.mean(x)*m)

    return [m,c]


 
#to find predicted values of y
def predicted(x,m,c):
    pred=[]
    for val in x:
        pred.append(m*val+c)
    return pred

 
#to find r square value
def r_square(y,yp):
    num=np.sum(np.square((yp-np.mean(y))))
    den=np.sum(np.square((y-np.mean(y))))

    return num/den

 
def fun():
    x = np.array([5, 1, 12, 13, 14, 15, 16, 17, 18, 19])
    y = np.array([31, 3, 12, 18, 17, 18, 18, 19, 20, 22])

    m,c=coeff(x,y)
    yp=predicted(x,m,c)
    print(yp)
    print(r_square(y,yp))
    plt.scatter(x,y,color = "r",marker = "o",s = 30)
    
    # plot the regression line
    plt.plot(x,yp)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

fun()


 




import numpy as np
from matplotlib import pyplot as plt


def linePlot(x,y,xlabel="",ylabel="",title="",axis=None):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if axis:
        plt.axis(axis)
    

#por ahora dos parámetros
def h(x,theta):
    return theta[0]+x*theta[1]

#una variable
def costFunction(x,y,theta):
    m=np.size(x) #asumiendo que size(x)=size(y)
    suma=sum(map(lambda x,y: (h(x,theta)-y)**2,x,y))
    return suma/(2*m)

def gradientDescent(x,y,theta,alfa):
    
    m=np.size(x)
    temp=theta[0]-(1/m)*alfa*sum(map(lambda x,y: h(x,theta)-y,x,y))
    theta[1]=theta[1]-(1/m)*alfa*sum(map(lambda x,y: (h(x,theta)-y)*x,x,y))
    theta[0]=temp
    #print(theta)
    return theta

def regresionLineal(x,y,alfa,maxit,umbralError):
    """jajajaja"""
    theta=[0,1]
    costArray=np.zeros(maxit)

    for k in range(maxit):
        theta=gradientDescent(x,y,theta,alfa)
        costArray[k]=costFunction(x,y,theta)
        #print(costArray[k])
        if(np.abs(costArray[k])<umbralError):
            break
        elif any(np.isnan(theta)):
            raise Exception("Favor validar sus parámetros de entrada.")

    return theta,costArray


if __name__ == "__main__":
    #variables de entrada
    x=np.array([0,1,2,3,4,5,6,7])
    y=np.array([0,3,6,9,12,15,18,22])
    linePlot(x,y)

    #variables del sistema
    alfa=0.1
    maxit=10000
    umbralError=0.01

    theta,costArray=regresionLineal(x,y,alfa,maxit,umbralError)

    #ver resultados
    linePlot(x,h(x,theta))
    plt.figure()
    linePlot(range(maxit),costArray)

    plt.show()


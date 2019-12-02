import numpy as np
from matplotlib import pyplot as plt
from functools import reduce


def linePlot(x,y,xlabel="",ylabel="",title="",axis=None):
    
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if axis:
        plt.axis(axis)
    

#recibe el arreglo de entrada
def hipotesis(theta,*x):
    m=len(x[0])
    x=reorganizarParametros(x,m)
    res=np.zeros(m)
    
    return list(map(lambda ex:h(theta,ex),x))
        

#RECIBE UNA FILA
def h(theta,x):
    #x=np.append(1,x)
    #print((x*theta))
    return sum(x*theta)
    #return np.transpose(theta)@x

def reorganizarParametros(x,m):
    #reorganizando x para más fácil manejo
    
    x2=[]
    for i in range(m):   
        x2.append([x_j[i] for x_j in x ])        

    x=np.insert(x2,0,1,axis=1) #agregando la columna x_0
    return x

#una variable
def costFunction(x,y,theta):
    
    m=np.size(y) #asumiendo que size(x)=size(y)
    x=reorganizarParametros(x,m)
    suma=sum(map(lambda x,y: (h(theta,x)-y)**2,x,y))
    return suma/(2*m)

def gradientDescent(x,y,theta,alfa):
    
    m=len(y) #número de filas
    n=len(x) #número de parámetros
    temp=theta.copy()

    x=reorganizarParametros(x,m)
    
    for j in range(n+1):
        #print(n)
        theta[j]=temp[j]-(1/m)*alfa*sum(map(lambda x_i,y: (h(temp,x_i)-y)*x_i[j],x,y))

    return theta

def regresionLineal(*x,y,alfa=0.01,maxit=1000,umbralError=0.001):
    """jajajaja"""
    theta=np.ones(len(x)+1).astype(float)
    
    costArray=np.zeros(maxit)
    last_it=0

    for k in range(maxit):
        theta=gradientDescent(x,y,theta,alfa)
        
        costArray[k]=costFunction(x,y,theta)
        #print(theta)
        if(np.abs(costArray[k])<umbralError):
            last_it=k
            break
        elif any(np.isnan(theta)):
            raise Exception("Favor validar sus parámetros de entrada.")

    return theta,costArray


if __name__ == "__main__":
    #variables de entrada
    x=np.array([0,1,2,3,4,5,6,7,8])
    y=np.array([0,2,4,6,8,10,12,14,20])
    linePlot(x,y)

    #variables del sistema
    alfa=0.05
    maxit=1000
    umbralError=0.00000001

    theta,costArray=regresionLineal(x,y=y,alfa=alfa,maxit=maxit,umbralError=umbralError)

    #ver resultados
    #plt.figure()
    linePlot(x,hipotesis(theta,x))
    plt.figure()
    linePlot(costArray,range(len(costArray)))
    #print(theta)
    plt.show()


import numpy as np
from matplotlib import pyplot as plt

#TODO
#ECUACIÓN NORMAL
def linePlot(x,y,xlabel="",ylabel="",title="",axis=None):
    '''Grafica en el plano x-y'''
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if axis:
        plt.axis(axis)

def countourPlot(*x,y,xlabel="",ylabel="",title="",axis=None):

    assert len(x)==2,"Solo es posible graficar para tres dimensiones."
    if not len(x[0])==len(x[1])==len(y):
        raise Exception("Los tamaños de las entradas deben ser los mismos. "+str(len(x[0]))+" "+str(len(x[1]))+" "+str(len(y)))


    plt.contour(x[0], x[1], y, cmap='viridis', levels=np.logspace(-2, 3, 20))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x[0], x[1], 'ro', ms=10, lw=2)
    plt.title(title)
    

#recibe el arreglo de entrada
def hipotesis(theta,*x):
    '''Recibe: 
        valores de los parámetros.
        n listas de m elementos con los valores de cada feature
       Retorna predicciones para cada fila'''
    m=len(x[0])
    x=reorganizarParametros(x,m)
    res=np.zeros(m)
    
    return list(map(lambda ex:h(theta,ex),x))
        


def h(theta,x):
    #x=np.append(1,x)
    #print((x*theta))
    return sum(x*theta)
    #return np.transpose(theta)@x

def reorganizarParametros(x,m):
    '''Recibe un tuple de n elementos donde cada elemento es un array de m  filas'''    
    #print(x)
    x2=[]
    for i in range(m):   
        x2.append([x_j[i] for x_j in x ])        

    x=np.insert(x2,0,1,axis=1) #agregando la columna x_0
    #print(x)
    return x


def costFunction(x,y,theta):
    """error cuadrático medio de h(x) comparado con y"""
    m=np.size(y) #asumiendo que size(x)=size(y)
    x=reorganizarParametros(x,len(x))
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

def regresionLineal(*x,y,alfa,maxit=1000,umbralError=0.001):
    """"""
    theta=[np.ones(len(x)+1).astype(float)]
    
    costArray=np.nan*np.zeros(maxit+1)
    costArray[0]=costFunction(x,y,theta[-1])

    for k in range(maxit):

        theta.append(gradientDescent(x,y,theta[-1].copy(),alfa))    
        costArray[k+1]=costFunction(x,y,theta[-1])
        if np.abs(costArray[k+1]-costArray[k]) < umbralError:
            break
        elif any(np.isnan(theta[-1])):
            raise Exception("Favor validar sus parámetros de entrada.")
        elif costArray[k+1]>costArray[k]:
            raise Exception("Algoritmo divergente. Disminuya la tasa de aprendizaje.") 

    print("Convergencia alcanzada con "+str(k+1)+" iteraciones." if k+1<maxit  else "Número máximo de iteraciones alcanzado")
    print("ECM Final: "+str(costArray[k+1]))
    return theta,costArray[:k+1]

def featureScaling(x,tipo="std",normalizar=True):
    """x es la característica. 
    si tipo=std, normaliza con respecto a la dev. estándar (por defecto)
    si tipo=rango, normaliza con respecto a max(x)-min(x)
    si normalizar=False, no normaliza solo centra con respecto a la media

    retorna x_normalizado,media[,std/rango]
    """
    media=np.mean(x)
    std=np.std(x)
    rango=(np.max(x)-np.min(x))
    x_centrado=x-media

    if normalizar:
        assert tipo in ["std","rango"]
        
        if tipo=="std":
            return x_centrado/std,media,std
        else: 
            return x_centrado/rango,media,rango
        
    else:
        return x_centrado,media

def featureDescaling(x_n,media,s=""):
    if s:
        return s*x_n+media
    else:
        return x_n+media

if __name__ == "__main__":
    #variables de entrada
    #countourPlot("jajaja","jiji","jjojoo",y="jijiji")
    x=np.linspace(0,2*np.pi,num=100)
    y=np.linspace(0,2*np.pi,num=100)
    linePlot(x,y)
    
    #variables del sistema
    alfa=0.33
    maxit=10000
    umbralError=1e-6

    x_n,media,rango=featureScaling(x,tipo="std")
    #linePlot(x,y)
    thetaArray,costArray=regresionLineal(x_n,y=y,alfa=alfa,maxit=maxit,umbralError=umbralError)

    #ver resultados
    #plt.figure()
    linePlot(featureDescaling(x_n,media,rango),hipotesis(thetaArray.pop(),x_n))
    plt.legend(["original","prediccion"])
    plt.figure()
    linePlot(range(len(costArray)),costArray)

    #print(theta)
    plt.show()


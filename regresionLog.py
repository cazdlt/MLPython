from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

f,ax=plt.subplots(1,2)
centers=[[-2,-2],[0,-2]]
X, y = make_blobs(n_samples=8000, centers=centers, cluster_std=1,random_state=42)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
ax[0].scatter(X_test[:,0],X_test[:,1],c=y_test)

reg=LogisticRegression(random_state=42,max_iter=10000).fit(X_train,y_train)

y_pred=reg.predict(X_test)
ax[1].scatter(X_test[:,0],X_test[:,1],c=y_pred)

x0_min,x0_max=ax[1].get_xlim()
y_min,y_max=ax[1].get_ylim()
coef=reg.coef_[0]
intercept=reg.intercept_

x1=lambda x0,coef,intercept: -(x0*coef[0]+intercept)/coef[1]

ax[1].plot([x0_min,x0_max],[x1(x0_min,coef,intercept),x1(x0_max,coef,intercept)],color="red")
ax[1].set_ylim(bottom=y_min,top=y_max)
print("Puntaje: "+str(reg.score(X_test,y_test)))



plt.show()




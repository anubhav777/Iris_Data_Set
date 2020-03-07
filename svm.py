import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV

iris = sns.load_dataset('iris')
setosa = sns.load_dataset('setosa')

print(iris.head())

sns.pairplot(iris,hue='species')

sns.pairplot(setosa)

sns.kdeplot(setosa['sepal_length'],setosa['sepal_width'],cmap="plasma", shade=True, shade_lowest=False)

X = iris.drop('species',axis=1)
y = iris['species']
print(X,y)
X_train,X_Test,y_train,y_test=train_test_split(X,y,test_size=0.3)
sv=SVC()

sv.fit(X_train,y_train)

y_pred=sv.predict(X_Test)

X_train,X_Test,y_train,y_test=train_test_split(X,y,test_size=0.3)

print(confusion_matrix(y_test,y_pred))

param_grid={'C':[0.1,1,10,100,100],'gamma':[1,0.1,0.01,0.001,0.0001]}

gs=GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
gs.fit(X_train,y_train)

y_pred=gs.predict(X_Test)

print(confusion_matrix(y_test,y_pred))
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris=load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
correct_predictions=0
wrong_predictions=0
for i in range(len(y_test)):
    if y_pred[i]==y_test[i]:
        print(f'correct_prediction:predicted_class{y_pred[i]}|true_class{y_test[i]}')
        correct_predictions+=1
    else:
        print(f'wronng_prediction:predicted_class{y_pred[i]}|true_class{y_test[i]}')
        wrong_predictions+=1
accuracy=accuracy_score(y_pred,y_test)       
print("accuracy:",accuracy)
print("correct_predictions:",correct_predictions)
print("wrong_predictions:",wrong_predictions)
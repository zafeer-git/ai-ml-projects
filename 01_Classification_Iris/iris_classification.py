#import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#load the iris dataset
iris=load_iris()
iris_df= pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species']=iris.target

#split the data into features and target variable
X=iris_df.drop('species', axis=1)
y=iris_df['species']

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#initialize the random forest training
model=RandomForestClassifier(n_estimators=100, random_state=42)

#to train the model
model.fit(X_train, y_train)

#creating array with my variables to get the prediciton
new_data=[[5.1,3.5,1.4,0.2]]

#make predictions on the testing set
y_pred_test=model.predict(X_test)

#Evaluate the model
accuracy=accuracy_score(y_test, y_pred_test)
print(f'Accuracy: {accuracy:2f}')

import matplotlib as mp 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics


#Importing and preprocessing data from the dataset
dataSet=pd.read_csv("data.csv")
X = dataSet.iloc[:, :-1].values
Y = dataSet.iloc[:, 1].values

#Plotting raw data to visualize and deduce a relationship
plt.scatter(X,Y,color="red")
plt.title('Hours Studied vs Scores Achieved')
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.show()

#spliting the data into training and testing sets
trainingX,testingX,trainingY,testingY=train_test_split(X,Y,test_size=1/3,random_state=0)

#Applying the linear regression model to our processed training dataset
lireg=LinearRegression() 
lireg.fit(trainingX,trainingY) 
y_pred=lireg.predict(testingX) 

#plotting the regression line and visualing the corelation
plt.scatter(trainingX,trainingY,color="red")
plt.plot(trainingX,lireg.predict(trainingX),color="blue")
plt.title('Hours Studied vs Scores Achieved(Training Set)')
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.show()

#Plotting the Regression line about the testing set values 
plt.scatter(testingX,testingY,color="red")
plt.plot(trainingX,lireg.predict(trainingX),color="blue")
plt.title('Hours Studied vs Scores Achieved(Testing Set)')
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.show()

#Comparing the predicted values with the testing values in a tabular visualization
df=pd.DataFrame({'Testing ':testingY,'Predictions ':y_pred })
print(df)
 
#predicting the test scores after 9.25hrs/day
sample=np.array([9.25])
sample=sample.reshape(-1,1)
pred=lireg.predict(sample)
print("After 9.25hrs/day of studying the predict test scores will be = " , pred[0])

#Calculating the mean square error to check the efficacy of the used algorithm(model in this case)
Error=metrics.mean_absolute_error(testingY,y_pred)
print("Absolute mean error = ",Error)




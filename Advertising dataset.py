#We start by importing the libraries. Here we have uploaded the basic libraries to begin the project
#I will upload the machine learning libraries later as I need them
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Importing the dataset.Here I have left the document source empty
#The dataset source will be different for each person
ad_data = pd.read_csv('DATASET SOURCE')

#Most of the time its hard to understand the data simply by looking at it in spreadsheet viewer.
#First, we would like to know what the data is. How many columns, rows and data entries are there ?
ad_data.info()

#Next, its a good idea to have a look at the data to get a feel of it
#We are going to analyze the data first. This is done to get a better idea of what relationships we can explore
#To understand how the rows and columns look like, whether they have categorical data among the other things
ad_data.head()

#Now, we know that there is indeed numerical data (discrete form). Let's have a look at the basic statistical makeup
ad_data.describe()

#I want to conduct some exploartory data analysis
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')


#I want to see if there is a relationship between the Age of the Customer and the area income
sns.jointplot(x='Age',y='Area Income',data=ad_data)

#To see if there is a relationship between Daily Time Spent on Site and the Age of the customers
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');

#To see if there is a relationship betwee the Daily Time Spent on Site vs Daily Internet Usage
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')

#Since there are a lot of factors involed in finding the relationship,
#I am going to see all of them, to observe if I can specifically target any relationship
sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')


#We will now import the machine learning library and split the data into train and test
from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['CLicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X,y,test size = 0.3, random_state = 101)

#We will now train the data. First, we will upload the library to train and test the data
from sklearn.linear_model import LogisticRegression
#we will create an instance of LinearRegression, so we can pass X_train and y_train as arguments in it
#Next, we train and fit the training data.After that lets have a look at the coefficients of the model
logmodel = LogisticRegression
logmodel.fit(X_train,y_train)

#After fitting the data, I would like to evaluate the performance of the model
#For this I am going to create a classification report for the model.
#I want to have a look at the precision and the f-score
from sklearn.metrics import classification_report
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))

#!/usr/bin/env python
# coding: utf-8

# TASK 1: TITANIC SURVIVAL PREDICTION
# 
#     Founder: Swati Srivastava
#     
#     Domain: Data Science
#     
#     Name: Bobade Adedamola Timilehin
#     
#     Student ID: CA/JN1/13239
#     
#     Aim: To build a model that predicts whether a passanger on the Titanic survived or not  
#     
# Task:TITAN CLASSIFICATION
# 
#         1. Use the Titanic dataset to build a model that predicts whether a passenger on the Titanic survived or not. This is a classic beginner project with readily available data
#         2. The dataset typically used for this project contains information about individual passengers, such as their age, ticket class, fare, cabin, and whether or not they survived.
#         
# Data Description
# 
#         Pclass: Aproxy for socio-economic status
#         1st=Upper
#         2nd=Middle
#         3rd=Lower
#         Sibsp: The dataset defines family relations in this way...
#         siblings=brother, sister, stepbrother,stepsister
#         spouse=Husband, wife(mistress and fiancies were ignored)
#         Parch:  The dataset defines family relation in this way...
#         parent=mother, father
#         child=daughter, son, stepdaughter, stepson. Some children travelled only with a nanny therefore Parch=0 for them.

# In[6]:


#IMPORTING IMPORTANT LIBRARIES


# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#IMPORTING THE DATASET


# In[25]:


df = pd.read_csv('Titanic-Dataset.csv')


# In[26]:


df.head()


# In[27]:


#view 10 rows


# In[28]:


df.head(10)


# In[29]:


#number of rows and columns
df.shape


# In[30]:


df.describe()


# In[31]:


#information about the data
df.info()


# In[32]:


#checking the number of missing values
df.isnull().sum()


# In[33]:


#Age,Cabin and Embarked has Null values 


# In[34]:


#filling the blank with median value
df['Age'].fillna(df['Age'].median(), inplace=True)


# In[35]:


#Counting the Embarked
df['Embarked'].value_counts()


# In[36]:


#replace blanks with mode value
df['Embarked'].fillna('S', inplace=True)


# In[37]:


#instead of counting the Embarked cloumn we can find the mode value in the column
print(df['Embarked'].mode())


# In[38]:


print(df['Embarked'].mode()[0])


# In[39]:


#we can also fill the blanks space like this
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


# In[40]:


#null value in dataset
df.isnull().sum()


# In[41]:


#removing cabin column because it contains a significant number of missing values.or df.drop(columns="cabin", axis=1)
df.drop(columns="Cabin", inplace=True)


# In[42]:


#Checking if there is Null value 
print(df.isnull().sum())


# In[ ]:





# In[18]:


print(df.head())


# In[ ]:


Dataset is Cleaned

Exploratory Data Analysis 
i described survival variable as
0=NO, 1=YES


# In[19]:


df['Survived'].value_counts()


# In[ ]:


#Data visualization


# In[20]:


sns.countplot(data=df, x='Survived')


# In[ ]:


Pclass column is described as:
    1st= Upper
    2nd= Middle
    3rd= Lower


# In[21]:


sns.countplot(data=df, x='Pclass')


# In[22]:


df['Pclass'].value_counts()


# In[23]:


sns.countplot(data=df, x='Sex')


# In[24]:


df['Sex'].value_counts()


# In[25]:


import seaborn as sns


# In[26]:


import seaborn as sns
print(sns.__version__)


# In[27]:


import matplotlib.pyplot as plt


# In[ ]:


#Showing count of survival wrt Pclass


# In[46]:


#Survival wrt to Ticket class
sns.countplot(df['Survived'], hue=df['Pclass'])


# In[47]:


#Ticket class wrt to Survived
sns.countplot(df['Pclass'], hue=df['Survived'])


# In[48]:


df['Sex'].head()


# In[49]:


sns.countplot(x=df['Survived'], hue=df['Sex'])


# In[ ]:


# coverting categorical columns
df.replace(('Sex':('male':1, 'female':0), 'Embarked':('S':0,'C':1,'Q':2)), inplace=True)


# In[ ]:


modeling the data
transforming gender(Sex) into numeric
male=1
female=0
using encoder from sklearn library


# In[52]:


from sklearn.preprocessing import LabelEncoder
Labelencoder =LabelEncoder()
df['Sex']= Labelencoder.fit_transform(df['Sex'])
df.head()


# In[53]:


sns.countplot(x=df['Sex'], hue=df['Survived'])
plt.show()


# In[54]:


df.drop(columns=["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Fare", "Age", "Embarked"], inplace=True)


# In[55]:


df.head()


# In[ ]:


#Modelling


# In[56]:


x=df[['Sex', 'Pclass']]
y=df['Survived']


# In[57]:


#spliting data into test and train by using sklearn library


# In[58]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)


# In[59]:


print("x_train shape:", x_train.shape)


# In[64]:


x_train.head()


# In[60]:


print("x_test shape:", x_test.shape)


# In[61]:


print("y_train shape:", y_train.shape)


# In[62]:


print("y_test shape:", y_test.shape)


# In[63]:


print(x.shape, x_train.shape, x_test.shape)


# In[ ]:


#creating training model


# In[67]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix


# In[68]:


#Instantiate the  Prediction(LogisticRegression) model

model = LogisticRegression()
# In[69]:


#fitting the model on the training data


# In[ ]:





# In[70]:


print(type(x_train))


# In[71]:


print(type(y_train))


# In[72]:


print(x_train[:5])


# In[93]:


print(x_train)


# In[73]:


print(y_train[:5])


# In[74]:


#Convert y_train to a NumPy array since is a 'pandas.Series'


# In[75]:


y_train = y_train.values


# In[76]:


print(y_train)


# In[77]:


#Instantiate the  Prediction(LogisticRegression) model


# In[78]:


model = LogisticRegression()


# In[79]:


#fitting the model on the training data


# In[80]:


model.fit(x_train, y_train)


# In[81]:


#Model Evaluation


# In[82]:


y_pred=model.predict(x_test)


# In[83]:


print(y_pred)


# In[84]:


print("Accuracy_score:", accuracy_score(y_test, y_pred))


# In[85]:


print("Matrix:", confusion_matrix(y_test, y_pred))


# In[86]:


y_test


# In[87]:


submission=x.iloc[:,:].values
y_final=model.predict(submission)


# In[88]:


y_final.shape


# In[89]:


final = pd.DataFrame()
final["Sex"]=x['Sex']
final["Survived"] =y_final


# In[90]:


final.to_csv("submission.csv",index=False)


# In[91]:


#PREDICT([[PCLASS, SEX]]) IF SURVIVED OR NOT


# In[92]:


import warnings
warnings.filterwarnings('ignore')
result=model.predict([[3, 1]])
if result == 0:
    print("so sorry, No Survival")
else:
    print("Survived")


# Interpretation and Summary Report for Titanic Survival Prediction
# 
# Summary of the Analysis
# 
# Data Preprocessing:
# 
# Data is cleaned and preprocessed, including encoding categorical variables (e.g., converting 'Sex' to numerical values).
# The dataset is split into training and test sets.
# 
# Model Training:
# 
# A logistic regression model is trained on the training data.
# The model is used to predict whether a passenger survived based on the features.
# 
# Model Evaluation:
# 
# The model's performance is evaluated using accuracy and confusion matrix metrics.
# 
# Prediction:
# 
# The model is applied to test data to predict survival outcomes.
# A submission file is created with predicted survival outcomes for all passengers.
# 
# Key Metrics and Results
# 
# Accuracy:
# 
# The model achieves an accuracy score of 0.7877, meaning it correctly predicts survival about 78.77% of the time on the test data.
# 
# Confusion Matrix:
# 
# The confusion matrix is as follows:
# [[92 18]
#  [20 49]]
# True Negatives (92): Passengers correctly predicted as not survived.
# False Positives (18): Passengers incorrectly predicted as survived.
# False Negatives (20): Passengers incorrectly predicted as not survived.
# True Positives (49): Passengers correctly predicted as survived.
# 
# Visualization
# 
# Scatter Plot of Predicted vs. Actual Survival:
# A scatter plot is used to visualize the alignment of predicted survival outcomes with actual outcomes.
# 
# Conclusion
# 
# The logistic regression model shows reasonable performance in predicting the survival of Titanic passengers with an accuracy of approximately 78.77%. The model's predictions are well-aligned with actual outcomes, as indicated by the confusion matrix and accuracy score. This analysis demonstrates the effectiveness of logistic regression for binary classification tasks such as survival prediction based on passenger data.

# In[ ]:





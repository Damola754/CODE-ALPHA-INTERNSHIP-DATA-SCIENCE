#!/usr/bin/env python
# coding: utf-8

# TASK 4: SALES PREDICTION 
# 
# Founder: Swati Srivastava
# 
# Domain: Data Science
# 
# Name: Bobade Adedamola Timilehin
# 
# Student ID: CA/JN1/13239
# 
# Aim: To build a model which predicts sales based on the money spent on different platforms for marketing
# 
# Task: Predictive Model With Linear Regression
# 
#     Use the advertising dataset given and analyse the relationship between 'TV advertising' and 'Sales' using a simple linear regression model
#      

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#geting the dataset


# In[3]:


df =pd.read_csv('advertising.csv')


# In[4]:


df.head()


# In[ ]:





# In[5]:


df.shape


# In[6]:


df.describe()


# Data Cleaning
# 

# In[7]:


#Checking for null values in the dataset
df.isnull().sum()


# In[8]:


#Outlier 
fig, axs = plt.subplots(4, figsize =(6,6))
firstplt=sns.boxplot(df['TV'], ax = axs[0])
secondplt=sns.boxplot(df['Newspaper'], ax = axs[1])
thirdplt=sns.boxplot(df['Radio'], ax = axs[2])
plt.tight_layout()


# In[9]:


#Using scatterplot to visualize sales wrt TV, Radio and Newspaper
sns.pairplot(df, x_vars=['TV', 'Radio','Newspaper'], y_vars='Sales', kind='scatter')
plt.show()


# Another way to visualize sales wrt other variables 

# In[10]:


# Creating a figure and a set of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot TV vs Sales
sns.scatterplot(ax=axes[0], data=df, x='TV', y='Sales')
axes[0].set_title('TV vs Sales')

# Plot Radio vs Sales
sns.scatterplot(ax=axes[1], data=df, x='Radio', y='Sales')
axes[1].set_title('Radio vs Sales')

# Plot Newspaper vs Sales
sns.scatterplot(ax=axes[2], data=df, x='Newspaper', y='Sales')
axes[2].set_title('Newspaper vs Sales')

# Display the plots
plt.tight_layout()
plt.show()


# In[11]:


# Creating a histogram using matplotlib
plt.hist(df['Newspaper'], bins=10, color='green')

# Setting the x-axis label
plt.xlabel('Newspaper')

# Setting the title
plt.title('Histogram of Newspaper')

# Show the plot
plt.show()


# In[12]:


# Create a histogram using matplotlib
plt.hist(df['TV'], bins=10, color='blue')

# Set the x-axis label
plt.xlabel('TV')

# Set the title
plt.title('Histogram of TV')

# Show the plot
plt.show()


# In[ ]:





# In[13]:


# Create a histogram using matplotlib
plt.hist(df['Radio'], bins=10, color='brown')

# Set the x-axis label
plt.xlabel('Radio')

# Set the title
plt.title('Histogram of Radio')

# Show the plot
plt.show()


# In[14]:


#Correlation between different variables
sns.heatmap(df.corr(),annot = True)
plt.show()


# From the heatmap, the variable TV appears to be the most correlated with Sales.

# Building the model
# 

# In[15]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(df[['TV']], df[['Sales']], test_size=0.2, random_state=42)


# In[16]:


print(X_train)


# In[17]:


print(Y_train)


# In[18]:


X_test.head()


# In[19]:


Y_test.head()


# In[ ]:





# In[20]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)


# In[21]:


Y_test_prediction = model. predict(X_test)


# In[22]:


print(Y_test_prediction)


# In[23]:


Y_train_prediction = model.predict(X_train)


# In[24]:


print(Y_train_prediction)


# In[25]:


model.coef_


# In[26]:


model.intercept_


# In[27]:




from sklearn.metrics import mean_squared_error, r2_score
#Evaluate the Model for test data
mse = mean_squared_error(Y_test, Y_test_prediction)
r2 = r2_score(Y_test, Y_test_prediction)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

#Visualize the Results  for the test data
plt.scatter(X_test, Y_test, color='blue', label='Actual Sales')
plt.scatter(X_test, Y_test_prediction, color='red', label='Predicted Sales')
plt.plot(X_test, Y_test_prediction, color='red', linewidth=2)
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('TV vs Sales')
plt.legend()
plt.show()


# In[28]:


plt.plot(Y_test_prediction)


# In[29]:


plt.plot(Y_train_prediction)


# In[30]:



from sklearn.metrics import mean_squared_error, r2_score
#Evaluate the Model for train data
mse = mean_squared_error(Y_train, Y_train_prediction)
r2 = r2_score(Y_train, Y_train_prediction)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

#Visualize the Results  for the train data
plt.scatter(X_train, Y_train, color='blue', label='Actual Sales')
plt.scatter(X_train, Y_train_prediction, color='red', label='Predicted Sales')
plt.plot(X_train, Y_train_prediction, color='red', linewidth=2)
plt.xlabel('TV')
plt.ylabel('Sales')
plt.title('TV vs Sales')
plt.legend()
plt.show()


# 
# Interpretation and Summary Report for Sales Prediction
# 
# 
# 
# 
# Summary of the Analysis
# 
# Data Preprocessing and Splitting:
# The dataset is split into training and test sets.
# Features (TV advertising spending) and target (Sales) are defined.
# 
# Model Training:
# A linear regression model is trained using the training data.
# The model aims to predict sales based on TV advertising spending.
# 
# Model Evaluation:
# The performance of the model is evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics.
# The model's predictions are compared with actual sales data for both the training and test sets.
# Interpretation of Results
# 
# Training Data Evaluation:
# 
# Mean Squared Error (MSE): 4.998
# This indicates the average squared difference between the actual sales and the predicted sales. A lower value indicates better fit.
# R-squared (R²): 0.813
# This indicates that approximately 81.3% of the variance in sales can be explained by the TV advertising spending in the training data.
# 
# Test Data Evaluation:
# 
# Mean Squared Error (MSE): 6.10
# This is slightly higher than the MSE for the training data, indicating a slightly worse fit on the test data.
# R-squared (R²): 0.803
# This indicates that approximately 80.3% of the variance in sales can be explained by the TV advertising spending in the test data, which is very close to the training data performance.
# 
# Visualizations
# 
# Training Data Visualization:
# A scatter plot of actual vs. predicted sales for the training data is shown.
# The red line represents the linear regression fit.
# The close clustering of points around the red line indicates a good fit
# 
# Visualizations
# 
# Test Data Visualization:
# A scatter plot of actual vs. predicted sales for the training data is shown.
# The red line represents the linear regression fit.
# The close clustering of points around the red line indicates a good fit. 
# 
# 
# Interpretation
# 
# The analysis involves a linear regression model to predict sales based on TV advertising spending. The model shows good performance with an R-squared (R²) value of 0.813 for training data and 0.803 for test data, indicating that around 80% of the variance in sales is explained by TV spending. The Mean Squared Error (MSE) is 4.998 for training and 6.10 for test data, showing slightly worse performance on test data but still a reasonable fit. Visualizations confirm a good alignment between actual and predicted sales. Overall, TV advertising is a strong predictor of sales, and the model generalizes well to new data.

# In[ ]:





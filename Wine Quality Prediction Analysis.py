#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('winequalityN.csv')


# In[3]:


df.head()


# In[4]:


# statistical Data 
   
df.describe()    


# In[5]:


# Data types info

df.info()


# In[24]:


# check for the Null values

df.isnull().sum()


# In[22]:


# fill the missing values

for col, vaule in df.items():
    if col != 'type': # this is for if the values have strings in it
        df[col] = df[col].fillna(df[col].mean())
        


# In[25]:


# Run the code above to check the null values again so that we can figure out that Null values have been filled

df.isnull().sum()


# As in the output you can see below there are no missing values now


# # Exploratory Data Analysis

# In[26]:


fig, ax = plt.subplots(ncols=6,nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    if col != 'type':
        sns.boxplot(y=col,data=df, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5,w_pad=0.7, h_pad=5.0)      


# In[27]:


fig, ax = plt.subplots(ncols=6,nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    if col != 'type':
        sns.distplot(value, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5,w_pad=0.7, h_pad=5.0)   


# In[29]:


# Log Transformation

df['free sulfur dioxide'] = np.log(1 + df['free sulfur dioxide'])


# In[30]:


# check back the Free sulfur Dioxide plot

sns.distplot(df['free sulfur dioxide'])


# In[31]:


sns.countplot(df["type"])


# In[32]:


sns.countplot(df["quality"])


# # Correlation Matrix

# In[33]:


corr = df.corr()
plt.figure(figsize=(20,10)) # Please increase the font if the figures looks compressed
sns.heatmap(corr, annot=True,cmap='coolwarm')


# # Input Split

# In[34]:


x = df.drop(columns=['type', 'quality'])
y= df['quality']


# # Class Imbalancement

# In[35]:


y.value_counts()


# In[36]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE(k_neighbors=4)
#transfor the data set 

x,y = oversample.fit_resample(x,y)


# In[37]:


y.value_counts()


# # MODEL TRAINING

# In[39]:


# Classify Function 

from sklearn.model_selection import cross_val_score, train_test_split
def classify (model, x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state=42)
    # Train Model
    model.fit(x_train,y_train)
    print ("Accuracy:model:",model.score(x_test,y_test)*100)
    
    # Cross validation
    
    score= cross_val_score(model,x,y,cv=5)
    print("CV Score:", np.mean(score)*100)


# In[40]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model,x,y)


# In[41]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify (model,x,y)


# In[43]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify (model,x,y)


# In[45]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
classify (model,x,y)


# In[ ]:





# In[ ]:





# In[ ]:





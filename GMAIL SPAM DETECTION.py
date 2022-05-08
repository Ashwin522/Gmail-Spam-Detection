#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import os


# In[2]:


dt = pd.read_csv("spam.csv",encoding='latin1')
dt.head(10)


# In[3]:


dt['spam'] = dt['type'].map({'spam':1 , 'ham' :0}).astype(int)
dt.head(5)


# In[4]:


print("COLUMNS IN THE GIVEN DATA: ")
for col in dt.columns:
    print(col)


# In[5]:


t=len(dt['type'])
print("NO OF ROWS IN REVIEW COLUMN: ",t)
t=len(dt['text'])
print("NO OF COLUMNS IN LIKED COLUMN: ",t)


# In[6]:


dt['text'][1]#before


# In[7]:


def tokenizer(text):
    return text.split()


# In[8]:


dt['text']=dt['text'].apply(tokenizer)


# In[9]:


dt['text'][1]#after


# In[ ]:





# In[10]:


dt['text'][1]
from nltk.stem.snowball import SnowballStemmer
porter = SnowballStemmer("english",ignore_stopwords = False)


# In[11]:


def stem_it(text):
    return [porter.stem(word) for word in text]


# In[12]:


dt['text'] = dt['text'].apply(stem_it)


# In[13]:


dt['text'][1]


# In[14]:


dt['text'][109]


# In[15]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


# In[16]:


def lemmit_it(text):
    return [lemmatizer.lemmatize(word, pos ="a") for word in text]


# In[17]:


dt['text'] = dt['text'].apply(lemmit_it)


# In[18]:


dt['text'][109]


# In[19]:


dt['text'][50]


# In[20]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[21]:


def stop_it(text):
    review = [word for word in text if not word in stop_words]
    return review


# In[22]:


dt['text'] = dt['text'].apply(stop_it)


# In[23]:


dt['text'][50]


# In[29]:


dt.head()


# In[30]:


dt['text'] = dt['text'].apply(' '.join)
dt.head()


# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
y = dt.spam.values
x = tfidf.fit_transform(dt['text'])


# In[32]:


from sklearn.model_selection import train_test_split
x_train,x_text,y_train,y_text=train_test_split(x,y,random_state=1,test_size=0.2,shuffle=False)


# In[34]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_text)
from sklearn.metrics import accuracy_score
acc_log = accuracy_score(y_pred , y_text)*100
print("accuracy: ",acc_log)


# In[39]:


file spam.csv


# In[38]:


os.stat("spam.csv")


# In[40]:


pip3 install chardet


# In[ ]:





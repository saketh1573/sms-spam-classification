#import all basic libraries
import numpy as np
import pandas as pd
#import dataset
dataset=pd.read_csv("SMSSpamCollection",sep="\t",names=['label','message'])
print(dataset.describe())
#giving numerical values
dataset['label']=dataset['label'].map({'ham':0,'spam':1})
print(dataset)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#countplot for spam v/s ham dataset(because the dataset is imbalenced)
plt.figure(figsize=(8,8))
g=sns.countplot(x="label",data=dataset)
p=plt.title("countplot for spam v/s ham dataset")
p=plt.xlabel('spam')
p=plt.ylabel('count')
#balencing the dataset unsing oversampling
spamdata=dataset[dataset['label']==1]
print(spamdata)
#ham=~6*spam
for i in range(0,5):
    dataset=pd.concat([dataset,spamdata])
dataset.shape
#count plot for balenced dataset
plt.figure(figsize=(8,8))
g=sns.countplot(x="label",data=dataset)
p=plt.title("countplot for spam v/s ham dataset")
p=plt.xlabel('spam')
p=plt.ylabel('count')
#histogram to check if there is a relativity between word count and message type
plt.figure(figsize=(12,6))
#(1,1)
plt.subplot(1,2,1)
g=sns.histplot(dataset[dataset['label']==0].word_count,kde=True)
p=plt.title('distribution of wordcount for Ham messages')
#(1,1)
plt.subplot(1,2,2)
g=sns.histplot(dataset[dataset['label']==1].word_count,color="red",kde=True)
p=plt.title('distribution of wordcount for Spam messages')
plt.show()
#Creating new features wrt symbols
def currency_present(data):
    symbols=['$','€',"¥","£","₹","฿"]
    for i in symbols:
        if i in data:
            return 1
    return 0
dataset["contains_currency_symbol"]=dataset['message'].apply(currency_present)
print(dataset)
#countplot for contains_currency_symbol
plt.figure(figsize=(8,8))
g=sns.countplot(x='contains_currency_symbol',data=dataset,hue='label')
p=plt.title('containing currency symbol')
p=plt.xlabel('currency symbol')
p=plt.ylabel('count')
p=plt.legend(labels=['ham','spam'],loc=9)
#creating new feature of containing numbers
def number(data):
    for i in data:
        if ord(i)>=48 and ord(i)<=57:
            return 1
    return 0
dataset['contains number']=dataset['message'].apply(number)
dataset
#countplot for containing numbers
plt.figure(figsize=(8,8))
g=sns.countplot(x='contains number',data=dataset,hue='label')
p=plt.title('containing number')
p=plt.xlabel('contains numbers')
p=plt.ylabel('count')
p=plt.legend(labels=['ham','spam'],loc=9)
#data cleaning
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
corpus = []
wnl=WordNetLemmatizer()

for sms in list(dataset.message):
    message=re.sub(pattern='[^a-zA-Z]',repl=' ',string=sms)#filtering out special charecters and numbers
    message=message.lower()
    words =message.split()
    filtered_words=[word for word in words if word not in set(stopwords.words('english'))] #removing stopwords
    lemm_words=[wnl.lemmatize(word) for word in filtered_words] #bringing words to its simplets forms
    message=' '.join(lemm_words)
    corpus.append(message)
    #creating the bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(max_features=500)
vectors=tf.fit_transform(corpus).toarray()
feature_names = tf.get_feature_names_out()
x = pd.DataFrame(vectors, columns=feature_names)
y=dataset['label']
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import classification_report,confusion_matrix
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#naive bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
cv=cross_val_score(nb,x,y,scoring='f1',cv=10)
print(round(cv.mean(),3))
print(round(cv.std(),3))
nb.fit(x_train,y_train)
y_pred=nb.predict(x_test)

print(classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(8, 8))
axis_labels = ['ham', 'spam']
g = sns.heatmap(data=cm, xticklabels=axis_labels, yticklabels=axis_labels,annot=True,fmt='g', cmap='Blues',cbar_kws={'shrink':0.5})
p = plt.title("Confusion Matrix of Multinomial Naive Bayes Model")
p = plt.xlabel('Actual Labels')


#decision tree model

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
cv1=cross_val_score(dt,x,y,scoring='f1')
cv1=cross_val_score(dt,x,y,scoring='f1',cv=10)
print(round(cv1.mean(),3))
print(round(cv1.std(),3))
dt.fit(x_train,y_train)
y_pred1=dt.predict(x_test)

cm=confusion_matrix(y_test,y_pred1)
plt.figure(figsize=(8, 8))
axis_labels = ['ham', 'spam']
g = sns.heatmap(data=cm, xticklabels=axis_labels, yticklabels=axis_labels,annot=True,fmt='g', cmap='Blues',cbar_kws={'shrink':0.5})
p = plt.title("Confusion Matrix of Multinomial Naive Bayes Model")
p = plt.xlabel('Actual Labels')

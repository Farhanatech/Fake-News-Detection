## Loading Datasets:
import pandas as pd

fake=pd.read_csv('fake.csv')
real=pd.read_csv('true.csv')
#print(fake.head())
#print(real.head())


fake['label']=0
real['label']=1

##Merge and Cleaning data:
df=pd.concat([fake,real])
df=df.sample(frac=1).reset_index(drop=True)

df=df.drop(['title','subject','date'],axis=1)

##Text Preprocessing:
import re

def preprocess_text(text):
    text=text.lower()
    text=re.sub(r'[^\w\s]','',text)
    return text

df['text']=df['text'].apply(preprocess_text)

##Feature Extraction:
from sklearn.feature_extraction.text import TfidfVectorizer

vector=TfidfVectorizer(max_features=5000)
X=vector.fit_transform(df['text'])
y=df['label']

##Spliting and Training the model:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(X_train,y_train)


y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_acc)

#save the model
import joblib

joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vector, 'tfidf_vectorizer.pkl')


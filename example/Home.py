
#importing the modules

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import joblib
from sklearn.metrics import accuracy_score

#Creating the navbar


#creating an input text box

URL=st.text_input("Enter your URL: ",placeholder='example.com')

#Training the model
# try:
#     pipeline=joblib.load("pipeline.pkl")
#     st.success("Pre-trained model got loaded")
# except:
# st.error('Training the data manually')
try:
    df=pd.read_csv('archive/data.csv')
except:
    st.failure('File not found')

suspicious_words=[
    'free',
    'html',
    'verify',
    'verification',
    'account',
    'secure',
    'update',
    'login',
    'password',
    'bank',
    'confirm',
    'virus',
    'unsafe'
]


df['URL']=df['URL'].str.lower()
df['dot_count']=df['URL'].str.count(r'\.')
df['url_length']=df['URL'].str.len()
df['hyphen_count']=df['URL'].str.count(r'\-')
df['slash_count']=df['URL'].str.count(r'\/')
df['numbers_count']=df['URL'].str.count(r'\d')

counts=[]
for d in df['URL']:
    word=0
    for suspicious_word in suspicious_words:
        if suspicious_word in d:
            word+=1
    counts.append(word)
    
df['suspicious_word_count']=counts



numerical_features=['dot_count',
                    'url_length',
                    'hyphen_count',
                    'slash_count',
                    'suspicious_word_count',
                    'numbers_count']

numerical_transformer=Pipeline([
    ('imputer',SimpleImputer(strategy='median',add_indicator=False)),
    ('scalar',StandardScaler()),
])

preprocessor=ColumnTransformer([
    ('numerical_data',numerical_transformer,numerical_features),
])

pipeline=Pipeline([
    ('preprocessor',preprocessor),
    ('model',LogisticRegression())
])

X=df.drop(columns=['Label'])
y=df['Label']

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# joblib.dump(pipeline,'pipeline.pkl')

cv_score=cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=2,
        scoring='accuracy'
    )
pipeline.fit(X_train,y_train)

#ai code 

# Convert URL to lowercase
url = URL.lower()

# Create features
dot_count = url.count(".")
url_length = len(url)
hyphen_count = url.count("-")
slash_count = url.count("/")
numbers_count = sum(c.isdigit() for c in url)
# Count suspicious words
word_count = 0

for suspicious_word in suspicious_words:
    if suspicious_word in url:
        word_count += 1

# Put everything into a DataFrame
input_data = pd.DataFrame([{
    "dot_count": dot_count,
    "url_length": url_length,
    "hyphen_count": hyphen_count,
    "slash_count": slash_count,
    "suspicious_word_count": word_count,
    'numbers_count':numbers_count
}])


probability=pipeline.predict_proba(input_data)
predictions=pipeline.predict(X_test)
st.write('Train_accuracy',round(cv_score.mean(),2))
accuracy=accuracy_score(y_test,predictions)
st.write('Model Accuracy',round(accuracy,2))
is_safe=probability[0][1]*100
is_notsafe=probability[0][0]*100
st.success(f'Safe probabiity {round(is_safe,2)}%')
st.error(f'Phishing probability {round(is_notsafe,2)}%')
# cv_score=r2_score()
fig,ax=plt.subplots()
ax.scatter(is_notsafe,is_safe)
ax.plot(
    [0,100],
    [0,100],
    '-g',
)
st.pyplot(fig)

if URL:
    result=pipeline.predict(input_data)
    if result=='bad':
        st.error("Phishing Website")
    else:
        st.success('Safe Website')



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {'text': ['Free money win now claim prize','Hi team meeting at 10am','Congratulations you won lottery free','Project report attached','Urgent claim your free gift','Reminder IBM task due','Win cash prize click here','Hello sir minor project update'], 'label': [1,0,1,0,1,0,1,0]}
df = pd.DataFrame(data)
vec = TfidfVectorizer()
X = vec.fit_transform(df['text'])
model = MultinomialNB()
model.fit(X, df['label'])
def predict(text):
    p = model.predict(vec.transform([text]))[0]
    return "SPAM" if p==1 else "HAM"
print("Model Ready")

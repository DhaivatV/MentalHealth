from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.svm import LinearSVC
import pickle

Query= input()

tfidf = TfidfVectorizer(use_idf=True, analyzer='word', stop_words='english', token_pattern=r'\b[^\d\W]+\b', ngram_range=(1,2))
faq =pd.read_csv('/home/dhaivat/SEPM/Mental_Health_FAQ.csv')
faq_quest = faq[['Question_ID', 'Questions']]
faq_answ = faq[['Question_ID', 'Answers']]
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
faq['AnswersEncode'] = label.fit_transform(faq['Answers'])

filename = '/home/dhaivat/SEPM/Mental_health_FAQ.h5'
lsvc = pickle.load(open(filename, 'rb'))
print("Query:", Query)
search_test = [Query]
    
filename1 = "/home/dhaivat/SEPM/vectorizer.pickle"
tfidf = pickle.load(open(filename1, 'rb'))
search_engine = tfidf.transform(search_test)
result = lsvc.predict(search_engine)
    
for question in result:
    faq_data = faq.loc[faq.isin([question]).any(axis=1)]
    Answer = faq_data['Answers'].values
    print(Answer[0])
        

    
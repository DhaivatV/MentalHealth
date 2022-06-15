from doctest import FAIL_FAST
import json
from operator import imod
import pickle
from sklearn.svm import LinearSVC
from fastapi import FastAPI
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


tfidf = TfidfVectorizer(use_idf=True, analyzer='word', stop_words='english', token_pattern=r'\b[^\d\W]+\b', ngram_range=(1,2))
faq =pd.read_csv('/home/dhaivat/SEPM/Mental_Health_FAQ.csv')
faq_quest = faq[['Question_ID', 'Questions']]
faq_answ = faq[['Question_ID', 'Answers']]
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
faq['AnswersEncode'] = label.fit_transform(faq['Answers'])



@app.get ("/")
def index():
    return {'message': 'Hello, Stranger'}

@app.post("/Query/")
async def create_item(Query: str):
 	  
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
        

    return {"Search_result" : str(Answer[0])}
    
    
    
    



if __name__ == "__main__":
     uvicorn.run(app, host= '127.0.0.1', port= 8000)



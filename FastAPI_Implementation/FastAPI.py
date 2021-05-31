# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import uvicorn, os, csv, gzip, json
from torch import load
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles



## SIMILAR QUESTION ##
 
    ## CHECK IF DATASET EXISTS ELSE DOWNLOAD AND CLEAN THE DATASET ##
    
if os.path.exists('C:/Users/utkar/Downloads/GitHub/quora_duplicate_questions_corpus.pt'):
    corpus = load('C:/Users/utkar/Downloads/GitHub/quora_duplicate_questions_corpus.pt',
                  map_location='cpu')
else:
    dataset_url = 'http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv'
    max_corpus = 100000
    dataset_file = "C:/Users/utkar/Downloads/GitHub/quora_duplicate_questions.tsv"
    if not os.path.exists(dataset_file):
        util.http_get(dataset_url, dataset_file)
    
    ## CLEANING THE DATASET ## 
        
    corpus = set()

    with open(dataset_file, encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if(len(corpus) < max_corpus):
                corpus.add(row['question1'])
            else:
                break
            if(len(corpus) < max_corpus):
                corpus.add(row['question2'])
            else:
                break
    
    corpus = list(corpus)
        
    ## IF MODEL IS ALREADY DOWNLOADED, LOAD IT ELSE DOWNLOAD THE MODEL ##
    
if os.path.exists('C:/Users/utkar/Downloads/GitHub/Saved_model_Similar_Question_Retrieval.pt'):
    encoder = load('C:/Users/utkar/Downloads/GitHub/Saved_model_Similar_Question_Retrieval.pt',
                         map_location='cpu')
else:
    model_name = 'quora-distilbert-multilingual'
    encoder = SentenceTransformer(model_name)
    
    ## IF EMBEDDINGS DO NOT EXIST, COMPUTE THEM FROM SCRATCH ##
    
if not os.path.exists('C:/Users/utkar/Downloads/GitHub/quora_duplicate_questions_embeddings.pt'):
    corpus_embeddings_test = encoder.encode(corpus, show_progress_bar=True, 
                                       convert_to_tensor=True)
else:   
    corpus_embeddings = load('C:/Users/utkar/Downloads/GitHub/quora_duplicate_questions_embeddings.pt',
                               map_location='cpu')

## SIMILAR QUESTION ##



## ANSWER RETRIEVAL##

    ## CHECK IF DATASET EXISTS ELSE DOWNLOAD AND CLEAN IT FROM SCRATCH ##
passages_file = 'C:/Users/utkar/Downloads/GitHub/simplewiki-2020-11-01-msmarco-distilberlt-base-v2-passages.pt'
    
if os.path.exists(passages_file):
    passages = load(passages_file, map_location='cpu')
        
else:
    dataset_url = 'http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz'
    passages_file = 'C:/Users/utkar/Downloads/GitHub/'
    util.http_get(dataset_url, passages_file)
    
    passages = []
    with gzip.open(passages_file, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            data = json.loads(line.strip())
            passages.extend(data['paragraphs'])
    
model_file = 'C:/Users/utkar/Downloads/GitHub/simplewiki-2020-11-01-msmarco-distilberlt-base-v2-bi_encoder.pt'
if os.path.exists(model_file):
    bi_encoder = load(model_file, map_location='cpu')
else:
    model_name = 'msmarco-distilbert-base-v2'
    bi_encoder = SentenceTransformer(model_name)
    
    ## CHECK IF CROSS_ENCODER PATH EXISTS, DOWNLOAD IT IF NOT ##
    
cross_file = 'C:/Users/utkar/Downloads/SopraSteria/simplewiki-2020-11-01-msmarco-distilberlt-base-v2-cross_encoder.pt'
if os.path.exists(cross_file):
    cross_encoder = load(cross_file, map_location='cpu')
else:
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')
    
    ## IF EMBEDDINGS FILE EXISTS THEN LOAD IT ELSE COMPUTE FROM SCRATCH ##
    
embeddings_file = 'C:/Users/utkar/Downloads/GitHub/simplewiki-2020-11-01-msmarco-distilberlt-base-v2.pt'
if os.path.exists(embeddings_file):
    answers_embeddings = load(embeddings_file, map_location='cpu')
else:
    answers_embeddings = bi_encoder.encode(passages, show_progress_bar=True, convert_to_tensor=True)

## ANSWER RETRIEVAL ##
     

### CREATING API ###

app = FastAPI()


### FOR USING STATIC FILES ###

app.mount("/static", StaticFiles(directory="static"), name="static")

### FOR LOADING HTML PAGES ###

templates = Jinja2Templates(directory="Templates")



### HOME PAGE ###

@app.get('/', response_class=HTMLResponse)
async def first(request: Request):
    return templates.TemplateResponse("output.html", {"request": request})



### PREDICTIONS PAGE ###

@app.post('/GetPrediction')
async def get_similar(request: Request, question: str = Form(...), response_class=HTMLResponse):
    question = question.strip()
    query_embedding = encoder.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)  
    hits = hits[0]
    similar_questions = []
    for hit in hits:
        similar_questions.append(corpus[hit['corpus_id']])
        
    ## SIMILAR QUESTIONS FINISHED ##
    
    query_embedding = bi_encoder.encode(question, convert_to_tensor=True)
    hits1 = util.semantic_search(query_embedding, answers_embeddings, top_k=10)
    hits1 = hits1[0]
        
    cross_pairs = [[question, passages[hit['corpus_id']]] for hit in hits1]
    cross_scores = cross_encoder.predict(cross_pairs)

    #Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits1[idx]['cross-score'] = cross_scores[idx]
            
    hits1 = sorted(hits1, key=lambda x: x['cross-score'], reverse=True)
    relevant_answers = []
    for hit in hits1[:5]:
        relevant_answers.append(passages[hit['corpus_id']])
    return templates.TemplateResponse("output.html", {"request": request, 
                                                      "similar_questions": similar_questions, 
                                                      "relevant_answers": relevant_answers
                                                      })


if __name__ == '__main__':
    uvicorn.run(app, host='<Your IP ADDRESS HERE>', port=8000)

  
    

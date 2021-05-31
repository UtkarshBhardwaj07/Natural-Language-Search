# -*- coding: utf-8 -*-
"""
Created on Mon May 24 16:41:40 2021

@author: utkarsh
"""
import streamlit as st
import os, csv
from torch import load, cat, multinomial
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import top_k_top_p_filtering, AutoModelForCausalLM, AutoTokenizer
from torch.nn import functional as F
import time

st.title('Natural Language Search')

navbar = st.sidebar.radio('List of functionalities', ['Prediction',
                                                      'Similar Questions',
                                                      'Get Answers'])
if( navbar == 'Prediction'):
    ## IF MODEL EXISTS THEN LOAD IT ELSE DOWNLOAD IT ##
    
    if os.path.exists('C:/Users/utkar/Downloads/GitHub/Next-Word-Prediction-Tokenizer.pt'):
        tokenizer = load('C:/Users/utkar/Downloads/GitHub/Next-Word-Prediction-Tokenizer.pt', 
                       map_location='cpu')

        model = load('C:/Users/utkar/Downloads/SopraSteria/Next-Word-Prediction-GPT2.pt',
                   map_location='cpu')
    else:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
     
    prediction = st.text_input("", "Type here")
    samples = st.slider('How many predictions you want ?', 1, 3, 1)
    if( st.button("Get Prediction") and prediction):
        prediction = prediction.strip()
        with st.spinner('Text successfully registered, calculating prediction'):
            progress_bar = st.progress(0)
            for percent in range(100):
                time.sleep(0.0001)
                progress_bar.progress(percent + 1) 
                
            input_ids = tokenizer.encode(prediction, return_tensors="pt")

        # get logits of last hidden state
            next_token_logits = model(input_ids)[0][:, -1, :]

            filtered_next_token_logits = top_k_top_p_filtering(next_token_logits,
                                                               top_k=50, top_p=1.0)

        # no.of samples
            probs = F.softmax(filtered_next_token_logits, dim=-1)
            next_token = multinomial(probs, num_samples=samples)

            generated = cat([input_ids, next_token], dim=-1)

            resulting_string = tokenizer.decode(generated.tolist()[0])
        st.success('Done')
        st.header(resulting_string)
    else:
        st.error("Please enter some text")
        
if( navbar == 'Similar Questions'):
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
    
    
    similar = st.text_input("", "Type Here")
    samples = st.slider('How many similar questions you want ?', 1, 10, 1)
    if( st.button('Get Questions') and similar):
        with st.spinner('Text successfully registered, calculating similar questions'):
            progress_bar = st.progress(0)
            for percent in range(100):
                time.sleep(0.00001)
                progress_bar.progress(percent + 1)
            query_embedding = encoder.encode(similar, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, corpus_embeddings,
                                        top_k=samples)
            hits = hits[0]
            
        st.success('Done')
        for hit in hits:
            st.subheader(corpus[hit['corpus_id']])
            st.subheader("Similarity Score:  {}".format((hit['score']*100)))
    else:
        st.error("Please enter a question to get it's related questions")
        
if( navbar == 'Get Answers'):
    
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
    
    answer = st.text_input("", "Type Here")
    samples = st.slider('How many answers you want ?', 1, 5, 1)
    if( st.button("Get relevant answers") and answer):
        with st.spinner('Question successfully registered, getting the answers'):
            answer = answer.strip()
            query_embedding = bi_encoder.encode(answer, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, answers_embeddings,
                                        top_k=samples)
            hits = hits[0]
            
            progress_bar = st.progress(0)
            for percent in range(100):
                time.sleep(0.000001)
                progress_bar.progress(percent + 1)
        
            cross_pairs = [[answer, passages[hit['corpus_id']]] for hit in hits]
            cross_scores = cross_encoder.predict(cross_pairs)
        
            #Sort results by the cross-encoder scores
            for idx in range(len(cross_scores)):
                hits[idx]['cross-score'] = cross_scores[idx]
            
            
            hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        st.success('Done')
        for hit in hits[:samples]:
            st.subheader(passages[hit['corpus_id']])
            st.subheader("Similarity Score:  {}".format((hit['score']*100)))
    else:
        st.error("Please enter a question first !")
    

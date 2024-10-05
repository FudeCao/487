import numpy as np
import scipy
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import gensim.downloader
embed = gensim.downloader.load("glove-wiki-gigaword-100")
nltk.download('punkt')          # For tokenization
nltk.download('averaged_perceptron_tagger') # For part-of-speech tagging
nltk.download('maxent_ne_chunker')         # For named entity recognition
nltk.download('words') 

import json

with open('train-v2.0.json', 'r') as file:
   data = json.load(file)    

from collections import defaultdict

# Extract question, answer (the first), and the correspoding context
data_dict = defaultdict(lambda: None)
counter = 0
for passage in data['data']:
    for paragraph in passage['paragraphs']:
        id = counter
        counter += 1
        context = paragraph['context']
        data_dict[id] = { 'context': context, 'questions': [], 'answers': []}
        for qa in paragraph['qas']:
            question = qa['question']
            if qa['is_impossible'] == False:
                answer = qa['answers'][0]
                answer_text = answer['text']
                answer_start_pos = answer['answer_start']
            else:
                answer = qa['answers']
            data_dict[id]['questions'].append(question)
            data_dict[id]['answers'].append(answer)

# Answer start in sentence
for key in data_dict.keys():
    sentences=sent_tokenize(data_dict[key]['context'])
    # print(data_dict[key]['answers'][0])
    count = 0
    i = 0
    for idx, answer in enumerate(data_dict[key]['answers']):
        if answer:
            data_dict[key]['answers'][idx]['answer_start_sentence'] = -1
    for sentence in sentences:
        count += len(sentence)
        #print((i, count))
        for idx, answer in enumerate(data_dict[key]['answers']):
            # if answer and 'answer_start_sentence' not in answer.keys():
            if answer and answer['answer_start_sentence'] == -1:
                if data_dict[key]['answers'][idx]['answer_start'] <= count:
                    data_dict[key]['answers'][idx]['answer_start_sentence'] = i   
        i += 1

# Feature 1-3: 1,2,3-gram
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import numpy as np
def ngram(sentence1,sentence2):
    sentence1=sentence1.lower()
    sentence2=sentence2.lower()

    sentence1 = word_tokenize(sentence1)
    sentence2 = word_tokenize(sentence2)

    bigrams1 = list(ngrams(sentence1, 2))
    bigrams2 = list(ngrams(sentence2, 2))
    trigrams1 = list(ngrams(sentence1, 3))
    trigrams2 = list(ngrams(sentence2, 3))
    
    result=[]
    count=0
    for word in sentence1:
        if word in sentence2:
            count+=1
    result.append(count)

    count=0
    for bigram in bigrams1:
        if bigram in bigrams2:
            count+=1
    result.append(count)

    count=0
    for trigram in trigrams1:
        if trigram in trigrams2:
            count+=1
    result.append(count)
    return result

def ngram_idx(question,context):
    sentences = sent_tokenize(context)
    result=[]
    for sentence in sentences:
        result.append(ngram(sentence,question))
    result=np.array(result)
    max_elements=np.max(result,axis=0)
    max_indices = np.argmax(result, axis=0)
    return max_elements,max_indices

# Feature 4: cosine similarities
# Helper function
def sentence_embedding(s):
    words = word_tokenize(s)
    result = [0]*len(embed['hello'])
    for word in words:
        if word not in embed:
            result += embed['unk']
        else:
            word = word.lower()
            embedding = embed[word]
            result += embedding
    return result

def cosine_similarity_idx(question, context):
# First, obtain a sentence embedding for each sentence
    # 1. import GloVe embeddings
    # 2. sent_tokenize context
    sentences = sent_tokenize(context)
    # 3. obtain sentence embeddings: summation of word embeddings
    sentence_embeddings = []
    question_embedding = sentence_embedding(question)
    for sentence in sentences:
        sentence_embeddings.append(sentence_embedding(sentence))

# Second, calculate cosine similaritis
# Third, find highest cosine similary
    current_highest = 0
    current_highest_idx = 0
    q_e = np.array(question_embedding)
    i = 0
    for s_e in sentence_embeddings:
        s_e = np.array(s_e)
        cosine_sim = np.dot(q_e, s_e)
        divide = np.linalg.norm(q_e) * np.linalg.norm(s_e)
        cosine_sim = cosine_sim / divide
        if cosine_sim > current_highest:
            current_highest = cosine_sim
            current_highest_idx = i
        i += 1
    return (current_highest_idx, current_highest)

# Return idx, highest cosine similarity

# Feature 5: name-entity count
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        if current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    return continuous_chunk

def name_entity(question, context):
    question_entities = get_continuous_chunks(question)
    for ent in question_entities:
        #print(ent)
        continue
    sentences = sent_tokenize(context)
    matched_entity_count = []
    for sentence in sentences:
        sentence_entities = get_continuous_chunks(sentence)
        entity_count = 0
        for qe in question_entities:
            for se in sentence_entities:
                if qe == se:
                    entity_count += 1
        matched_entity_count.append(entity_count)
    matched_entity_count = np.array(matched_entity_count)
    #print(matched_entity_count)
    max_count = np.max(matched_entity_count, axis=0)
    max_idx = np.argmax(matched_entity_count, axis=0)
    #print(max_count,max_idx)
    return (max_count, max_idx)

# Get embedding for each question
# id: idx_embedding, val_embedding, answer_start, is_impossible
id = 0
question_dict = defaultdict(lambda: None)
for passage in data['data']:
    for paragraph in passage['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            question_dict[id] = {}
            
            f1_idx = ngram_idx(qa['question'], context)[1]
            f2_idx = cosine_similarity_idx(qa['question'], context)[0]
            idx_emb = np.append(f1_idx, f2_idx)
            question_dict[id]['idx_embedding'] = idx_emb

            f1_val = ngram_idx(qa['question'], context)[0]
            f2_val = cosine_similarity_idx(qa['question'], context)[1]
            val_emb = np.append(f1_val, f2_val)
            question_dict[id]['val_embedding'] = val_emb

            question_dict[id]['is_impossible'] = qa['is_impossible']
            if qa['is_impossible'] == False:
                question_dict[id]['answer_start'] = qa['answers'][0]['answer_start']
                question_dict[id]['is_impossible'] = 0
            else:
                question_dict[id]['answer_start'] = -1
                question_dict[id]['is_impossible'] = 1
            id += 1

# id: context, question, idx_embedding, val_embedding, answer_start, answer_start_sentence, is_impossible
question_dict = {}
id = 0
question_dict = defaultdict(lambda: None)
for key in data_dict:
    context = data_dict[key]['context']
    for i, question in enumerate(data_dict[key]['questions']):
        question_dict[id] = {}
        question_dict[id]['context'] = context
        question_dict[id]['question'] = question
        answer = data_dict[key]['answers'][i]
        if answer:
            answer_start = answer['answer_start']
            answer_start_sentence = answer['answer_start_sentence']
            is_impossible = 0
        else:
            answer_start = -1
            answer_start_sentence = -1
            is_impossible = 1
        question_dict[id]['answer_start'] = answer_start
        question_dict[id]['answer_start_sentence'] = answer_start_sentence
        question_dict[id]['is_impossible'] = is_impossible
        
        f1_idx = ngram_idx(question, context)[1]
        f2_idx = cosine_similarity_idx(question, context)[0]
        f3_idx = name_entity(question, context)[1]
        idx_emb = np.append(f1_idx, f2_idx)
        idx_emb = np.append(idx_emb, f3_idx)
        question_dict[id]['idx_embedding'] = idx_emb

        f1_val = ngram_idx(question, context)[0]
        f2_val = cosine_similarity_idx(question, context)[1]
        f3_val = name_entity(question, context)[0]
        val_emb = np.append(f1_val, f2_val)
        val_emb = np.append(val_emb, f3_val)
        question_dict[id]['val_embedding'] = val_emb

        question_dict[id]['embedding'] = np.append(idx_emb, val_emb)
        
        id += 1
print(question_dict[0])   

# Write to json dictionary to store for use
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

json_object = json.dumps(question_dict, indent=4, cls=NpEncoder)
# Writing to sample.json
with open("sample_100.json", "w") as outfile:
    outfile.write(json_object)
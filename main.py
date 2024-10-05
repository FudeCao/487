import numpy as np
import scipy
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim.downloader
import nltk
import json
from collections import defaultdict
from models import most_possible_answer_idx, regression, answer_idx, mlpclassifier
from data_structure import ngram_idx, name_entity, cosine_similarity_idx
from sklearn.model_selection import train_test_split

with open('dev-v2.0.json', 'r') as file:
   test = json.load(file)    

print(len(test['data']))

test_dict = defaultdict(lambda: None)
counter = 0
for passage in test['data']:
    for paragraph in passage['paragraphs']:
        id = counter
        counter += 1
        context = paragraph['context']
        test_dict[id] = { 'context': context, 'questions': [], 'answers': []}
        for qa in paragraph['qas']:
            question = qa['question']
            if qa['is_impossible'] == False:
                answer = qa['answers'][0]
                answer_text = answer['text']
                answer_start_pos = answer['answer_start']
            else:
                answer = qa['answers']
            test_dict[id]['questions'].append(question)
            test_dict[id]['answers'].append(answer)
print(test_dict[1])
for key in test_dict.keys():
    sentences=sent_tokenize(test_dict[key]['context'])
    count = 0
    i = 0
    for idx, answer in enumerate(test_dict[key]['answers']):
        if answer:
            test_dict[key]['answers'][idx]['answer_start_sentence'] = -1
    for sentence in sentences:
        count += len(sentence)
        #print((i, count))
        for idx, answer in enumerate(test_dict[key]['answers']):
            # if answer and 'answer_start_sentence' not in answer.keys():
            if answer and answer['answer_start_sentence'] == -1:
                if test_dict[key]['answers'][idx]['answer_start'] <= count:
                    test_dict[key]['answers'][idx]['answer_start_sentence'] = i   
        i += 1
test_question_dict = {}
print(test_dict[1])
id = 0
test_question_dict = defaultdict(lambda: None)
for key in test_dict:
    context = test_dict[key]['context']
    for i, question in enumerate(test_dict[key]['questions']):
        test_question_dict[id] = {}
        test_question_dict[id]['context'] = context
        test_question_dict[id]['question'] = question
        answer = test_dict[key]['answers'][i]
        if answer:
            answer_start = answer['answer_start']
            answer_start_sentence = answer['answer_start_sentence']
            is_impossible = 0
        else:
            answer_start = -1
            answer_start_sentence = -1
            is_impossible = 1
        test_question_dict[id]['answer_start'] = answer_start
        test_question_dict[id]['answer_start_sentence'] = answer_start_sentence
        test_question_dict[id]['is_impossible'] = is_impossible
        
        f1_idx = ngram_idx(question, context)[1]
        f2_idx = cosine_similarity_idx(question, context)[0]
        f3_idx = name_entity(question, context)[1]
        idx_emb = np.append(f1_idx, f2_idx)
        idx_emb = np.append(idx_emb, f3_idx)
        test_question_dict[id]['idx_embedding'] = idx_emb

        f1_val = ngram_idx(question, context)[0]
        f2_val = cosine_similarity_idx(question, context)[1]
        f3_val = name_entity(question, context)[0]
        val_emb = np.append(f1_val, f2_val)
        val_emb = np.append(val_emb, f3_val)
        test_question_dict[id]['val_embedding'] = val_emb

        test_question_dict[id]['embedding'] = np.append(idx_emb, val_emb)
        
        id += 1
print(test_question_dict[0]['embedding'])

with open('sample.json', 'r') as file:
    question_dict = json.load(file)

# Classfier for if there's an answer for the question
from sklearn.model_selection import train_test_split
question_value_embedding_list = []
is_impossible_list = []
for id in question_dict:
    if question_dict[id]!=None:
        question_value_embedding_list.append(question_dict[id]['val_embedding'])
        is_impossible_list.append(question_dict[id]['is_impossible'])
X_train, X_test, y_train, y_test = train_test_split(question_value_embedding_list, is_impossible_list, test_size=0.3)

id = 0
test_question_value_embedding_list = []
#test_is_impossible_list = []
for id in test_question_dict:
    if test_question_dict[id]!=None:
        test_question_value_embedding_list.append(test_question_dict[id]['val_embedding'])
        #test_is_impossible_list.append(test_question_dict[id]['is_impossible'])
clf=mlpclassifier(X_train, y_train)
test_answerable_list = clf.predict(test_question_value_embedding_list)

question_feature_list = []
start_sentence_list = []
for id in question_dict:
    if question_dict[id]!=None:
        if question_dict[id]['is_impossible']==0:
            question_feature_list.append(np.concatenate((question_dict[id]['val_embedding'],question_dict[id]['idx_embedding'])))
            start_sentence_list.append(question_dict[id]['answer_start_sentence'])
X_train, X_test, y_train, y_test = train_test_split(question_feature_list, start_sentence_list, test_size=0.3)

test_question_feature_list = []
#test_start_sentence_list = []
for id in test_question_dict:
    if test_question_dict[id]!=None:
        #if test_question_dict[id]['is_impossible']==0:
        test_question_feature_list.append(np.concatenate((test_question_dict[id]['val_embedding'],test_question_dict[id]['idx_embedding'])))
            #test_start_sentence_list.append(question_dict[id]['answer_start_sentence'])
reg=regression(X_train,y_train)
test_sentence_idx_list=reg.predict(test_question_feature_list)
test_sentence_idx_list=[round(elem, 0) for elem in test_sentence_idx_list]
test_sentence_idx_list=np.array(test_sentence_idx_list, dtype=int)    
answer_idx_list=[]
for id in test_question_dict:
    if test_question_dict[id]!=None:
        if len(sent_tokenize(test_question_dict[id]['context'])) <= test_sentence_idx_list[id]:
            test_sentence_idx_list[id]=len(sent_tokenize(test_question_dict[id]['context']))-1
        sentence = sent_tokenize(test_question_dict[id]['context'])[test_sentence_idx_list[id]]
        word_idx = most_possible_answer_idx(test_question_dict[id]['question'], sentence)
        if word_idx == -1:
            test_answerable_list[id] = 1
        answer_idx_list.append(answer_idx(test_sentence_idx_list[id], test_question_dict[id]['context'], word_idx))

rough_accuracy=0
exact_accuracy=0
answerable_accuracy=0
count= 0
precision_deno=0
precision=0
recall =0
recall_deno=0
   
for id in test_question_dict:
    count+=1
    if test_answerable_list[id] == 0 and test_question_dict[id]['is_impossible']==0:
        if answer_idx_list[id]-test_question_dict[id]['answer_start'] <= 25 and answer_idx_list[id]-test_question_dict[id]['answer_start'] >= -25:
            rough_accuracy +=1
        if answer_idx_list[id] == test_question_dict[id]['answer_start']:
            exact_accuracy +=1
        answerable_accuracy +=1
        recall +=1
    else:
        if test_answerable_list[id] == 1 and test_question_dict[id]['is_impossible'] == 1:
            answerable_accuracy +=1
            exact_accuracy +=1
            rough_accuracy +=1
            precision += 1
    if test_answerable_list[id] == 1:
        precision_deno +=1
    if test_answerable_list[id] == 0:
        recall_deno +=1
precision = precision/precision_deno
recall = recall / recall_deno
F1 = 2*precision*recall/(precision + recall)
rough_accuracy = rough_accuracy/count
exact_accuracy = exact_accuracy/count
answerable_accuracy = answerable_accuracy/count
print(rough_accuracy)
print(exact_accuracy)
print(answerable_accuracy)
print(precision)
print(recall)
print(F1)
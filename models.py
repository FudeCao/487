import numpy as np
import scipy
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim.downloader
import nltk
import json
from data_structure import get_continuous_chunks, cosine_similarity_idx

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

from sklearn.neural_network import MLPClassifier
def mlpclassifier(X,y):
    clf = MLPClassifier(hidden_layer_sizes=50,max_iter=300,alpha=0.1).fit(X,y)
    return clf

from sklearn.neighbors import KNeighborsClassifier
def knn(X,y):
    neigh = KNeighborsClassifier(n_neighbors=500).fit(X,y)
    return neigh
neigh=knn(X_train,y_train)
print(neigh.score(X_test,y_test))

import torch
import torch.nn as nn
class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_function = torch.sigmoid
        self.linear1 = nn.Linear(5, 500)
        self.linear2 = nn.Linear(500, 100)
        self.linear3 = nn.Linear(100, 2)

    def forward(self, x):
        x=self.linear1(x)
        x=self.activation_function(x)
        x=self.linear2(x)
        x=self.activation_function(x)
        x=self.linear3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Setup the device for tensor to be stored

import torch.optim as optim
model=classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for X,y in zip(X_train,y_train):
    X=torch.tensor(X).float().to(device)
    y=torch.tensor(y).to(device)
    optimizer.zero_grad()
    output=model.forward(X).to(device)
    loss=criterion(output,y)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    correct=0
    total=0
    for X,y in zip(X_train,y_train):
        X = torch.tensor(X).float().to(device)
        y = torch.tensor(y).to(device)
        output=model.forward(X).to(device)
        max_index = torch.argmax(output)
        if max_index==y:
            correct+=1
        total+=1
    test_accuracy=correct/total

# Model for predicting starting sentence
question_feature_list = []
start_sentence_list = []
for id in question_dict:
    if question_dict[id]!=None:
        if question_dict[id]['is_impossible']==0:
            question_feature_list.append(np.concatenate((question_dict[id]['val_embedding'],question_dict[id]['idx_embedding'])))
            start_sentence_list.append(question_dict[id]['answer_start_sentence'])
X_train, X_test, y_train, y_test = train_test_split(question_feature_list, start_sentence_list, test_size=0.3)

from sklearn.linear_model import LinearRegression 
def regression(X,y):
    reg = LinearRegression().fit(X, y)
    return reg

reg=regression(X_train,y_train)
#reg.predict()
#print(reg.predict([X_train[0], X_train[1], X_train[50]]))
#print(reg.score(X_test,y_test))

# Predict the correct word from a sentence, returns the index of word
def most_possible_answer_idx(question, sentence):
    list_of_ne = get_continuous_chunks(sentence)
    questionNE = get_continuous_chunks(question)
    new_list_of_ne = []
    for ne in list_of_ne:
        for ne1 in questionNE:
            if ne != ne1:
                new_list_of_ne.append(ne)
    print(new_list_of_ne)
    nes = ''
    for ne in new_list_of_ne:
        nes += ne
        nes += ' '

    idx = cosine_similarity_idx(question, nes)[0]
    counter = 0
    res_idx = 0
    for word in word_tokenize(sentence):
        if word == new_list_of_ne[idx]:
            res_idx = counter
        counter += 1
    return res_idx

def answer_idx(sentence_idx, context, word_idx):
    if word_idx == -1:
        return -1
    sentences=sent_tokenize(context)
    start_pos = 0
    for idx in range(sentence_idx):
        start_pos += len(sentences[idx])
    start_pos += word_idx
    return start_pos
# -*- coding: utf-8 -*-
"""
Building a ChatBot with Deep NLP

@author: Bogdan
"""

#Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time

#Importing the dataset
lines = open("movie_lines.txt", encoding = "utf_8", errors = "ignore").read().split("\n")
conversations = open("movie_conversations.txt", encoding = "utf_8", errors = "ignore").read().split("\n")

#Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[4] 

#Creating a list of all the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(","))
    
#Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) -1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
#Doing a first celeaning of the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

#Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
#Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
    
#Create a dictionary that maps each word to its number of occurences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
    
#Creating two dictionaries that map the question words and the answer words to a unique integer
threshold = 20
questionwords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionwords2int[word] = word_number
        word_number += 1

answerwords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerwords2int[word] = word_number
        word_number += 1
    
#Adding the last tokens to these two dictionaries
tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]
for token in tokens:
    questionwords2int[token] = len(questionwords2int) + 1
    
for token in tokens:
    answerwords2int[token] = len(answerwords2int) + 1

#Creating the inverse dictionary of the answerswords2int dictionary
answersints2words = {w_i: w for w, w_i in answerwords2int.items() }
    
#Adding the End Of String token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>" 

#Translating all the questions and the asnwers into integers
# and Replacing all the words that were filtered out with <OUT>
questions_into_int = []
for questions in clean_questions:
    ints = []
    for word in questions.split():
        if word not in questionwords2int:
            ints.append(questionwords2int["<OUT>"])
        else:
            ints.append(questionwords2int[word])
    questions_into_int.append(ints)

answers_into_int = []
for answers in clean_questions:
    ints = []
    for word in answers.split():
        if word not in answerwords2int:
            ints.append(answerwords2int["<OUT>"])
        else:
            ints.append(answerwords2int[word])
    answers_into_int.append(ints)

#Sorting questions and answers by the length of the questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])


# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = "input")
    targets = tf.placeholder(tf.int32, [None, None], name = "target")
    lr = tf.placeholder(tf.float32, name = "learning_rate")
    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
    
    return inputs, targets, lr, keep_prob

# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int["<SOS>"])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    
    return preprocessed_targets

# Creating the Encoder RNN Layer
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)





















 
    
    
    
    
    
    
    







    
    
    
    
    
    
    
    
    
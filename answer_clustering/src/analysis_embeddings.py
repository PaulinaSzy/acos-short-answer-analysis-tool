#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import nltk
import string
import sklearn
import timeit
from sklearn.cluster import DBSCAN
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download("wordnet")
nltk.download("punkt")
from nltk.corpus import stopwords
import pickle as pkl
import difflib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import tensorflow_hub as hub

from . import functions as f


def tag_answer(answer):
    tokens = nltk.word_tokenize(answer)
    return nltk.pos_tag(tokens)


def clean_answer(answer):
    exclist = string.punctuation
    table = str.maketrans("", "", exclist)
    return answer.translate(table).lower()


stop_words = set(stopwords.words("english"))


def remove_stopwords_answer(answer_lemmatized):
    not_stopword = lambda s: s not in stop_words
    filtered = list(filter(not_stopword, answer_lemmatized))
    return filtered


def get_wordnet_tag(nltk_tag):
    if nltk_tag.startswith("J"):
        return wordnet.ADJ
    elif nltk_tag.startswith("V"):
        return wordnet.VERB
    elif nltk_tag.startswith("N"):
        return wordnet.NOUN
    elif nltk_tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def lemmatize_answer(answer_tagged):
    lemmatizer = WordNetLemmatizer()
    wordnet_tagged = map(lambda x: (x[0], get_wordnet_tag(x[1])), answer_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return lemmatized_sentence


def run_pipeline(answer):
    return remove_stopwords_answer(lemmatize_answer(tag_answer(clean_answer(answer))))


def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


from numpy import dot
from numpy.linalg import norm


def create_similarity_matrix(embedding_list):
    size = len(embedding_list)
    similarity_matrix = np.zeros((size, size))
    for i in range(0, len(embedding_list)):
        # print("Iteration: {0}/{1}".format(i, size))
        for j in range(0, i):
            similarity_matrix[i][j] = cosine(embedding_list[i], embedding_list[j])
    return similarity_matrix


def get_w2v_model(answers, task_name):
    model = Word2Vec(answers, min_count=1, vector_size=100)
    model.save("w2c" + task_name)
    return model


def vectorize_sentence(sent, model):
    sent_vec = np.zeros(100)
    for w in sent:
        # print(model[w])
        try:
            sent_vec = np.add(sent_vec, model[w])
        except:
            pass
    # print(sent_vec)
    return sent_vec / np.sqrt(sent_vec.dot(sent_vec))


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def create_feature_vectors(data):
    style_exercise_names = list(set(data["problemName"].tolist()))
    # if analysis_type == "error":
    #     data = data.loc[data["answer"].apply(f.is_answer_correctly_spelled)]
    data_per_exercise = {}
    for ex in style_exercise_names:
        ex_data = {}
        all_data = data.loc[data["problemName"] == ex]
        answers = all_data["answer"].tolist()
        closest = all_data["closest"].tolist()
        combined_answers = list(set(answers + closest))
        ex_data["data"] = combined_answers
        embeddings = embed(combined_answers)
        ex_data["embeddings"] = embeddings
        ex_data["matrix"] = create_similarity_matrix(embeddings)
        data_per_exercise[ex] = ex_data

    return data_per_exercise


def run(alldata):
    data_per_exercise = create_feature_vectors(alldata)
    for analysis_type in ["error", "progress"]:
        with open(
            "data-processed/data_per_exercise_style_embedding_" + analysis_type, "wb"
        ) as pkl_file:
            pkl.dump(data_per_exercise, pkl_file)

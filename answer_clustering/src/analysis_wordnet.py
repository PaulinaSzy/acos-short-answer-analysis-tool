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
from nltk.corpus import stopwords
import pickle as pkl
import difflib
from . import functions as f

# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

stop_words = set(stopwords.words("english"))

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', 80)


def filter_test_answers(data, allowed_missing_words_num):
    data["closest_splitted_len"] = data["closest"].apply(
        lambda elem: len(elem.split(" "))
    )
    data["answer_splitted_len"] = data["answer"].apply(
        lambda elem: len(elem.split(" "))
    )
    return data.loc[
        data["answer_splitted_len"]
        >= data["closest_splitted_len"] - allowed_missing_words_num
    ]


def clean_answer(answer):
    exclist = string.punctuation + string.digits
    table = str.maketrans("", "", exclist)
    return answer.translate(table).lower()


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


def tag_answer(answer):
    tokens = nltk.word_tokenize(answer)
    return nltk.pos_tag(tokens)


def lemmatize_answer(answer_tagged):
    lemmatizer = WordNetLemmatizer()
    wordnet_tagged = map(lambda x: (x[0], get_wordnet_tag(x[1])), answer_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append((word, tag))
        else:
            lemmatized_sentence.append((lemmatizer.lemmatize(word, tag), tag))
    return lemmatized_sentence


def remove_stopwords_answer(answer_lemmatized):
    not_stopword = lambda s: s[0] not in stop_words
    filtered = list(filter(not_stopword, answer_lemmatized))
    return filtered


def get_synsets_answer(lemma_tag_pairs_filtered):
    synset_list = []
    for lemma, tag in lemma_tag_pairs_filtered:
        syns = wordnet.synsets(lemma, pos=tag)
        if syns:
            synset_list.append(syns)
    return list(np.concatenate(synset_list).flat)


def run_pipeline(answer):
    return get_synsets_answer(
        remove_stopwords_answer(lemmatize_answer(tag_answer(clean_answer(answer))))
    )


def sim_score(synsets1, synsets2):

    sumSimilarityscores = 0
    scoreCount = 0

    synsets_similarities = {}

    for synset1 in synsets1:

        synsetScore = 0
        similarityScores = []

        for synset2 in synsets2:

            if synset1.pos() == synset2.pos():

                if synset1.name().split(".")[0] == synset2.name().split(".")[0]:
                    synsetScore = 1
                elif (synset1, synset2) in synsets_similarities:
                    synsetScore = synsets_similarities[(synset1, synset2)]
                else:
                    synsetScore = synset1.path_similarity(synset2)
                    synsets_similarities[(synset1, synset2)] = synsets_similarities[
                        (synset2, synset1)
                    ] = synsetScore

                if synsetScore != None:
                    similarityScores.append(synsetScore)

                synsetScore = 0

        if len(similarityScores) > 0:
            sumSimilarityscores += max(similarityScores)
            scoreCount += 1

    if scoreCount > 0:
        avgScores = sumSimilarityscores / scoreCount

    return avgScores


def symmetric_sim_score(synsets1, synsets2):
    return (sim_score(synsets1, synsets2) + sim_score(synsets2, synsets1)) / 2


# import random

# def estimate_time(task):
#     s = data_per_exercise[task]['synsets']
#     synsets_rand_1 = random.sample(s, 100)
#     synsets_rand_2 = random.sample(s, 100)
#     start = timeit.default_timer()
#     for i in range(100):
#          symmetric_sim_score(synsets_rand_1[i],synsets_rand_2[i])
#     end = timeit.default_timer()
#     one_example_time = (end-start)/100
#     total_time = one_example_time*len(s)**2/3600/2
#     return total_time


# total = 0
# for ex in style_exercise_names:
#     t = estimate_time(ex)
#     total += t
#     print(ex +': ', estimate_time(ex))
# print('Total: ', total)


def create_similarity_matrix(synset_list):
    similarity_matrix = np.zeros((len(synset_list), len(synset_list)))
    for i in range(0, len(synset_list)):
        for j in range(0, i):
            similarity_matrix[i][j] = symmetric_sim_score(
                synset_list[i], synset_list[j]
            )
    return similarity_matrix


# def create_feature_vectors(analysis_type):
#     if analysis_type=='error':
#         data = data_order.loc[data_order["answer"].apply(f.is_answer_correctly_spelled)]
#         data.drop_duplicates(subset=["answer"], keep="first", inplace=True)
#     else:
#         data = data_order
#     data_per_exercise = {}
#     for ex in order_exercise_names:
#         print("Exercise: ", format(ex))
#         ex_data = {}
#         all_data = data.loc[data_order['problemName'] == ex]
#         answers = all_data['answer'].tolist()
#         closest = all_data['closest'].tolist()
#         combined_answers = list(set(answers + closest))
#         ex_data['data'] = combined_answers
#         feature_vectors = []
#         for answer in combined_answers:
#             feature_vectors.append(answer_to_vector(answer,'output.tsv'))
#         ex_data['embeddings'] = feature_vectors
#         ex_data['matrix'] = f.create_similarity_matrix(feature_vectors)
#         data_per_exercise[ex] = ex_data

#     with open('data-processed/data_per_exercise_syntax_'+analysis_type, 'wb') as pkl_file:
#         pkl.dump(data_per_exercise, pkl_file)


def create_feature_vectors(data):
    style_exercise_names = list(set(data["problemName"].tolist()))
    # if analysis_type == "error":
    #     data = data.loc[data["answer"].apply(f.is_answer_correctly_spelled)]
    data_per_exercise = {}
    sum_of_lens = 0
    for ex in style_exercise_names:
        ex_data = {}
        all_data = data.loc[data["problemName"] == ex]
        answers = all_data["answer"].tolist()
        closest = all_data["closest"].tolist()
        combined_answers = list(set(answers + closest))
        sum_of_lens += len(combined_answers)
        ex_data["data"] = combined_answers
        synsets = list(map(run_pipeline, combined_answers))
        ex_data["matrix"] = create_similarity_matrix(synsets)
        data_per_exercise[ex] = ex_data

    return data_per_exercise


def run(alldata):
    data_per_exercise = create_feature_vectors(alldata)
    for analysis_type in ["error", "progress"]:
        with open(
            "data-processed/data_per_exercise_style_wordnet_" + analysis_type, "wb"
        ) as pkl_file:
            pkl.dump(data_per_exercise, pkl_file)

import string
from numpy import dot
from numpy.linalg import norm
import numpy as np
import nltk
import json


def clean_answer(answer, remove_all=True):
    punct_list = string.punctuation
    if not remove_all:
        l = punct_list.replace(",", "")
        punct_list = l.replace(".", "")
    exclist = punct_list + "\n\r"
    table = str.maketrans("", "", exclist)
    return answer.translate(table).lower()


def cosine(u, v):
    return dot(u, v) / (norm(u) * norm(v))


def create_similarity_matrix(embedding_list):
    size = len(embedding_list)
    similarity_matrix = np.zeros((size, size))
    for i in range(0, len(embedding_list)):
        #         print("Iteration: {0}/{1}".format(i, size))
        for j in range(0, i):
            similarity_matrix[i][j] = cosine(embedding_list[i], embedding_list[j])
    return similarity_matrix


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


def preprocessAnswer(answer):
    exclist = string.punctuation
    table = str.maketrans(exclist, " " * len(exclist))
    answerCleaned = answer.translate(table).lower()
    return nltk.word_tokenize(answerCleaned)


with open("static/englishdict.json") as json_file:
    data = json.load(json_file)
dictset = set(data)


def isValidEnglishWord(word):
    return word.strip() in dictset


def is_answer_correctly_spelled(answer):
    ans = preprocessAnswer(answer)
    iscorrect = True
    for token in ans:
        if not isValidEnglishWord(token):
            iscorrect = False
            return iscorrect
    return iscorrect

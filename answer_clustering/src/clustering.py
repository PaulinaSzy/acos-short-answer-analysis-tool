#!/usr/bin/env python
# coding: utf-8


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import pickle as pkl
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
from time import time
from sklearn import metrics


def get_data_clustered_per_exercise(data_file, params_file):
    data_per_exercise = pkl.load(open(data_file, "rb"))
    params = pkl.load(open(params_file, "rb"))

    data_clustered = {}

    for exercise in data_per_exercise:
        e = params[exercise]["eps"]
        ms = params[exercise]["min_samples"]
        similarity_matrix = data_per_exercise[exercise]["matrix"]
        similarity_matrix = np.clip(similarity_matrix, -1, 1)
        complete_similarity_matrix = similarity_matrix + similarity_matrix.T
        np.fill_diagonal(complete_similarity_matrix, 1)
        distance_matrix = np.arccos(complete_similarity_matrix) / np.pi

        X_embedded = TSNE(n_components=2, metric="precomputed").fit_transform(
            distance_matrix
        )
        clustering = DBSCAN(eps=e, min_samples=ms).fit(X_embedded)
        clustering_labels = clustering.labels_
        data_tsne = pd.DataFrame(
            list(
                zip(
                    data_per_exercise[exercise]["data"],
                    clustering_labels,
                    X_embedded[:, 0],
                    X_embedded[:, 1],
                )
            ),
            columns=["answer", "label", "x_component", "y_component"],
        )
        data_clustered[exercise] = data_tsne

    return data_clustered


def save_clustered_data(filename, params_file):
    d = get_data_clustered_per_exercise(
        "data-processed/data_per_exercise_" + filename, "data-processed/" + params_file
    )
    with open("data-processed/data_clustered_" + filename, "wb") as pkl_file:
        pkl.dump(d, pkl_file)


# Params selection

data_paths = [
    ("allsyntax_error", "bestparams_allsyntax_error"),
    ("allsyntax_progress", "bestparams_allsyntax_progress"),
    ("constituency_error", "bestparams_constituency_error"),
    ("constituency_progress", "bestparams_constituency_progress"),
    ("dependency_error", "bestparams_dependency_error"),
    ("dependency_progress", "bestparams_dependency_progress"),
    ("sentence_error", "bestparams_sentence_error"),
    ("sentence_progress", "bestparams_sentence_progress"),
    ("style_embedding_error", "bestparams_style_embedding_error"),
    ("style_embedding_progress", "bestparams_style_embedding_progress"),
    ("style_wordnet_error", "bestparams_style_wordnet_error"),
    ("style_wordnet_progress", "bestparams_style_wordnet_progress"),
]


def run():
    for file_path, param_path in data_paths:
        save_clustered_data(file_path, param_path)


# def get_data_for_scoring(data_per_exercise):
#     exercises = {}

#     for exercise in data_per_exercise:
#         similarity_matrix = data_per_exercise[exercise]["matrix"]
#         similarity_matrix = np.clip(similarity_matrix, -1, 1)
#         complete_similarity_matrix = similarity_matrix + similarity_matrix.T
#         np.fill_diagonal(complete_similarity_matrix, 1)
#         distance_matrix = np.arccos(complete_similarity_matrix) / np.pi
#         X_embedded = TSNE(n_components=2, metric="precomputed").fit_transform(
#             distance_matrix
#         )
#         exercises[exercise] = (distance_matrix, X_embedded)

#     return exercises


# def get_random_search_results(distance_matrix, x_embedded):
#     def infinite_datasource():
#         idxs = list(range(x_embedded.shape[0]))
#         yield (idxs, idxs)

#     def silhouette_score(estimator, X):
#         clusters = estimator.fit_predict(X)
#         score = metrics.silhouette_score(
#             distance_matrix, clusters, metric="precomputed"
#         )
#         return score

#     n_iter_search = 1000
#     distributions = {"eps": stats.uniform(0.1, 10), "min_samples": stats.randint(2, 30)}
#     ca = DBSCAN()
#     random_search = RandomizedSearchCV(
#         ca,
#         param_distributions=distributions,
#         n_iter=n_iter_search,
#         scoring=silhouette_score,
#         cv=infinite_datasource(),
#     )
#     start = time()

#     random_search.fit(x_embedded)
#     #     print("RandomizedSearchCV took %.2f seconds for %d candidates"
#     #           " parameter settings." % ((time() - start), n_iter_search))
#     params_fit = random_search.cv_results_["params"]
#     scores_fit = random_search.cv_results_["mean_test_score"]
#     ranks_fit = random_search.cv_results_["rank_test_score"]

#     for (idx, (params, score, rank)) in enumerate(
#         zip(params_fit, scores_fit, ranks_fit)
#     ):
#         #         print('IDX ', idx)
#         #         print("Params: ", params)
#         #         print("Score: ", score)
#         #         print("Rank: ", rank)
#         #         print('---------------')
#         if rank == 1:
#             best_params = params

#     return best_params


# def save_best_params(infile, outfile):
#     data_per_ex = pkl.load(open(infile, "rb"))
#     data_for_scoring = get_data_for_scoring(data_per_ex)
#     best_params_per_ex = {}
#     for exercise in data_for_scoring:
#         print(exercise)
#         best_params = get_random_search_results(
#             data_for_scoring[exercise][0], data_for_scoring[exercise][1]
#         )
#         best_params_per_ex[exercise] = best_params
#     with open(outfile, "wb") as pkl_file:
#         pkl.dump(best_params_per_ex, pkl_file)


# data_paths = [
#     ("data-processed/data_per_exercise_allsyntax_error", "bestparams_allsyntax_error"),
#     (
#         "data-processed/data_per_exercise_allsyntax_progress",
#         "bestparams_allsyntax_progress",
#     ),
#     (
#         "data-processed/data_per_exercise_constituency_error",
#         "bestparams_constituency_error",
#     ),
#     (
#         "data-processed/data_per_exercise_constituency_progress",
#         "bestparams_constituency_progress",
#     ),
#     (
#         "data-processed/data_per_exercise_dependency_error",
#         "bestparams_dependency_error",
#     ),
#     (
#         "data-processed/data_per_exercise_dependency_progress",
#         "bestparams_dependency_progress",
#     ),
#     ("data-processed/data_per_exercise_sentence_error", "bestparams_sentence_error"),
#     (
#         "data-processed/data_per_exercise_sentence_progress",
#         "bestparams_sentence_progress",
#     ),
#     (
#         "data-processed/data_per_exercise_style_embedding_error",
#         "bestparams_style_embedding_error",
#     ),
#     (
#         "data-processed/data_per_exercise_style_embedding_progress",
#         "bestparams_style_embedding_progress",
#     ),
#     (
#         "data-processed/data_per_exercise_style_wordnet_error",
#         "bestparams_style_wordnet_error",
#     ),
#     (
#         "data-processed/data_per_exercise_style_wordnet_progress",
#         "bestparams_style_wordnet_progress",
#     ),
# ]

# for in_path, out_path in data_paths:
#     save_best_params(in_path, out_path)


# params = pkl.load(open("bestparams_allsyntax_error", "rb"))
# params


# import plotly.graph_objects as go
# import plotly.express as px
# from random import randint


# def get_max_cluster_size(data):
#     return data["label"].max()


# def show_graph(d):
#     max_clust = get_max_cluster_size(d)
#     colorsIdx = {}
#     for l in range(max_clust + 1):
#         colorsIdx[f"{l}"] = "#%06X" % randint(0, 0xFFFFFF)
#     colorsIdx["-1"] = "rgb(166, 166, 166)"
#     sizesIdx = {False: 15, True: 30}
#     stylesIdx = {True: "x-open-dot", False: "circle"}
#     # sizs = d['correct'].map(sizesIdx)
#     # stys = d['correct'].map(stylesIdx)
#     # cols = d['label'].map(colorsIdx)
#     fig = go.FigureWidget(
#         data=go.Scatter(
#             x=d["x_component"],
#             y=d["y_component"],
#             mode="markers",
#             marker_color=d["label"]
#             # marker = dict(color=cols),
#         )
#     )
#     fig.update_layout(
#         width=1000,
#         height=1000,
#         hoverdistance=1,
#         dragmode="select",
#     )
#     fig.show()


# # from sklearn.model_selection import GridSearchCV

# # epsilons = np.arange(0.1, 1, 0.05).tolist()
# # min_samples = np.arange(1, 16, 1).tolist()

# # parameters = {'eps':epsilons, 'min_samples':min_samples}


# from sklearn.preprocessing import StandardScaler


# def score_clustering(data, epsilon, samples):
#     similarity_matrix = data["matrix"]
#     similarity_matrix = np.clip(similarity_matrix, -1, 1)
#     complete_similarity_matrix = similarity_matrix + similarity_matrix.T
#     np.fill_diagonal(complete_similarity_matrix, 1)
#     distance_matrix = np.arccos(complete_similarity_matrix) / np.pi

#     X_embedded = TSNE(
#         n_components=2, metric="precomputed", square_distances=True
#     ).fit_transform(distance_matrix)
#     db_distance = DBSCAN(eps=epsilon, min_samples=samples).fit(X_embedded)
#     labels_distance = db_distance.labels_
#     n_clusters_distance = len(set(labels_distance)) - (
#         1 if -1 in labels_distance else 0
#     )
#     n_noise_distance = list(labels_distance).count(-1)

#     print("Epsilon: {}, min_samples: {}".format(epsilon, samples))
#     print("DBSCAN on distance matrix")
#     print("Estimated number of clusters: %d" % n_clusters_distance)
#     print("Estimated number of noise points: %d" % n_noise_distance)
#     score_1 = metrics.silhouette_score(distance_matrix, labels_distance)
#     print("Silhouette Coefficient: %0.3f" % score_1)
#     print("----------------------------")
#     # embeddings = data['embeddings']
#     # data_scaled = StandardScaler().fit_transform(embeddings)
#     # print(len(data_scaled))
#     # print(len(data_scaled[0]))

#     # db_embeddings = DBSCAN(eps=epsilon, min_samples=samples).fit(data_scaled)
#     # labels_embeddings = db_embeddings.labels_
#     # # Number of clusters in labels, ignoring noise if present.
#     # n_clusters_embeddings = len(set(labels_embeddings)) - (1 if -1 in labels_embeddings else 0)
#     # n_noise_embeddings = list(labels_embeddings).count(-1)

#     # print("DBSCAN on scaled embeddings")
#     # print('Estimated number of clusters: %d' % n_clusters_embeddings)
#     # print('Estimated number of noise points: %d' % n_noise_embeddings)
#     # score_2 = metrics.silhouette_score(data_scaled, labels_embeddings)
#     # print("Silhouette Coefficient: %0.3f" % score_2)
#     # print("--------------------------------------------------------")

#     return


# # word2vec
# # eps=3.75,sampl=4

# # for e in epsilons:
# #     for s in min_samples:
# #         for ex in ['technical_debt']:
# #             try:
# #                 score_clustering(data_per_exercise[ex],e,s)
# #             except ValueError:
# #                 print('wrong params', e,s)

# #         0.55,2 -> 0.215
# #         0.95,2
# #         1.1,2 -> 0.279
# #         1.2,3


# for s in [2, 3, 4, 5, 6, 7, 8]:
#     try:
#         score_clustering(data_per_exercise["technical_debt"], 1.5, s)
#     except:
#         print("wrong")


# file1 = 'syntax_error'
# file2 = 'syntax_progress'

# d1 = get_data_clustered_per_exercise('data-processed/data_per_exercise_'+file1, 1.5, 5)
# with open('data-processed/data_clustered_'+file1, 'wb') as pkl_file:
#     pkl.dump(d1, pkl_file)

# d2 = get_data_clustered_per_exercise('data-processed/data_per_exercise_'+file2, 1.5, 5)
# with open('data-processed/data_clustered_'+file2, 'wb') as pkl_file:
#     pkl.dump(d2, pkl_file)

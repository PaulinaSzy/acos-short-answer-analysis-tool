import pickle as pkl
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
from time import time
from sklearn import metrics


def get_data_for_scoring(data_per_exercise):
    exercises = {}

    for exercise in data_per_exercise:
        similarity_matrix = data_per_exercise[exercise]["matrix"]
        similarity_matrix = np.clip(similarity_matrix, -1, 1)
        complete_similarity_matrix = similarity_matrix + similarity_matrix.T
        np.fill_diagonal(complete_similarity_matrix, 1)
        distance_matrix = np.arccos(complete_similarity_matrix) / np.pi
        X_embedded = TSNE(n_components=2, metric="precomputed").fit_transform(
            distance_matrix
        )
        exercises[exercise] = (distance_matrix, X_embedded)

    return exercises


def get_random_search_results(distance_matrix, x_embedded):
    def infinite_datasource():
        idxs = list(range(x_embedded.shape[0]))
        yield (idxs, idxs)

    def silhouette_score(estimator, X):
        clusters = estimator.fit_predict(X)
        score = metrics.silhouette_score(
            distance_matrix, clusters, metric="precomputed"
        )
        return score

    n_iter_search = 1000
    distributions = {"eps": stats.uniform(0.1, 10), "min_samples": stats.randint(2, 30)}
    ca = DBSCAN()
    random_search = RandomizedSearchCV(
        ca,
        param_distributions=distributions,
        n_iter=n_iter_search,
        scoring=silhouette_score,
        cv=infinite_datasource(),
    )
    start = time()

    random_search.fit(x_embedded)
    params_fit = random_search.cv_results_["params"]
    scores_fit = random_search.cv_results_["mean_test_score"]
    ranks_fit = random_search.cv_results_["rank_test_score"]

    for (idx, (params, score, rank)) in enumerate(
        zip(params_fit, scores_fit, ranks_fit)
    ):
        if rank == 1:
            best_params = params

    return best_params


def save_best_params(infile, outfile):
    data_per_ex = pkl.load(open(infile, "rb"))
    data_for_scoring = get_data_for_scoring(data_per_ex)
    best_params_per_ex = {}
    for exercise in data_for_scoring:
        print(exercise)
        best_params = get_random_search_results(
            data_for_scoring[exercise][0], data_for_scoring[exercise][1]
        )
        best_params_per_ex[exercise] = best_params
    with open(outfile, "wb") as pkl_file:
        pkl.dump(best_params_per_ex, pkl_file)


data_paths = [
    (
        "data-processed/data_per_exercise_allsyntax_error",
        "data-processed/bestparams_allsyntax_error",
    ),
    (
        "data-processed/data_per_exercise_allsyntax_progress",
        "data-processed/bestparams_allsyntax_progress",
    ),
    (
        "data-processed/data_per_exercise_constituency_error",
        "data-processed/bestparams_constituency_error",
    ),
    (
        "data-processed/data_per_exercise_constituency_progress",
        "data-processed/bestparams_constituency_progress",
    ),
    (
        "data-processed/data_per_exercise_dependency_error",
        "data-processed/bestparams_dependency_error",
    ),
    (
        "data-processed/data_per_exercise_dependency_progress",
        "data-processed/bestparams_dependency_progress",
    ),
    (
        "data-processed/data_per_exercise_sentence_error",
        "data-processed/bestparams_sentence_error",
    ),
    (
        "data-processed/data_per_exercise_sentence_progress",
        "data-processed/bestparams_sentence_progress",
    ),
    (
        "data-processed/data_per_exercise_style_embedding_error",
        "data-processed/bestparams_style_embedding_error",
    ),
    (
        "data-processed/data_per_exercise_style_embedding_progress",
        "data-processed/bestparams_style_embedding_progress",
    ),
    (
        "data-processed/data_per_exercise_style_wordnet_error",
        "data-processed/bestparams_style_wordnet_error",
    ),
    (
        "data-processed/data_per_exercise_style_wordnet_progress",
        "data-processed/bestparams_style_wordnet_progress",
    ),
]


def run():
    for in_path, out_path in data_paths:
        save_best_params(in_path, out_path)


# params = pkl.load(open("bestparams_allsyntax_error", "rb"))
# params

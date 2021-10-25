import pandas as pd

import shared_functions
import shared_variables

data_clustered_semantic_embedding_error = pd.read_pickle(
    "data-processed/data_clustered_style_embedding_error"
)
data_clustered_semantic_embedding_progress = pd.read_pickle(
    "data-processed/data_clustered_style_embedding_progress"
)
data_clustered_semantic_wordnet_error = pd.read_pickle(
    "data-processed/data_clustered_style_wordnet_error"
)
data_clustered_semantic_wordnet_progress = pd.read_pickle(
    "data-processed/data_clustered_style_wordnet_progress"
)

(
    years,
    year_labels,
    periods,
    period_labels,
    users,
    user_labels,
    problems,
    problem_labels,
) = shared_functions.get_year_period_user_problem(
    shared_variables.path_to_all_data, "style"
)

# color_seq = shared_variables.color_sequence
# max_clust_error = shared_functions.get_max_cluster_size(data_clustered_semantic_error)
# colorsIdx_error = {}

# for label in range(max_clust_error + 1):
#     if label < len(color_seq):
#         colorsIdx_error[f"{label}"] = color_seq[label]
#     else:
#         colorsIdx_error[f"{label}"] = "#%06X" % randint(0, 0xFFFFFF)

# colorsIdx_error["-1"] = "rgb(166, 166, 166)"


# max_clust_progress = shared_functions.get_max_cluster_size(
#     data_clustered_semantic_progress
# )
# colorsIdx_progress = {}

# for label in range(max_clust_progress + 1):
#     if label < len(color_seq):
#         colorsIdx_progress[f"{label}"] = color_seq[label]
#     else:
#         colorsIdx_progress[f"{label}"] = "#%06X" % randint(0, 0xFFFFFF)

# colorsIdx_progress["-1"] = "rgb(166, 166, 166)"

datafr = shared_variables.datafr.loc[shared_variables.datafr["problemType"] == "style"]

colorsIdx_error = shared_functions.create_color_idxs(
    data_clustered_semantic_embedding_error
)

scatterplots_semantic_error, alldata_semantic_error = shared_functions.create_plots(
    "semantic_error",
    datafr,
    data_clustered_semantic_embedding_error,
    remove_duplicates=True,
    cluster_fig_size=1500,
)

(
    scatterplots_semantic_progress,
    alldata_semantic_progress,
) = shared_functions.create_plots(
    "semantic_progress",
    datafr,
    data_clustered_semantic_embedding_progress,
    remove_duplicates=False,
    stats_table=True,
    stats_graph=True,
    cluster_fig_size=1000,
)


content_semantic_error = shared_functions.create_page_content(
    "semantic_error",
    scatterplots_semantic_error,
    periods,
    period_labels,
    years,
    year_labels,
    users,
    user_labels,
)

content_semantic_progress = shared_functions.create_page_content(
    "semantic_progress",
    scatterplots_semantic_progress,
    periods,
    period_labels,
    years,
    year_labels,
    users,
    user_labels,
)

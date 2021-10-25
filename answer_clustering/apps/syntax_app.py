import pandas as pd

import shared_functions
import shared_variables

# data_clustered_syntax_error = pd.read_pickle(
#     "data-processed/data_clustered_syntax_error"
# )

data_clustered_syntax_all_error = pd.read_pickle(
    "data-processed/data_clustered_allsyntax_error"
)
data_clustered_syntax_all_progress = pd.read_pickle(
    "data-processed/data_clustered_allsyntax_progress"
)


data_clustered_syntax_sentence_error = pd.read_pickle(
    "data-processed/data_clustered_sentence_error"
)
data_clustered_syntax_sentence_progress = pd.read_pickle(
    "data-processed/data_clustered_sentence_progress"
)


data_clustered_syntax_dependency_error = pd.read_pickle(
    "data-processed/data_clustered_dependency_error"
)
data_clustered_syntax_dependency_progress = pd.read_pickle(
    "data-processed/data_clustered_dependency_progress"
)


data_clustered_syntax_constituency_error = pd.read_pickle(
    "data-processed/data_clustered_constituency_error"
)
data_clustered_syntax_constituency_progress = pd.read_pickle(
    "data-processed/data_clustered_constituency_progress"
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
    shared_variables.path_to_all_data, "order"
)

# color_seq = shared_variables.color_sequence
# max_clust_error = shared_functions.get_max_cluster_size(data_clustered_syntax_error)
# colorsIdx_error = {}

# for label in range(max_clust_error + 1):
#     if label < len(color_seq):
#         colorsIdx_error[f"{label}"] = color_seq[label]
#     else:
#         colorsIdx_error[f"{label}"] = "#%06X" % randint(0, 0xFFFFFF)

# colorsIdx_error["-1"] = "rgb(166, 166, 166)"

# max_clust_progress = shared_functions.get_max_cluster_size(
#     data_clustered_syntax_progress
# )

# colorsIdx_progress = {}

# for label in range(max_clust_progress + 1):
#     if label < len(color_seq):
#         colorsIdx_progress[f"{label}"] = color_seq[label]
#     else:
#         colorsIdx_progress[f"{label}"] = "#%06X" % randint(0, 0xFFFFFF)

# colorsIdx_progress["-1"] = "rgb(166, 166, 166)"

datafr = shared_variables.datafr.loc[shared_variables.datafr["problemType"] == "order"]

colorsIdx_error = shared_functions.create_color_idxs(
    data_clustered_syntax_sentence_error
)

scatterplots_syntax_error, alldata_syntax_error = shared_functions.create_plots(
    "syntax_error",
    datafr,
    data_clustered_syntax_sentence_error,
    remove_duplicates=True,
    cluster_fig_size=1500,
)

scatterplots_syntax_progress, alldata_syntax_progress = shared_functions.create_plots(
    "syntax_progress",
    datafr,
    data_clustered_syntax_sentence_progress,
    remove_duplicates=False,
    stats_table=True,
    stats_graph=True,
    cluster_fig_size=1000,
)

content_syntax_error = shared_functions.create_page_content(
    "syntax_error",
    scatterplots_syntax_error,
    periods,
    period_labels,
    years,
    year_labels,
    users,
    user_labels,
)

content_syntax_progress = shared_functions.create_page_content(
    "syntax_progress",
    scatterplots_syntax_progress,
    periods,
    period_labels,
    years,
    year_labels,
    users,
    user_labels,
)

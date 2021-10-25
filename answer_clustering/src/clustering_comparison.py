#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn import metrics
import math
import numpy as np
import pickle as pkl
import statistics


data_clustered_syntax_all_error = pd.read_pickle(
    "data-processed/data_clustered_allsyntax_error"
)
data_per_exercise_syntax_all_progress = pd.read_pickle(
    "data-processed/data_per_exercise_allsyntax_error"
)

# data_clustered_syntax_all_error['synthesis']['label'].tolist()





# data_per_exercise_syntax_all_progress['synthesis']['matrix']





import numpy as np

data_paths = [('allsyntax_error'),
('constituency_error'),
('dependency_error'),
('sentence_error'),
('style_embedding_error'),
('style_wordnet_error')]


def get_silhouette_scores(data_file,cluster_file):
    scores = []
    for ex in data_file:
        similarity_matrix = data_file[ex]['matrix']
        similarity_matrix = np.clip(similarity_matrix, -1, 1)
        complete_similarity_matrix = similarity_matrix + similarity_matrix.T
        np.fill_diagonal(complete_similarity_matrix, 1)
        distance_matrix = np.arccos(complete_similarity_matrix) / np.pi
        clustering_labels = cluster_file[ex]['label'].tolist()
        if np.unique(clustering_labels).size > 1:
            score = metrics.silhouette_score(distance_matrix, clustering_labels, metric='precomputed')
        else:
            score = np.nan
        scores.append(score)
    new_scores = [item for item in scores if not(math.isnan(item)) == True]
    return new_scores

silhouette_scores = {}

for path in data_paths:
    data = pd.read_pickle("data-processed/data_per_exercise_"+path)
    data_clustered = pd.read_pickle("data-processed/data_clustered_"+path)
    res = get_silhouette_scores(data,data_clustered)
    silhouette_scores[path]=res
    
with open('silhouette_scores_per_analysis', 'wb') as pkl_file:
    pkl.dump(silhouette_scores, pkl_file)





def get_avg_silhouette_scores(dict_keys):
    all_scores = pd.read_pickle("silhouette_scores_per_analysis")
    res = []
    for key in dict_keys:
        res.append(statistics.mean(all_scores[key]))
    return res





import plotly.graph_objects as go

syntactic_features=['sentence based', 'constituency based', 'dependency based','all']
syntax_score_keys = ['sentence_error','constituency_error','dependency_error','allsyntax_error']
syntax_y_values = get_avg_silhouette_scores(syntax_score_keys)

semantic_features=['wordnet based', 'sentence embedding based']
semantics_score_keys = ['style_wordnet_error','style_embedding_error']
semantics_y_values = get_avg_silhouette_scores(semantics_score_keys)

# fig1 = go.Figure([go.Bar(x=syntactic_features, y=syntax_y_values,textposition='auto',marker_color='rgb(55, 83, 109)')])
# fig1.show()

# fig2 = go.Figure([go.Bar(x=semantic_features, y=semantics_y_values,textposition='auto',marker_color='rgb(55, 83, 109)')])
# fig2.show()


def plot(x_values,y_values,title_text,y_axis_text):
    y_vals_text = list(map(lambda x: str(round(x,2)), y_values))
    fig = go.Figure([go.Bar(x=x_values, y=y_values,textposition='auto',text=y_vals_text,marker_color='rgb(55, 83, 109)',width=0.5)])
    fig.update_layout(
        title=title_text,
        barmode='group', 
#         width=500,
#         height=500,
        legend=dict(
#             x=0.94,
#             y=1.0,
            # bgcolor='rgba(255, 255, 255, 0)',
            # bordercolor='rgba(255, 255, 255, 0)',
            # bgcolor="white",
        ),
#         bargap=0.1,
        # bargroupgap=0.1
        xaxis_tickfont_size=16,
        yaxis=dict(
            title=y_axis_text,
            titlefont_size=18,
            tickfont_size=16,

        ),
    )
    fig.show()





plot(syntactic_features,syntax_y_values,'Average silhouette scores of clusters based on syntactic features', 'Average silhouette score')





plot(semantic_features,semantics_y_values,'Average silhouette scores of clusters based on semantic features', 'Average silhouette score')





from plotly import tools

def plot(data, titel, y_axis, data1, bar1, data2, bar2, y_min=0, y_max=100):
    fig = go.Figure(data=[
        go.Bar(name=bar1, x=data.index.values, y=data[data1],text=round(data[data1], 2),textposition='auto',marker_color='rgb(55, 83, 109)'),
        go.Bar(name=bar2, x=data.index.values, y=data[data2],text=round(data[data2], 2),textposition='auto',marker_color='rgb(26, 118, 255)')
    ])
    # fig.update_yaxes(range=[y_min,y_max])

    fig.update_layout(
        title=titel,
        barmode='group', 
        width=500,
        height=500,
        legend=dict(
            x=0.94,
            y=1.0,
            # bgcolor='rgba(255, 255, 255, 0)',
            # bordercolor='rgba(255, 255, 255, 0)',
            # bgcolor="white",
        ),
        bargap=0.1,
        # bargroupgap=0.1
        xaxis_tickfont_size=14,
        yaxis=dict(
            title=y_axis,
            titlefont_size=16,
            tickfont_size=14,

        ),
    )
    fig.show()





list(map(lambda x: str(round(x,2)), [1.22,2.333,2.333])








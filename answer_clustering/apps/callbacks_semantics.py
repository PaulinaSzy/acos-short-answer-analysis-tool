from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

import shared_variables
import shared_functions
from apps import semantics_app

from app import app


color_seq = shared_variables.color_sequence


@app.callback(
    [
        Output("semantic_error-period-year-selection", "value"),
        # Output("syntax_error-student-selection", "value"),
    ],
    Input("semantic_error-feature-selection", "value"),
    [
        State("semantic_error-period-year-selection", "value"),
        # State("syntax_error-student-selection", "value"),
    ],
)
def update_feature_selection(feature_selection, period_year_sel):

    if feature_selection == "embedding":
        (
            semantics_app.alldata_semantic_error,
            semantics_app.colorIdx_error,
        ) = shared_functions.create_all_data(
            semantics_app.datafr,
            semantics_app.data_clustered_semantic_embedding_error,
            remove_duplicates=True,
        )

    if feature_selection == "wordnet":
        (
            semantics_app.alldata_semantic_error,
            semantics_app.colorIdx_error,
        ) = shared_functions.create_all_data(
            semantics_app.datafr,
            semantics_app.data_clustered_semantic_wordnet_error,
            remove_duplicates=True,
        )

    return [period_year_sel]


@app.callback(
    Output("semantic_error-student-selection", "value"),
    [Input("semantic_error-select-all", "value")],
    [
        State("semantic_error-student-selection", "options"),
        State("semantic_error-student-selection", "value"),
    ],
)
def update_student_selection_error(selected, options, values):
    if len(selected) > 0 and selected[0] == 1:
        return [i["value"] for i in options]
    else:
        return values


@app.callback(
    Output("semantic_progress-student-selection", "value"),
    [Input("semantic_progress-select-all", "value")],
    [
        State("semantic_progress-student-selection", "options"),
        State("semantic_progress-student-selection", "value"),
    ],
)
def update_student_selection_progress(selected, options, values):
    if len(selected) > 0 and selected[0] == 1:
        return [i["value"] for i in options]
    else:
        return values


# -------------------------------------- Main graphs --------------------------------------


@app.callback(
    [
        Output(f"semantic_error-graph-{semantics_app.problems[i]}", "figure")
        for i in range(len(semantics_app.problems))
    ],
    [
        Input("semantic_error-period-year-selection", "value"),
        Input("semantic_error-student-selection", "value"),
    ],
)
def update_main_graph_semantic_error(period_year, student):
    all_figs = []
    for problem_name in semantics_app.problems:
        data_problem = semantics_app.alldata_semantic_error[problem_name]
        data_filtered_problem = data_problem.loc[
            (data_problem["period"].isin(period_year))
            & (data_problem["year"].isin(period_year))
            & (data_problem["uid"].isin(student))
        ]
        if len(data_filtered_problem) > 0:
            cols = data_filtered_problem["label"].map(semantics_app.colorsIdx_error)
            sizs = data_filtered_problem["correct"].map(shared_variables.sizesIdx)
            stys = data_filtered_problem["correct"].map(shared_variables.stylesIdx)
            fig = go.FigureWidget(
                data=go.Scatter(
                    x=data_filtered_problem["x_component"],
                    y=data_filtered_problem["y_component"],
                    mode="markers",
                    marker=dict(size=sizs, color=cols, symbol=stys),
                    hovertext=shared_functions.create_hovertext(data_filtered_problem),
                    hovertemplate="%{hovertext}<br><extra></extra>",
                    hoverlabel_bgcolor="rgb(255, 255, 255)",
                )
            )
            all_figs.append(fig)
        else:
            fig = go.Figure()
            fig.update_layout(
                # width=200,
                # height=200,
                xaxis={"visible": False},
                yaxis={"visible": False},
                plot_bgcolor="rgba(0,0,0,0)",
                annotations=[
                    {
                        "text": "No data found",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 28},
                    }
                ],
            )
            all_figs.append(fig)

    return all_figs


# @app.callback(
#     [
#         Output(f"semantic_progress-graph-{semantics_app.problems[i]}", "figure")
#         for i in range(len(semantics_app.problems))
#     ],
#     [
#         Input("semantic_progress-period-year-selection", "value"),
#         Input("semantic_progress-student-selection", "value"),
#     ],
# )
# def update_main_graph_semantic_progress(period_year, student):
#     all_figs = []
#     for problem_name in semantics_app.problems:
#         data_problem = semantics_app.alldata_semantic_progress[problem_name]
#         data_filtered_problem = data_problem.loc[
#             (data_problem["period"].isin(period_year))
#             & (data_problem["year"].isin(period_year))
#             & (data_problem["uid"] == student)
#         ]
#         if len(data_filtered_problem) > 0:
#             cols = data_filtered_problem["label"].map(semantics_app.colorsIdx_progress)
#             sizs = data_filtered_problem["correct"].map(shared_variables.sizesIdx)
#             stys = data_filtered_problem["correct"].map(shared_variables.stylesIdx)
#             fig = go.FigureWidget(
#                 data=go.Scatter(
#                     x=data_filtered_problem["x_component"],
#                     y=data_filtered_problem["y_component"],
#                     mode="markers",
#                     marker=dict(size=sizs, color=cols, symbol=stys),
#                     hovertext=shared_functions.create_hovertext(data_filtered_problem),
#                     hovertemplate="%{hovertext}<br><extra></extra>",
#                     hoverlabel_bgcolor="rgb(255, 255, 255)",
#                 )
#             )
#             all_figs.append(fig)
#         else:
#             fig = go.Figure()
#             fig.update_layout(
#                 # width=200,
#                 # height=200,
#                 xaxis={"visible": False},
#                 yaxis={"visible": False},
#                 plot_bgcolor="rgba(0,0,0,0)",
#                 annotations=[
#                     {
#                         "text": "No data found",
#                         "xref": "paper",
#                         "yref": "paper",
#                         "showarrow": False,
#                         "font": {"size": 28},
#                     }
#                 ],
#             )
#             all_figs.append(fig)

#     return all_figs


# -------------------------------------- Concordance --------------------------------------


@app.callback(
    [
        Output(
            f"semantic_error-concordance-output-{semantics_app.problems[i]}", "children"
        )
        for i in range(len(semantics_app.problems))
    ],
    [
        inp
        for i in range(len(semantics_app.problems))
        for inp in (
            Input(
                f"semantic_error-concordance-input-{semantics_app.problems[i]}", "value"
            ),
            Input(
                f"semantic_error-concordance-width-input-{semantics_app.problems[i]}",
                "value",
            ),
            Input(
                f"semantic_error-concordance-lines-input-{semantics_app.problems[i]}",
                "value",
            ),
        )
    ],
)
def show_concordance(*args):
    conc_results = []
    for idx, problem in enumerate(semantics_app.problems):
        conc_results.append(
            shared_functions.get_concordance(
                semantics_app.datafr,
                problem,
                args[3 * idx],
                args[3 * idx + 1],
                args[3 * idx + 2],
            )
        )
    return conc_results


# -------------------------------------- Refresh selected data --------------------------------------


@app.callback(
    [
        Output(f"semantic_error-graph-{semantics_app.problems[i]}", "selectedData")
        for i in range(len(semantics_app.problems))
    ],
    [
        Input("semantic_error-period-year-selection", "value"),
        Input("semantic_error-student-selection", "value"),
    ],
)
def refresh_selected_data_on_filter_semantic_error(period_year, student):
    return [None for _ in semantics_app.problems]


# @app.callback(
#     [
#         Output(f"semantic_progress-graph-{semantics_app.problems[i]}", "selectedData")
#         for i in range(len(semantics_app.problems))
#     ],
#     [
#         Input("semantic_progress-period-year-selection", "value"),
#         Input("semantic_progress-student-selection", "value"),
#     ],
# )
# def refresh_selected_data_on_filter_semantic_progress(period_year, student):
#     return [None for _ in semantics_app.problems]


# -------------------------------------- Change table --------------------------------------

outputs = [
    Output(f"semantic_error-table-{semantics_app.problems[i]}", "data")
    for i in range(len(semantics_app.problems))
]
inputs = [
    Input("semantic_error-period-year-selection", "value"),
    Input("semantic_error-student-selection", "value"),
] + [
    Input(f"semantic_error-graph-{semantics_app.problems[i]}", "selectedData")
    for i in range(len(semantics_app.problems))
]


@app.callback(outputs, inputs)
def change_table_semantic_error(*args):
    outs = []
    for index, problem_name in enumerate(semantics_app.problems):
        selected_data = args[index + 2]
        problem_data = semantics_app.alldata_semantic_error[problem_name]
        filtered_data = problem_data.loc[
            (problem_data["period"].isin(args[0]))
            & (problem_data["year"].isin(args[0]))
            & (problem_data["uid"].isin(args[1]))
        ]
        if selected_data:
            point_inds = [p["pointIndex"] for p in selected_data["points"]]
            out_data = (
                filtered_data[
                    [
                        "problemName",
                        "uid",
                        "date",
                        "period",
                        "year",
                        "answer",
                        "correct",
                    ]
                ]
                .iloc[point_inds]
                .to_dict("records")
            )
        else:
            out_data = filtered_data[
                [
                    "problemName",
                    "uid",
                    "date",
                    "period",
                    "year",
                    "answer",
                    "correct",
                ]
            ].to_dict("records")
        outs.append(out_data)
    return outs


outputs = [
    Output(f"semantic_progress-table-{semantics_app.problems[i]}", "data")
    for i in range(len(semantics_app.problems))
]
inputs = [
    Input("semantic_progress-period-year-selection", "value"),
    Input("semantic_progress-student-selection", "value"),
]
# + [
#     Input(f"semantic_progress-graph-{semantics_app.problems[i]}", "selectedData")
#     for i in range(len(semantics_app.problems))
# ]


@app.callback(outputs, inputs)
def change_table_semantic_progress(*args):
    outs = []
    for index, problem_name in enumerate(semantics_app.problems):
        selected_data = None
        problem_data = semantics_app.alldata_semantic_progress[problem_name]
        filtered_data = problem_data.loc[
            (problem_data["period"].isin(args[0]))
            & (problem_data["year"].isin(args[0]))
            & (problem_data["uid"] == args[1])
        ]
        if selected_data:
            point_inds = [p["pointIndex"] for p in selected_data["points"]]
            out_data = (
                filtered_data[
                    [
                        "problemName",
                        "uid",
                        "date",
                        "period",
                        "year",
                        "answer",
                        "correct",
                    ]
                ]
                .iloc[point_inds]
                .to_dict("records")
            )
        else:
            out_data = filtered_data[
                [
                    "problemName",
                    "uid",
                    "date",
                    "period",
                    "year",
                    "answer",
                    "correct",
                ]
            ].to_dict("records")
        outs.append(out_data)
    return outs


# -------------------------------------- Change progress table --------------------------------------


output_tables = [
    Output(f"semantic_progress-stats-table-{semantics_app.problems[i]}", "data")
    for i in range(len(semantics_app.problems))
]


@app.callback(
    output_tables,
    Input("semantic_progress-period-year-selection", "value"),
    Input("semantic_progress-student-selection", "value"),
)
def generate_student_stats(period_year, student):
    outs = []
    for problem_name in semantics_app.problems:
        data_problem = semantics_app.alldata_semantic_progress[problem_name]
        data_filtered_problem = data_problem.loc[
            (data_problem["period"].isin(period_year))
            & (data_problem["year"].isin(period_year))
            & (data_problem["uid"] == student)
        ]
        outs.append(
            shared_functions.get_statistics_per_student(data_filtered_problem).to_dict(
                "records"
            )
        )
    return outs


# -------------------------------------- Change progress graph --------------------------------------


output_graphs = [
    Output(f"semantic_progress-stats-graph-{semantics_app.problems[i]}", "figure")
    for i in range(len(semantics_app.problems))
]


@app.callback(
    output_graphs,
    Input("semantic_progress-period-year-selection", "value"),
    Input("semantic_progress-student-selection", "value"),
)
def update_student_progress_graph(period_year, student):
    out_figs = []
    for problem_name in semantics_app.problems:
        data_problem = semantics_app.alldata_semantic_progress[problem_name]
        data_filtered_problem = data_problem.loc[
            (data_problem["period"].isin(period_year))
            & (data_problem["year"].isin(period_year))
            & (data_problem["uid"] == student)
        ]
        if len(data_filtered_problem) > 0:
            date_sorted, score_sorted = shared_functions.sort_by_date(
                data_filtered_problem["date"], data_filtered_problem["score"]
            )

            fig = go.Figure()
            with fig.batch_update():
                trace = go.Scatter(
                    x=date_sorted,
                    y=score_sorted,
                    mode="lines+markers",
                    name=problem_name,
                )
                fig.add_trace(trace)
            fig.update_layout(
                title="Progress of student {} in exercise {}".format(
                    student, problem_name
                ),
                showlegend=True,
                legend=dict(x=0, y=1.0),
                margin=dict(l=40, r=0, t=40, b=30),
                yaxis_title="Score (%)",
            )
            out_figs.append(fig)
        else:
            fig = go.Figure()
            fig.update_layout(
                xaxis={"visible": False},
                yaxis={"visible": False},
                plot_bgcolor="rgba(0,0,0,0)",
            )
            out_figs.append(fig)

    return out_figs

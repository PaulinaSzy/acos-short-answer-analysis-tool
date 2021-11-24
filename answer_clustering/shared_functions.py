import datetime
import pandas as pd
import difflib
import textwrap
import nltk
from dash import dash_table
import plotly.graph_objects as go
from nltk.text import ConcordanceIndex
from dash import dcc
from dash import html
import layouts
import shared_variables
from random import randint
import src.functions as fff


def concordance(ci, word, width=100, lines=25):
    """
    Rewrite of nltk.text.ConcordanceIndex.print_concordance that returns results
    instead of printing them.

    See:
    http://www.nltk.org/api/nltk.html#nltk.text.ConcordanceIndex.print_concordance
    """
    half_width = (width - len(word) - 2) // 2
    context = width // 4  # approx number of words of context

    results = []
    offsets = ci.offsets(word)
    if offsets:
        lines = min(lines, len(offsets))
        for i in offsets:
            if lines <= 0:
                break
            left = " " * half_width + " ".join(ci._tokens[i - context : i])
            right = " ".join(ci._tokens[i + 1 : i + context])
            left = left[-half_width:]
            right = right[:half_width]
            results.append("%s %s %s \n" % (left, ci._tokens[i], right))
            lines -= 1

    return results


def get_concordance(d, problem, word, display_width, display_lines):
    display_width = int(display_width) if display_width != "" else 100
    display_lines = int(display_lines) if display_lines != "" else 25
    answers = d.loc[d["problemName"] == problem]["answer"].tolist()
    answer_text = " ".join(answers)
    tokens = nltk.word_tokenize(answer_text)
    ci = ConcordanceIndex(tokens)
    return concordance(ci, word, display_width, display_lines)


def get_max_cluster_size(data):
    max_label = 0
    for ex in data:
        max_label_task = data[ex]["label"].max()
        if max_label_task > max_label:
            max_label = max_label_task
    return max_label


def get_year_period_user_problem(path_to_all_data, analysis_type):
    datafr = pd.read_pickle(path_to_all_data)
    datafr = datafr.loc[datafr["problemType"] == analysis_type]

    periods = datafr["period"].unique().tolist()
    periods_labels = [{"label": "Period " + str(v), "value": v} for v in periods]

    years = datafr["year"].unique().tolist()
    years_labels = [{"label": "Year " + str(v), "value": v} for v in years]

    users = datafr["uid"].unique().tolist()
    users_labels = [{"label": "StudentId " + str(v), "value": v} for v in users]

    problems = datafr["problemName"].unique().tolist()
    problems_labels = [{"label": str(v), "value": v} for v in problems]

    return (
        periods,
        periods_labels,
        years,
        years_labels,
        users,
        users_labels,
        problems,
        problems_labels,
    )


def get_str_diff(wrong, correct):
    printout = ""
    matches = difflib.SequenceMatcher(None, wrong, correct).get_matching_blocks()
    for match in matches:
        printout = printout + " " + wrong[match.a : match.a + match.size]
    return printout


def mark_str_diff(wrong, correct):
    printout = ""
    wrong = "\n".join(textwrap.wrap(wrong, width=60, break_long_words=False))
    opcodes = difflib.SequenceMatcher(lambda x: x in "\n", wrong, correct).get_opcodes()
    for tup in opcodes:
        chunk = wrong[tup[1] : tup[2]]
        if tup[0] == "equal":
            printout = printout + chunk
        elif tup[0] == "insert":
            printout = printout + '<span style="color: #AA1100">' + "_" + "</span>"
        else:
            printout = (
                printout
                + '<span style="color: #AA1100; font-style: italic; text-decoration: underline;'
                + 'text-decoration-style: wavy; text-decoration-color: red;">'
                + chunk
                + "</span>"
            )

    return printout.replace("\n", "<br>")


def get_list_of_str_diffs(wrongs, corrects):
    res = []
    for w, c in zip(wrongs, corrects):
        res.append(mark_str_diff(w, c))
    return res


def create_hovertext(dataframe):
    return (
        "<b>Student's answer</b>: "
        + dataframe["difference"]
        + "<br><b>Correct</b>: "
        + dataframe["correct"].apply(lambda x: str(x))
        + "<br><b>StudentID</b>: "
        + dataframe["uid"].apply(lambda x: str(x))
        + "<br><b>Period</b>: "
        + dataframe["period"]
        + " <b>Year</b>: "
        + dataframe["year"].astype(str)
    )


def create_color_idxs(cluster_data):
    color_seq = shared_variables.color_sequence
    max_clust_error = get_max_cluster_size(cluster_data)

    colors_indexes = {}

    for label in range(max_clust_error + 1):
        if label < len(color_seq):
            colors_indexes[f"{label}"] = color_seq[label]
        else:
            colors_indexes[f"{label}"] = "#%06X" % randint(0, 0xFFFFFF)

    colors_indexes["-1"] = "rgb(166, 166, 166)"

    return colors_indexes


def create_all_data(
    data, cluster_data, remove_duplicates=False, remove_incorrectly_spelled=True
):
    all_data = {}
    for ex in cluster_data:
        clusters = cluster_data[ex]
        data_merged = pd.merge(data, clusters, how="inner", on="answer")

        if remove_incorrectly_spelled:
            data_merged = data_merged.loc[
                data_merged["answer"].apply(fff.is_answer_correctly_spelled)
            ]

        if remove_duplicates:
            data_merged.drop_duplicates(subset=["answer"], keep="first", inplace=True)

        data_merged["label"] = data_merged["label"].astype(str)
        data_merged["difference"] = get_list_of_str_diffs(
            data_merged["answer"], data_merged["closest"]
        )
        all_data[ex] = data_merged

    color_idxs = create_color_idxs(cluster_data)
    return all_data, color_idxs


def create_plots(
    question_analysis_type,
    data,
    cluster_data,
    remove_duplicates=False,
    stats_table=False,
    stats_graph=False,
    cluster_fig_size=1000,
):
    colors_indexes = create_color_idxs(cluster_data)

    all_elements = []
    all_data = {}
    for ex in cluster_data:
        clusters = cluster_data[ex]
        data_merged = pd.merge(data, clusters, how="inner", on="answer")
        if remove_duplicates:
            data_merged.drop_duplicates(subset=["answer"], keep="first", inplace=True)
        data_merged["label"] = data_merged["label"].astype(str)
        data_merged["difference"] = get_list_of_str_diffs(
            data_merged["answer"], data_merged["closest"]
        )

        graph = None

        if "error" in question_analysis_type:
            data_merged = data_merged.loc[
                data_merged["answer"].apply(fff.is_answer_correctly_spelled)
            ]
            cols = data_merged["label"].map(colors_indexes)
            sizs = data_merged["correct"].map(shared_variables.sizesIdx)
            stys = data_merged["correct"].map(shared_variables.stylesIdx)
            fig = go.FigureWidget(
                data=go.Scatter(
                    x=data_merged["x_component"],
                    y=data_merged["y_component"],
                    mode="markers",
                    marker=dict(size=sizs, color=cols, symbol=stys),
                    hovertext=create_hovertext(data_merged),
                    hovertemplate="%{hovertext}<br><extra></extra>",
                    hoverlabel_bgcolor="rgb(255, 255, 255)",
                )
            )
            fig.update_layout(
                width=cluster_fig_size,
                height=cluster_fig_size,
                hoverlabel_font_size=20,
                hoverdistance=1,
                dragmode="select",
            )
            graph = dcc.Graph(
                id=question_analysis_type + "-graph-" + ex,
                figure=fig,
                style={"display": "inline"},
            )

        table_cols = [
            "problemName",
            "uid",
            "date",
            "period",
            "year",
            "answer",
            "correct",
        ]
        table = (
            dash_table.DataTable(
                id=question_analysis_type + "-table-" + ex,
                columns=[{"name": i, "id": i} for i in table_cols],
                data=data_merged[table_cols].to_dict("records"),
                export_format="csv",
                sort_action="native",
                page_size=20,
                style_cell=dict(textAlign="left"),
                style_header=dict(backgroundColor="rgb(234, 234, 234)"),
                style_table={"overflowX": "auto", "max-width": "auto"},
            ),
        )

        table_div = html.Div(table, style={"margin": "10px"})
        elems = [graph, table_div]

        if stats_table:
            t = html.Div(
                dash_table.DataTable(
                    id=question_analysis_type + "-stats-table-" + ex,
                    columns=[
                        {"name": i, "id": j}
                        for (i, j) in [
                            ("Nr of attempts", "Attempt"),
                            ("Solved", "IsSolved"),
                            ("Nr of trials", "NumberOfTrials"),
                            ("Time spent", "TimeSpent"),
                        ]
                    ],
                    export_format="csv",
                    sort_action="native",
                    page_size=20,
                    style_cell=dict(textAlign="left"),
                    style_header=dict(backgroundColor="rgb(234, 234, 234)"),
                ),
                style={"display": "inline-block", "margin": "10px", "float": "left"},
            )
            elems.append(t)

        if stats_graph:
            g = dcc.Graph(
                id=question_analysis_type + "-stats-graph-" + ex,
                style={"height": 300, "display": "inline-block", "margin": "10px"},
                figure={"layout": {"yaxis": {"title": "Score (%)"}}},
            )
            elems.append(g)

        if "error" in question_analysis_type:
            conc_input = html.Div(
                className="concordance",
                children=[
                    html.H4("Concordance view: "),
                    dcc.Input(
                        className="concordance-input",
                        id=question_analysis_type + "-concordance-input-" + ex,
                        placeholder="Enter a word:",
                        type="text",
                        value="",
                    ),
                    dcc.Input(
                        className="concordance-width-input",
                        id=question_analysis_type + "-concordance-width-input-" + ex,
                        placeholder="Enter width:",
                        type="text",
                        value="",
                    ),
                    dcc.Input(
                        className="concordance-lines-input",
                        id=question_analysis_type + "-concordance-lines-input-" + ex,
                        placeholder="Enter nr of lines:",
                        type="text",
                        value="",
                    ),
                ],
                style={"display": "inline"},
            )

            conc_output = html.Pre(
                id=question_analysis_type + "-concordance-output-" + ex,
            )

            elems.append(conc_input)
            elems.append(conc_output)

        all_elements.append(
            html.Button(className="collapsible", children=[html.H3("Exercise: " + ex)])
        )
        all_elements.append(
            html.Div(
                className="collapsible-content",
                children=elems,
                style={"display": "block"},
            )
        )
        all_data[ex] = data_merged

    all_elements.append(html.Span(id="graphs-loaded-flag", style={"display": "none"}))
    return all_elements, all_data


def create_page_content(
    question_analysis_type,
    scatterplots,
    periods,
    period_labels,
    years,
    year_labels,
    users,
    user_labels,
):

    feature_selection_options = []
    if question_analysis_type == "syntax_error":
        question_type = "Order type exercises"
        toolbar_title = "Error analysis of wrong answers"
        description = (
            "The analysis helps to discover various student's errors.\n"
            + "Duplicate answers and answers with spelling errors were filtered out from the data."
        )
        feature_selection_options = [
            {"label": "sentence based", "value": "sentence"},
            {"label": "dependency based", "value": "dependency"},
            {"label": "constituency based", "value": "constituency"},
            {"label": "all", "value": "all"},
        ]
    if question_analysis_type == "syntax_progress":
        question_type = "Order type exercises"
        toolbar_title = "Individual student's progress analysis"
        description = "The analysis helps to track student's progress in solving the exercises. Data contains all entries (including duplicates)."

    if question_analysis_type == "semantic_error":
        question_type = "Style type exercises"
        toolbar_title = "Error analysis of wrong answers"
        description = (
            "The analysis helps to discover various student's errors.\n"
            + "Duplicate answers and answers with spelling errors were filtered out from the data."
        )
        feature_selection_options = [
            {"label": "word embeddings (word2vec)", "value": "embedding"},
            {"label": "wordnet based", "value": "wordnet"},
        ]
    if question_analysis_type == "semantic_progress":
        question_type = "Style type exercises"
        toolbar_title = "Individual student's progress analysis"
        description = "The analysis helps to track student's progress in solving the exercises. Data contains all entries (including duplicates)."

    check_all = None
    if "error" in question_analysis_type:
        stud_sel = dcc.Dropdown(
            id=question_analysis_type + "-student-selection",
            options=user_labels,
            value=users,
            multi=True,
        )
        check_all = dcc.Checklist(
            id=question_analysis_type + "-select-all",
            options=[{"label": "Select all students", "value": 1}],
            value=[],
        )

        feature_selection = dcc.RadioItems(
            id=question_analysis_type + "-feature-selection",
            options=feature_selection_options,
            value=feature_selection_options[0]["value"],
        )

        toolbar_options = [
            html.H3("Select feature selection method for analysis: "),
            feature_selection,
            html.H3("Select year & period: "),
            dcc.Checklist(
                id=question_analysis_type + "-period-year-selection",
                options=period_labels + year_labels,
                value=periods + years,
            ),
            html.H3("Select student: "),
            stud_sel,
            check_all,
        ]

    else:
        stud_sel = dcc.Dropdown(
            id=question_analysis_type + "-student-selection",
            options=user_labels,
            multi=False,
        )

        toolbar_options = [
            html.H3("Select year & period: "),
            dcc.Checklist(
                id=question_analysis_type + "-period-year-selection",
                options=period_labels + year_labels,
                value=periods + years,
            ),
            html.H3("Select student: "),
            stud_sel,
            check_all,
        ]

    toolbar = html.Div(
        className="toolbar",
        children=[
            html.Div(
                className="description",
                children=[
                    html.H1(question_type),
                    html.H2(toolbar_title),
                    html.P(description),
                ],
            ),
            html.Br(),
            html.Div(toolbar_options),
        ],
    )
    right_content = html.Div(
        className="right_content",
        id=question_analysis_type + "-graphs",
        children=scatterplots,
    )

    content = html.Div(
        className="content",
        children=[
            layouts.menu(question_analysis_type),
            toolbar,
            right_content,
        ],
    )

    return content


def get_statistics_per_student(d):
    # for each task number of trials, if solved or not, avg time spent on solving an exercise, within one period
    studentTaskDataPerDay = d.groupby("day")
    task_stats = {}
    task_stats["Attempt"] = [i for i in range(1, len(studentTaskDataPerDay) + 1)]

    def toYesNo(bool):
        if bool:
            return "yes"
        return "no"

    task_stats["IsSolved"] = map(
        toYesNo, studentTaskDataPerDay["correct"].any().tolist()
    )
    task_stats["NumberOfTrials"] = (
        studentTaskDataPerDay["answer"]
        .count()
        .reset_index(name="count")["count"]
        .tolist()
    )
    task_stats["TimeSpent"] = (
        studentTaskDataPerDay["dateIsoFormat"]
        .apply(lambda x: str(x.max() - x.min()))
        .tolist()
    )
    return pd.DataFrame(data=task_stats)


def datetime_sort_key(dt):
    return datetime.datetime.strptime(dt, "%d-%m-%Y %H:%M:%S")


def sort_by_date(date, other):
    xy = zip(date, other)
    x, y = list(zip(*sorted(xy, key=lambda p: datetime_sort_key(p[0]))))
    return (x, y)

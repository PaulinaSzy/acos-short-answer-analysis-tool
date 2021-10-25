import pandas as pd
import plotly.express as px
from datetime import datetime
import math


path_to_all_data = "data-processed/alldata.pkl"

data_clustered_syntax_error = ""
data_clustered_syntax_progress = ""
data_clustered_semantic_error = ""
data_clustered_semantic_progress = ""


def filter_answers(data, allowed_missing_words_num):
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


def getCorrectPercentage(answer, closest, matchingBeginning, matchingEnd):
    points = 0
    if answer == closest:
        points = 100
    else:
        if len(answer) < len(closest):
            points = math.floor(
                (len(matchingBeginning) + len(matchingEnd)) / len(closest) * 100
            )
        if len(answer) >= len(closest):
            ptsA = math.floor(
                (len(matchingBeginning) + len(matchingEnd)) / len(closest) * 100
            )
            ptsB = math.floor(len(closest) / len(answer) * 100)
            if ptsA > ptsB:
                points = ptsB
            else:
                points = ptsA

    return points


def create_datafr():
    datafr = pd.read_pickle(path_to_all_data)

    datafr["day"] = datafr["date"].apply(lambda elem: elem.day)
    datafr["dateIsoFormat"] = datafr["date"]
    datafr["date"] = datafr["date"].apply(
        lambda elem: datetime.fromisoformat(str(elem)).strftime("%d-%m-%Y %H:%M:%S")
    )
    datafr["score"] = datafr[
        ["answer", "closest", "matchingBeginning", "matchingEnd"]
    ].apply(
        lambda x: getCorrectPercentage(
            x["answer"], x["closest"], x["matchingBeginning"], x["matchingEnd"]
        ),
        axis=1,
    )

    datafr = filter_answers(datafr, allowed_missing_words_num=3)

    return datafr


datafr = create_datafr()

color_sequence = px.colors.qualitative.Dark24

sizesIdx = {False: 15, True: 30}
stylesIdx = {True: "x-open-dot", False: "circle"}

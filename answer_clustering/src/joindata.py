import os
import pandas as pd
import json
from datetime import datetime

# dirs = [
#     "data/logdata1",
#     "data/logdata2",
#     "data/logdata3",
#     "data/logdata4",
#     "data/logdata5",
#     "data/logdata6",
# ]


def check_period(date):
    p1_2019_start = datetime(2019, 9, 9)
    p1_2019_end = datetime(2019, 10, 25)

    if p1_2019_start < date < p1_2019_end:
        return ("p1", 2019)

    p2_2019_start = datetime(2019, 10, 28)
    p2_2019_end = datetime(2019, 12, 13)

    if p2_2019_start < date < p2_2019_end:
        return ("p2", 2019)

    p3_2020_start = datetime(2020, 1, 6)
    p3_2020_end = datetime(2020, 2, 21)

    if p3_2020_start < date < p3_2020_end:
        return ("p3", 2020)

    p4_2020_start = datetime(2020, 2, 24)
    p4_2020_end = datetime(2020, 4, 10)

    if p4_2020_start < date < p4_2020_end:
        return ("p4", 2020)

    p5_2020_start = datetime(2020, 4, 13)
    p5_2020_end = datetime(2020, 5, 29)

    if p5_2020_start < date < p5_2020_end:
        return ("p5", 2020)

    p1_2020_start = datetime(2020, 9, 7)
    p1_2020_end = datetime(2020, 10, 23)

    if p1_2020_start < date < p1_2020_end:
        return ("p1", 2020)

    p2_2020_start = datetime(2020, 10, 26)
    p2_2020_end = datetime(2020, 12, 11)

    if p2_2020_start < date < p2_2020_end:
        return ("p2", 2020)

    p3_2021_start = datetime(2021, 1, 11)
    p3_2021_end = datetime(2021, 2, 26)

    if p3_2021_start < date < p3_2021_end:
        return ("p3", 2021)

    p4_2021_start = datetime(2021, 3, 1)
    p4_2021_end = datetime(2021, 4, 16)

    if p4_2021_start < date < p4_2021_end:
        return ("p4", 2021)

    p5_2021_start = datetime(2021, 4, 19)
    p5_2021_end = datetime(2021, 6, 4)

    if p5_2021_start < date < p5_2021_end:
        return ("p5", 2021)

    else:
        return ("other", date.year)


def create_feedback_tuples(feedback_list):
    return [tuple(l) for l in feedback_list]


def remove_nested_feedback(feedback_list):
    l = [[i[0] if type(i) is list else i for i in j] for j in feedback_list]
    return l


def join_data(file_list):

    data = []
    for filename in file_list:
        log_data = open(filename, "r")
        for line in log_data:
            columns = line.split("\t")
            date = datetime.fromisoformat(columns[0][:-1])
            period, year = check_period(date)
            body = json.loads(columns[1])
            dataentry = body["payload"]
            dataentry["date"] = date
            dataentry["period"] = period
            dataentry["year"] = year
            dataentry["problemName"] = body["problemName"]
            dataentry["uid"] = int(json.loads(columns[2])["uid"])
            dataentry["feedback"] = remove_nested_feedback(dataentry["feedback"])
            dataentry["feedback"] = create_feedback_tuples(dataentry["feedback"])
            if dataentry["problemName"] in [
                "system_noerrors",
                "GPS",
                "cars",
                "LEDs",
                "mobile_payment",
                "online_shopping",
                "Recycling",
                "solar_power",
            ]:
                dataentry["problemType"] = "style"
            else:
                dataentry["problemType"] = "order"

            data.append(dataentry)

    return pd.DataFrame(data)

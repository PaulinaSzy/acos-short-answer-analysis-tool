# acos-short-answer-analysis-tool
An interactive application that allows clustering and analysis of student answers from Acos server short answer type exercises

Answer clustering module provides a NLP pipeline to analyze Acos log data and vizualize the results on interactive dashboard.

## Requirements

The application requires Python 3.8 and pipenv (https://github.com/pypa/pipenv) command line tool to be installed.

Then, run `pipenv install` in the root directory.

## Data analysis

In order to perform analysis of new log data:

- put the log data (in separate files or folders) in the Acos log data format into `answer_clustering/data`
- run command `pipenv run python run_analysis.py` from the folder `answer_clustering`

On regular computer the analysis might last up to few days, depending on the amount of data. 
It is recommended to run it on a computing cluster e.g. Triton (https://scicomp.aalto.fi/triton/).

## Clustering visualisation app

To run visualization application:

- enter the parent folder of `answer_clustering`
- run command `pipenv run python -m index`
- open link `http://127.0.0.1:8050/`




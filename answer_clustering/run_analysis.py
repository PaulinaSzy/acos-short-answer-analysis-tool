import os
from src import joindata as j
from src import functions as f
from src import analysis_embeddings
from src import analysis_syntax
from src import analysis_wordnet
from src import clustering_parameter_selection
from src import clustering
import nltk


def download_models():
    nltk.download("wordnet")
    nltk.download("punkt")
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')


def get_logdata_files():
    logdata_files = []
    for (dirpath, dirnames, filenames) in os.walk("data"):
        logdata_files += [os.path.join(dirpath, file) for file in filenames if file.endswith('.log')]

    return logdata_files


def run():
    download_models()
    print("Fetching NlTK models completed")

    logdata_files = get_logdata_files()
    alldata = j.join_data(logdata_files)
    alldata.to_pickle("data-processed/alldata.pkl")
    print("Data preprocessing completed")

    alldata_filtered = f.filter_test_answers(alldata, 3)
    alldata_filtered.drop_duplicates(subset=["answer"], keep="first", inplace=True)
    data_order = alldata_filtered.loc[alldata_filtered["problemType"] == "order"]

    data_style = alldata_filtered.loc[alldata_filtered["problemType"] == "style"]

    analysis_syntax.run(data_order)
    print("Syntactic analysis completed")

    analysis_embeddings.run(data_style)
    print("Embedding based analysis completed")

    analysis_wordnet.run(data_style)
    print("Wordnet based analysis completed")

    clustering_parameter_selection.run()
    print("Clustering parameter selection completed")

    clustering.run()
    print("Clustering completed")


if __name__ == "__main__":
    run()

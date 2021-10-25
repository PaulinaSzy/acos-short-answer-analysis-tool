#!/usr/bin/env python
# coding: utf-8


import spacy
import pandas as pd
import string
import numpy as np
from stanza.utils.conll import CoNLL
import os
import re
import csv
import argparse
import collections
import functools
import itertools
import json
import textcomplexity
import benepar
import spacy
from spacy.language import Language
import pickle as pkl

from textcomplexity import surface, sentence, dependency, constituency
from textcomplexity.text import Text
from textcomplexity.utils import conllu, custom_tsv

from . import functions as f

benepar.download("benepar_en3")


@Language.component("prevent_sent_segmentation")
def prevent_sentence_boundary_detection(doc):
    for token in doc:
        token.is_sent_start = False
    return doc


nlp = spacy.load("en_core_web_md", disable=["ner"])


if spacy.__version__.startswith("2"):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

nlp.add_pipe("prevent_sent_segmentation", name="prevent-sbd", before="parser")


def parse_string(span):
    constituent_data, position = benepar.integrations.spacy_extensions.get_constituent(
        span
    )
    label_vocab = constituent_data.label_vocab
    doc = span.doc

    idx = position - 1

    def make_str():
        nonlocal idx
        idx += 1
        i, j, label_idx = (
            constituent_data.starts[idx],
            constituent_data.ends[idx],
            constituent_data.labels[idx],
        )
        label = label_vocab[label_idx]
        if (i + 1) >= j:
            token = doc[i]
            s = "*"
        else:
            children = []
            while (
                (idx + 1) < len(constituent_data.starts)
                and i <= constituent_data.starts[idx + 1]
                and constituent_data.ends[idx + 1] <= j
            ):
                children.append(make_str())

            s = u" ".join(children)

        for sublabel in reversed(label):
            s = u"({} {})".format(sublabel, s)
        return s

    return make_str()


def parse_sentence(sent_doc):
    for sent in sent_doc.sents:
        parsed_sentence = "(TOP" + parse_string(sent) + ")"
    return parsed_sentence


def ps_tree_sentence_rows(parsed_sentence):
    tokens = re.split("(.\*)", parsed_sentence)
    joined_tokens = [i + j for i, j in zip(tokens[::2], tokens[1::2])]
    if "*" not in tokens[-1]:
        joined_tokens[-1] += tokens[-1]
    return map(lambda x: x.replace(" ", ""), joined_tokens)


Result = collections.namedtuple(
    "Result", ["name", "value", "stdev", "length", "length_stdev"]
)


def surface_based(tokens, window_size, all_measures):
    """"""
    results = []
    measures = [
        (surface.type_token_ratio, "type-token ratio", True),
        (surface.guiraud_r, "Guiraud's R", False),
        (surface.herdan_c, "Herdan's C", False),
        (surface.dugast_k, "Dugast's k", False),
        (surface.maas_a2, "Maas' a²", False),
        (surface.dugast_u, "Dugast's U", False),
        (surface.tuldava_ln, "Tuldava's LN", False),
        (surface.brunet_w, "Brunet's W", False),
        (surface.cttr, "CTTR", False),
        (surface.summer_s, "Summer's S", False),
        (surface.sichel_s, "Sichel's S", True),
        (surface.michea_m, "Michéa's M", False),
        (surface.honore_h, "Honoré's H", True),
        (surface.entropy, "Entropy", True),
        (surface.evenness, "Evenness", True),
        (surface.yule_k, "Yule's K", False),
        (surface.simpson_d, "Simpson's D", True),
        (surface.herdan_vm, "Herdan's Vm", False),
        (surface.hdd, "HD-D", True),
        (surface.average_token_length, "average token length", True),
        (surface.orlov_z, "Orlov's Z", True),
    ]
    for measure, name, subset in measures:
        if all_measures or subset:
            name += " (disjoint windows)"
            mean, stdev, _ = surface.bootstrap(
                measure, tokens, window_size, strategy="spread"
            )
            results.append(Result(name, mean, stdev, None, None))
    text = Text.from_tokens(tokens)
    mattr = surface.mattr(text, window_size)
    results.append(Result("type-token ratio (moving windows)", mattr, None, None, None))
    mtld = surface.mtld(text)
    results.append(Result("MTLD", mtld, None, None, None))
    return results


def sentence_based(sentences, punct_tags):
    """"""
    results = []
    pps = functools.partial(sentence.punctuation_per_sentence, punctuation=punct_tags)
    ppt = functools.partial(sentence.punctuation_per_token, punctuation=punct_tags)
    measures = [
        (sentence.sentence_length_words, "average sentence length (words)"),
        (sentence.sentence_length_characters, "average sentence length (characters)"),
        (pps, "punctuation per sentence"),
    ]
    for measure, name in measures:
        value, stdev = measure(sentences)
        results.append(Result(name, value, stdev, None, None))
    results.append(Result("punctuation per token", ppt(sentences), None, None, None))
    return results


def dependency_based(graphs):
    """"""
    results = []
    measures = [
        (dependency.average_dependency_distance, "average dependency distance"),
        (dependency.closeness_centrality, "closeness centrality"),
        (dependency.outdegree_centralization, "outdegree centralization"),
        (dependency.closeness_centralization, "closeness centralization"),
        (dependency.longest_shortest_path, "longest shortest path"),
        (dependency.dependents_per_word, "dependents per word"),
    ]
    for measure, name in measures:
        value, stdev = measure(graphs)
        results.append(Result(name, value, stdev, None, None))
    return results


def constituency_based(trees, lang):
    """"""
    results = []
    measures_with_length = [
        (constituency.t_units, "t-units"),
        (constituency.complex_t_units, "complex t-units"),
        (constituency.clauses, "clauses"),
        (constituency.dependent_clauses, "dependent clauses"),
        (constituency.nps, "noun phrases"),
        (constituency.vps, "verb phrases"),
        (constituency.pps, "prepositional phrases"),
        (constituency.coordinate_phrases, "coordinate phrases"),
    ]
    measures_wo_length = [
        (constituency.constituents, "constituents"),
        (constituency.constituents_wo_leaves, "non-terminal constituents"),
        (constituency.height, "parse tree height"),
    ]
    if lang == "de_negra":
        for measure, name in measures_with_length:
            value, stdev, length, length_sd = measure(trees)
            results.append(Result(name, value, stdev, length, length_sd))
    for measure, name in measures_wo_length:
        value, stdev = measure(trees)
        results.append(Result(name, value, stdev, None, None))
    return results


def answer_to_tsv(answer, filename):
    tsv_file = open(filename, "w", newline="")
    tsv_writer = csv.writer(
        tsv_file,
        delimiter="\t",
    )
    answer_doc = nlp(answer)
    doc_sents = list(answer_doc.sents)
    parsed_answer = parse_sentence(answer_doc)
    ps_tree_list = list(ps_tree_sentence_rows(parsed_answer))
    for i in range(2):
        for token, ps_tree_token in zip(doc_sents[0], ps_tree_list):
            if token.dep_ == "ROOT":
                head_i = -1
                dep = "--"
            else:
                head_i = token.head.i
                dep = token.dep_
            tsv_writer.writerow(
                [token.i, token.text, token.pos_, head_i, dep, ps_tree_token]
            )
        tsv_writer.writerow("")

    tsv_file.close()


def get_feature_vector_from_duplicate_sent(filename, features):
    fil = open(filename, "r", newline="")
    tokens, tagged, graphs, ps_trees = zip(*custom_tsv.read_tsv_sentences(fil))
    tokens = list(itertools.chain.from_iterable(tokens))
    results = []
    if features == "sentence":
        results.extend(sentence_based(tagged, "PUNCT"))
    elif features == "dependency":
        results.extend(dependency_based(graphs))
    elif features == "constituency":
        results.extend(constituency_based(ps_trees, "none"))
    else:
        results.extend(sentence_based(tagged, "PUNCT"))
        results.extend(dependency_based(graphs))
        results.extend(constituency_based(ps_trees, "none"))
    feature_vector = []
    for result in results:
        feature_vector.append(result.value)
    fil.close()
    return feature_vector


def answer_to_vector(answer, filename, features):
    answer_cleaned = f.clean_answer(answer, remove_all=False)
    answer_to_tsv(answer_cleaned, filename)
    vector = get_feature_vector_from_duplicate_sent(filename, features)
    return vector


def create_feature_vectors(data, features):
    order_exercise_names = list(set(data["problemName"].tolist()))
    # if analysis_type == "error":
    #     data = data.loc[data["answer"].apply(f.is_answer_correctly_spelled)]
    data_per_exercise = {}
    for ex in order_exercise_names:
        ex_data = {}
        all_data = data.loc[data["problemName"] == ex]
        answers = all_data["answer"].tolist()
        closest = all_data["closest"].tolist()
        combined_answers = list(set(answers + closest))
        ex_data["data"] = combined_answers
        feature_vectors = []
        for answer in combined_answers:
            feature_vectors.append(answer_to_vector(answer, "output.tsv", features))
        ex_data["embeddings"] = feature_vectors
        ex_data["matrix"] = f.create_similarity_matrix(feature_vectors)
        data_per_exercise[ex] = ex_data

    return data_per_exercise


def run(alldata):
    feature_selection = ["sentence", "dependency", "constituency", "allsyntax"]
    analysis_type = ["error", "progress"]

    for feature in feature_selection:
        data_per_exercise = create_feature_vectors(alldata, feature)
        for analysis in analysis_type:
            with open(
                "data-processed/data_per_exercise_" + feature + "_" + analysis, "wb"
            ) as pkl_file:
                pkl.dump(data_per_exercise, pkl_file)

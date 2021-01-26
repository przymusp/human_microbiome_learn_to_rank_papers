# pylint: disable = invalid-name, missing-docstring
import csv
import json
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.matutils import corpus2dense
from gensim.models import TfidfModel
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler


def get_annotated_papers(name):
    with open(name, "r") as f:
        return set([l.strip().split("/")[-1] for l in f.readlines()])


def weights_f_classif(X, Y):
    weights = f_classif(X, Y)
    weights = weights[0]
    return _weights_normalize(weights)


def _weights_normalize(weights):
    weights_sum = weights.sum()
    if weights_sum > 0:
        weights /= weights_sum
    return weights


def l2r(X, Y):

    # Select initial weights
    fs = SelectKBest(k=200)
    X = fs.fit_transform(X, Y)

    v = weights_f_classif(X, Y)

    # Compute initial ranking
    rank = np.dot(X, v)
    # Compute ameliorated ranking (move known relevant papers to top of the ranking)
    rank = rank + 1 * Y * rank

    rank = rank.reshape(-1, 1)

    scale = MinMaxScaler()
    X_s = scale.fit_transform(X)
    rank = scale.fit_transform(rank)
    predict = SGDRegressor()

    predict.fit(X_s, rank.ravel())

    data_p = predict.predict(X_s)
    m = np.min(data_p[Y == 1]) * 1
    rank_match = (data_p >= m) | (Y == 1)
    return rank_match, data_p

def read_biblio_csv(fname):
    titles = defaultdict(dict)
    with open(fname, "r") as f:
        r = csv.DictReader(f)
        for row in r:

            key = row["File Attachments"].split("/")[-1]
            columns = ["Item Type", "Publication Year", "Author", "Title", "Publication Title", "DOI", "Url"]
            titles[key] = { k:row[k] for k in columns }
    return titles


def annotate_papers():

    # Zotero like CSV describing bibliography (export library to CSV) 
    titles = read_biblio_csv("papers.csv")

    # list of annotated papers (hand picked or automatic categories) 
    anotated_papers = get_annotated_papers("anotated_check.csv")
    anotated_ml = get_annotated_papers("anotated_ml.csv")
    anotated_human = get_annotated_papers("anotated_human.csv")

    try:
        with open("data_clean.json", "r") as fin:
            data = OrderedDict(json.load(fin))
    except Exception as e:
        print("Error: File data_clean.json not found. See README.md")
        exit()

    data_dict = defaultdict(lambda: defaultdict(int))

    for d in data:
        data_dict[d]["papers"] = Counter(data[d])

        fname = d.strip().split("/")[-1].replace(".txt", ".pdf")

        data_dict[d].update(titles[fname])

        if fname in anotated_papers:
            data_dict[d]["is_ml"] = 1
            data_dict[d]["is_human"] = 1
            data_dict[d]["is_anotated"] = 1

        if fname in anotated_ml:
            data_dict[d]["is_ml"] = 1
            data_dict[d]["is_anotated"] = 1

        if fname in anotated_human:
            data_dict[d]["is_human"] = 1
            data_dict[d]["is_anotated"] = 1


    return data_dict


def compute_X(papers):
    dictionary = Dictionary([doc["papers"] for _, doc in papers.iterrows()])
    num_docs = dictionary.num_docs
    num_terms = len(dictionary.keys())

    corpus_bow = [dictionary.doc2bow(doc["papers"]) for _, doc in papers.iterrows()]

    tfidf = TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]
    return corpus2dense(corpus_tfidf, num_terms, num_docs).T


def main():

    data = annotate_papers()
    if not data:
        print("Error: No data found in data_clean.json. See README.md")
        exit()

    df = pd.DataFrame.from_dict(data, orient="index")
    df.fillna(0, inplace=True)

    X = compute_X(df)

    # rank papers by human category 
    rank_human, data_human = l2r(X, np.array(df["is_human"]))
    df["Human score"] = data_human

    # rank papers by ml category 
    rank_ml, data_ml = l2r(X, np.array(df["is_ml"]))
    df["ML score"] = data_ml

    # papers classified in both categories
    rank_combined = rank_ml & rank_human
    df["interesting_score"] = rank_combined

    # combined score
    df["Combined score"] =  0.5 * df["Human score"] + 0.5 * df["ML score"]

    # merge known annotations
    df["is_human_ml"] = (df["is_human"] == 1) & (df["is_ml"] == 1)

    # prepare output 
    df.drop(["papers"], axis=1, inplace=True)
    columns = ["Author", "Title", "Item Type", "Publication Year",  "Publication Title", "DOI", "Url", "is_ml", "is_human", "is_anotated", "is_human_ml", "Human score", "ML score", "Combined score", "interesting_score"]
    df = df[columns].sort_values(by=["interesting_score", "Combined score", "ML score", "Human score"], ascending=False)
    df.to_excel("check.xlsx", index=False)


if __name__ == "__main__":
    main()

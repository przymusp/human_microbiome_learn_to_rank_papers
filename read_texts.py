import glob
import json
import sys
import os

import tqdm
from gensim.parsing.preprocessing import preprocess_string
from joblib import Parallel, delayed



def read_and_preprocess(fname):

    txt_file, _ = os.path.splitext(fname)
    txt_file += ".txt"
    if not os.path.isfile(txt_file):
        os.system(f'pdftotext "{fname}" "{txt_file}"')

    with open(txt_file, "r") as fout:
        txt = fout.read()
    return fname, preprocess_string(txt)


if __name__ == "__main__":
    oname = "data_clean.json"
    dname = "files"
    print(sys.argv[0] + " [oname] [directory]")
    print(f" oname = {oname} \n directory = {dname}")

    if 1 < len(sys.argv) <= 3:
        oname = sys.argv[1]
    if len(sys.argv) == 3:
        dname = sys.argv[2]

    files = {}
    glob_exp = "%s/**/*.pdf" % (dname)
    results = Parallel(n_jobs=-1)(
        delayed(read_and_preprocess)(fname)
        for fname in tqdm.tqdm(glob.glob(glob_exp, recursive=True))
    )
    results = dict(results)

    if not results:
        print("Error: No files found in files directory. Read README.md")

    with open(oname, "w") as fin:
        json.dump(results, fin)

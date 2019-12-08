import pandas as pd
from sklearn.model_selection import train_test_split

import os
import re

punct = re.compile(r"([\.\,\?\!\:])")
ent = re.compile(r"(\w)(\<ENT\>)")

data = {
    "training": {
        "literal": ["conll_literal.txt", "semeval_literal_train.txt"],
        "metonymic": ["conll_metonymic.txt", "semeval_metonymic_train.txt"]
    },
    "testing": {
        "literal": ["semeval_literal_test.txt"],
        "metonymic": ["semeval_metonymic_test.txt"]
    }
}


def tokenize(text):
    return [ent.sub(r"\1 \2", punct.sub(r" \1", t)).split() for t in text]


def getSpan(text):
    s = []
    l = []
    r = []
    for v in text:
        i = 0
        left = -1
        right = -1
        while i < len(v):
            if "<ENT>" in v[i]:
                v[i] = v[i].replace("<ENT>", "")
                if left == -1:
                    left = i
                elif right == -1:
                    right = i
                    break

            i += 1

        v = list(filter(None, v))

        s.append(" ".join(v).lower())
        l.append(left)
        r.append(right)

    return s, l, r


def prep_metonymy():

    training_split = 0.75

    for split in data:
        splitd = pd.DataFrame(columns=["group", "entity", "sentence"])
        for group in data[split]:
            for file in data[split][group]:
                fd = pd.read_csv(os.path.join("epnl/data/metonymy/", file), delimiter="<SEP>", names=["entity", "sentence"])
                fd["sentence"] = tokenize(fd["sentence"])
                fd["group"] = group

                sentences, span_left, span_right = getSpan(fd["sentence"].values)

                fd["sentence"] = sentences
                fd["span1L"] = span_left
                fd["span1R"] = span_right

                splitd = pd.concat([splitd, fd], axis=0, sort=False)

                print(split, group, file)

        print(splitd.shape)
        print("Literal:", sum(splitd["group"] == "literal"))
        print("Metonymic:", sum(splitd["group"] == "metonymic"))

        fpt = f"epnl/data/metonymy/metonymy_{split}_train.csv"
        fpd = f"epnl/data/metonymy/metonymy_{split}_dev.csv"

        if not os.path.exists(fpt):
            with open(fpt, "w+"): pass

        if not os.path.exists(fpd):
            with open(fpd, "w+"): pass

        train, dev = train_test_split(splitd, shuffle=True, train_size=training_split)

        train.to_csv(fpt, index=False)
        dev.to_csv(fpd, index=False)


if __name__ == '__main__':
    prep_metonymy()
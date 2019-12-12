import pandas as pd
from sklearn.model_selection import train_test_split

import os
import re

punct = re.compile(r"([\.\,\?\!\:])")
ent = re.compile(r"(\w)(\<ENT\>)")

data = {
    "literal": ["conll_literal.txt", "semeval_literal_train.txt", "relocar_literal_train.txt", "wiki_literal.txt",
                "semeval_literal_test.txt", "relocar_literal_test.txt"],
    "metonymic": ["conll_metonymic.txt", "semeval_metonymic_train.txt", "relocar_metonymic_train.txt",
                  "wiki_metonymic.txt", "semeval_metonymic_test.txt", "relocar_metonymic_test.txt"]
}


def tokenize(text):
    return [ent.sub(r"\1 \2", punct.sub(r" \1", t)).split() for t in text]


def get_span(text):
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
    development_split = 0.80

    split = pd.DataFrame(columns=["group", "entity", "sentence"])
    for group in data:
        for file in data[group]:
            fd = pd.read_csv(os.path.join("epnl/data/metonymy/", file), delimiter="<SEP>", names=["entity", "sentence"])
            fd["sentence"] = tokenize(fd["sentence"])
            fd["group"] = group

            sentences, span_left, span_right = get_span(fd["sentence"].values)

            fd["sentence"] = sentences
            fd["span1L"] = span_left
            fd["span1R"] = span_right

            split = pd.concat([split, fd], axis=0, sort=False)

            print(group, file)

    print(split.shape)
    print("Literal:", sum(split["group"] == "literal"))
    print("Metonymic:", sum(split["group"] == "metonymic"))

    fpt = f"epnl/data/metonymy/metonymy_train.csv"
    fpd = f"epnl/data/metonymy/metonymy_dev.csv"
    fps = f"epnl/data/metonymy/metonymy_testing.csv"

    if not os.path.exists(fpt):
        with open(fpt, "w+"): pass

    if not os.path.exists(fpd):
        with open(fpd, "w+"): pass

    if not os.path.exists(fps):
        with open(fps, "w+"): pass

    train, test = train_test_split(split, shuffle=True, train_size=training_split)
    train, dev = train_test_split(train, shuffle=True, train_size=development_split)

    train.to_csv(fpt, index=False)
    dev.to_csv(fpd, index=False)
    test.to_csv(fps, index=False)


if __name__ == '__main__':
    prep_metonymy()
import re

import pandas as pd
from sklearn.model_selection import train_test_split

punct = re.compile(r"([\.\,\?\!\:])")

sentence_re = re.compile(r"(\d{1,5})\t\"(.+)\"")
relation_re = re.compile(r"^(.+)\((\w{2}),(\w{2})\)")


def getSpan(sentence):
    tokens = punct.sub(r" \1", sentence).split()

    left1, right1, left2, right2 = -1, -1, -1, -1

    for i, tok in enumerate(tokens):
        nw = tok
        if right1 == -1:
            if "<e" in tok:
                left1 = i
                nw = nw[4:]

            if "</e" in tok:
                right1 = i + 1
                nw = nw[:-5]

        else:
            if "<e" in tok:
                left2 = i
                nw = nw[4:]

            if "</e" in tok:
                right2 = i + 1
                nw = nw[:-5]

        tokens[i] = nw

    return " ".join(tokens), left1, right1, left2, right2


def parse_file(path):
    samples = []
    sample = {}

    with open(path, "r") as file:
        for i, line in enumerate(file.readlines()):

            l = line.strip()

            sentence_match = sentence_re.search(l)
            relation_match = relation_re.search(l)

            if l == "Other":
                sample["relation"] = "Other"
            elif l.startswith("Comment:"):
                pass
            elif sentence_match:
                sample["ix"] = sentence_match.group(1)
                sentence, left1, right1, left2, right2 = getSpan(sentence_match.group(2))
                sample["sentence"] = sentence
                sample["left1"] = left1
                sample["right1"] = right1
                sample["left2"] = left2
                sample["right2"] = right2
            elif relation_match:
                dir = "12" if relation_match.group(2) == "e1" else "21"
                sample["relation"] = relation_match.group(1) + dir
            elif line.strip() == "" and sample:
                samples.append(sample)
                sample = {}

    return pd.DataFrame(samples)


def prep_rel():
    print("Preparing Relationship Classification data")

    train_split = 0.85

    training_file = "epnl/data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
    testing_file = "epnl/data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"

    data = parse_file(training_file)
    train, dev = train_test_split(data, train_size=train_split)

    train.to_csv("epnl/data/rel/training.csv", index=False)
    dev.to_csv("epnl/data/rel/dev.csv", index=False)

    print(f"Training data created: {len(train):,}")
    print(f"Development data created: {len(dev):,}")

    data = parse_file(testing_file)
    data.to_csv("epnl/data/rel/testing.csv", index=False)

    print(f"Testing data created: {len(data):,}")


if __name__ == '__main__':
    prep_rel()
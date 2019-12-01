import pandas as pd
from sklearn.model_selection import train_test_split

### TroFi

def prep_trofi():

    training_split = 0.75

    data = []
    with open("epnl/data/trofi/TroFiExampleBase.txt", "r") as f:
        current_kw = ""

        for ix, line in enumerate(f):

            if line.startswith("***"):
                current_kw = line[3:-4]
            elif line.startswith("wsj"):
                sp = line.split("\t")
                if sp[1] == "U":
                    continue

                sent = sp[2].split()
                ix = 0
                while ix < len(sent):
                    if current_kw in sent[ix]:
                        break
                    ix += 1

                data.append({"kw": current_kw, "label": sp[1], "sentence": sp[2].strip(), "kw_ix": ix})

    data = pd.DataFrame(data)
    train, test = train_test_split(data, shuffle=True, train_size=training_split)
    train, dev = train_test_split(train, shuffle=True, train_size=training_split)
    train.to_csv("epnl/data/trofi/train.csv")
    dev.to_csv("epnl/data/trofi/dev.csv")
    test.to_csv("epnl/data/trofi/test.csv")

    print("TroFi datasets created")

###

if __name__ == '__main__':
    prep_trofi()
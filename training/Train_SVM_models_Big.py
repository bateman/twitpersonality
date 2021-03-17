import os

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR

import datasetUtils as dsu
import embeddings

# configs
post_threshold = 3
dataset_path = "../FastText/dataset.vec"
try:
    os.makedirs("Models/MPBig")
except FileExistsError:
    # directory already exists
    pass

types = {"userid": str, "ope": float, "con": float, "ext": float, "agr": float, "neu": float, "message": str}
print("Loading training set...")
df = pd.read_csv("dataset/join.csv", usecols=[0, 1, 2, 3, 4, 5, 6], dtype=types, error_bad_lines=False, engine="python",
                 nrows=20000000)
print("Training set correctly loaded.")

# shuffle
dfs = df.sample(3000)
print("Training set shuffled.")

print("Loading embeddings dataset...")
wordDictionary = dsu.parseFastText(dataset_path)
print("Dataset correctly laoded.")

posts = dfs["message"]
yO = np.array(dfs["ope"], dtype=pd.Series)
yC = np.array(dfs["con"], dtype=pd.Series)
yE = np.array(dfs["ext"], dtype=pd.Series)
yA = np.array(dfs["agr"], dtype=pd.Series)
yN = np.array(dfs["neu"], dtype=pd.Series)
[conE, yO, yC, yE, yA, yN] = embeddings.transformTextForTraining(wordDictionary, post_threshold, posts, yO, yC, yE, yA,
                                                                 yN, "conc", True)
print("\tEmbeddings computed.")

trait = 1
for labels in [yO, yC, yE, yA, yN]:

    if trait == 1:
        big5trait = "O"
        gamma = 1
        C = 1
        print("   Training model for Openness...")
    elif trait == 2:
        big5trait = "C"
        gamma = 1
        C = 1
        print("   Training model for Conscientiousness...")
    elif trait == 3:
        big5trait = "E"
        gamma = 1
        C = 10
        print("   Training model for Extraversion...")
    elif trait == 4:
        big5trait = "A"
        gamma = 1
        C = 1
        print("   Training model for Agreeableness...")
    elif trait == 5:
        big5trait = "N"
        gamma = 10
        C = 10
        print("   Training model for Neuroticism...")
    trait += 1

    model = SVR(kernel='rbf', gamma=gamma, C=C).fit(conE, labels)
    model_name = "SVM_Big_conc_" + big5trait + ".pkl"
    joblib.dump(model, "Models/MPBig/" + model_name)

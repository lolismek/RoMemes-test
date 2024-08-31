import pandas as pd
import os

df = pd.read_csv("corpus/metadata.tsv", sep = "\t")

texts = []
for filename in os.listdir("corpus/text"):
    with open("corpus/text/" + filename, "r") as file:
        text = file.read()
        texts.append(text)
    file.close()

df['text'] = texts
df = df[['ID', 'Political', 'text']]

df.to_csv("political_texts.csv")
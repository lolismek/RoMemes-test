from huggingface_hub import hf_hub_download
from tqdm import tqdm
import fasttext
import os

model = fasttext.load_model(hf_hub_download("facebook/fasttext-language-identification", "model.bin"))



PREDICT_FILES = "ocr_results/tessdata-best_filters-90"
REF_FILES = "corpus/text"


preds = []
for filename in os.listdir(REF_FILES):
    with open(PREDICT_FILES + "/" + filename, "r") as file:
        pred_text = file.read()
    file.close()

    new_pred_text = []
    for x in pred_text.split("\n"):
        if len(x) <= 10:
            p = model.predict(x, k = 2)
            if p[0][0] == "__label__ron_Latn" or p[0][1] == "__label__ron_Latn":
                print("!!!")
                new_pred_text.append(x)
            else:
                continue
                
        new_pred_text.append(x)
    
    if len(new_pred_text) == 0:
        new_pred_text = ""
    else:
        new_pred_text = "\n".join(new_pred_text)
    
    # pred_text = "\n".join(x for x in pred_text.split("\n"))

    preds.append(new_pred_text)

len(preds)

import pandas as pd
df = pd.DataFrame()
df['texts'] = preds
df.to_csv(PREDICT_FILES + "/" + "LD.csv")
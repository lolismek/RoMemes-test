import pandas as pd
import sacrebleu
import os
from unidecode import unidecode

PREDICT_FILES = "ocr_results/tessdata-best_filters-90"
REF_FILES = "corpus/text"

bleu_calc = sacrebleu.BLEU()
chrf_calc = sacrebleu.CHRF(word_order=2)

preds, refs = [], []

for filename in os.listdir(REF_FILES):
    with open(REF_FILES + "/" + filename, "r") as file:
        ref_text = file.read()
    file.close()

    with open(PREDICT_FILES + "/" + filename, "r") as file:
        pred_text = file.read()
    file.close()

    new_pred_text = []
    for x in pred_text.split("\n"):
        if len(x) <= -1:
            continue
        new_pred_text.append(x)
    
    if len(new_pred_text) == 0:
        new_pred_text = ""
    else:
        new_pred_text = "\n".join(new_pred_text)
    
    # pred_text = "\n".join(x for x in pred_text.split("\n"))

    preds.append(new_pred_text)
    refs.append(ref_text)

df = pd.read_csv(PREDICT_FILES + "/GEC.csv")
alt = df['texts'].tolist()
alt = [x if isinstance(x, str) else "" for x in alt]

print(bleu_calc.corpus_score(alt, [refs]))
print(chrf_calc.corpus_score(alt, [refs]))
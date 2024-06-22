from torchmetrics.functional.text import bleu_score
from torchmetrics.text import CHRFScore
import pandas as pd
import sys

df = pd.read_csv('data2.csv', sep=';')

categories = ['google_raw_all', 'google_raw_clean', 'google_preproc_all', 'google_preproc_clean', 'tess_preproc']

# the reference, hand-made, adnotations
aux = df['gold'].tolist()
gold = []
for x in aux:
    gold.append([x])

chrf = CHRFScore()

for category in categories:
    cand = df[category].tolist()
    cand = [str(x) for x in cand] # ???

    match = 0
    for i in range(0, len(gold)):
        if cand[i] == gold[i][0]:
            match += 1

    print(">> " + category)
    print(bleu_score(cand, gold))
    print(chrf(cand, gold))
    print(match / len(gold))


# >> OCR with Google API, on raw images
#     BLEU: 0.5759
#     CHRF: 0.8439
#     acc: 0.14
# >> OCR with Google API, on raw images, removing text that is not part of meme caption
#     BLEU: 0.5470
#     CHRF: 0.8421
#     acc: 0.14
# >> OCR with Google API, on preprocessed images
#     BLEU: 0.5516
#     CHRF: 0.7546
#     acc: 0.13
# >> OCR with Google API, on preprocessed images, removing text that is not part of meme caption 
#     BLEU: 0.5398
#     CHRF: 0.7520
#     acc: 0.14
# >> OCR with Tesseract, on preprocessed images
#     BLEU: 0.3447
#     CHRF: 0.6301
#     acc: 0.06
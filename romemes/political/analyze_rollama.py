import pandas as pd

df1 = pd.read_csv("rollama.csv")

preds = df1['ans'].tolist()
real = df1['Political'].tolist()

TP, TN, FP, FN = 0, 0, 0, 0
for i, pred in enumerate(preds):
    if "DA" in pred and "NU" in pred:
        continue
    if "DA" not in pred and "NU" not in pred:
        print(pred)
        continue

    x = 0
    if "DA" in pred or "Da" in pred:
        x = 1

    y = 0
    if real[i] == "Yes":
        y = 1

    if x == 1 and y == 1:
        TP += 1
    elif x == 1 and y == 0:
        FP += 1
    elif x == 0 and y == 0:
        TN += 1
    else:
        FN += 1

print(TP, TN, FP, FN)
    
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print(accuracy, precision, recall, f1_score)
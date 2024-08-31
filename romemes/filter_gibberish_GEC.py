import pandas as pd
import sacrebleu
from tqdm import tqdm
import os
from torch.utils.data import Dataset

from transformers import pipeline
pipe = pipeline("text2text-generation", model="readerbench/RoGEC-mt0-base")

TEXTS_PATH = "ocr_results/tessdata-best_filters-90"

texts = []
for filename in tqdm(os.listdir(TEXTS_PATH)):
    with open(TEXTS_PATH + "/" + filename, "r") as file:
        input_text = file.read()
    file.close()
    texts.append(input_text)

class ListDataset(Dataset):
    
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]
    
texts = ListDataset(texts)


corrected = []
for out in tqdm(pipe(texts, max_length = 1024)):
    corrected.append(out)

import pandas as pd
df = pd.DataFrame()
df['texts'] = corrected

corrected2 = []
for x in corrected:
    corrected2.append(x[0]['generated_text'])

df['texts'] = corrected2
df.to_csv("GEC.csv")



import pandas as pd
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import re
import sys
import csv

LIM = 15
# sheets -> tsv -> csv pipeline!!!

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/alexjerpelea/Desktop/memes/reflecting-card-426014-t7-1b8d496de916.json'

def detect_text(path):
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    # in case no text found
    if len(texts) < 1:
        return '', []



    # words with very small bounding boxes may not be part of meme caption,
    # but may be stuff like product labels, watermarks, etc.
    # here we try to remove these words
    ########
    all_text = texts[0].description
    small_words = set()

    for ind, text in enumerate(texts):
        if ind > 0:
            min_y = min(vertex.y for vertex in text.bounding_poly.vertices)
            max_y = max(vertex.y for vertex in text.bounding_poly.vertices)
            if max_y - min_y < LIM:
                small_words.add(text.description)

    aux = re.findall(r'\w+|[^\w\s]', all_text)
    cleaned_text = []
    for word in aux:
        if word not in small_words:
            cleaned_text.append(word)
    ########

    # return the raw text from API + text hopefully without non-caption words
    return all_text, cleaned_text
    

# preprocess image
def preprocess(im):
    if im.shape[-1] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)

    im= cv2.bilateralFilter(im, 5, 50, 50)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 240, 255, 1)
    return im

# save the preprocessed image and send it to API
def save_image(im):
    plt.figure(figsize=(10,10))
    plt.imshow(im, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('special.png')
    plt.close()

# removes multiple spaces and newlines
def light_text_clean(s):
    return re.sub(r'\s+', ' ', s.strip(), flags=re.MULTILINE)

# joins words and symbols obtained after the cleaning process when extracting from google API
def join_word_list(arr):
    s = ''
    for i, x in enumerate(arr):
        if arr[i] == '-':
            s += x
        elif i + 1 < len(arr) and arr[i + 1] not in ('.', ',', '!', '?', ':', ';', '-'):
            s += x
            s += ' '
        else:
            s += x
    return s

df = pd.read_csv('data.csv', sep=';')
memes = os.listdir('memes_path')

data = [['id', 'gold', 'google_raw_all', 'google_raw_clean', 'google_preproc_all', 'google_preproc_clean', 'tess_preproc']]

for meme in memes:
    try:
        meme_path = 'memes_path/' + meme


        # get gold text
        row = df.loc[df['Image'] == meme].iloc[0]
        gold_text = row['Text']

        # get OCR from raw image with google API
        result1, result1_clean = detect_text(meme_path)

        # get OCR from preprocessed image with google API
        im = np.array(Image.open(meme_path))
        im = preprocess(im)
        save_image(im)
        result2, result2_clean = detect_text('special.png')

        # get OCR from preprocessed image with tesseract
        custom_config = r"--oem 3 --psm 11"
        result3 = pytesseract.image_to_string(im, lang='ron', config = custom_config)

        # final processings:
        result1 = light_text_clean(result1)
        result2 = light_text_clean(result2)
        result3 = light_text_clean(result3)
        result1_clean = join_word_list(result1_clean)
        result2_clean = join_word_list(result2_clean)

        row = [meme, gold_text, result1, result1_clean, result2, result2_clean, result3]
        data.append(row)
    except:
        print(">> " + meme_path)


with open('data2.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=';')
    csv_writer.writerows(data)
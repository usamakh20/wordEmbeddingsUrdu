import requests
import pandas as pd
from bs4 import BeautifulSoup

base_translation_url = 'https://translate.ijunoon.com/'


# Todo: Generalize transliteration and translation function for large text and failed requests
def translate(text):
    while True:
        try:
            r = requests.get(base_translation_url, params={'text': text})
            soup = BeautifulSoup(r.text, 'html.parser')
            return soup.find('div', id='ctl00_inpageResult').find_all('p')[-1].text.strip()
        except Exception as e:
            print(e)
            pass


# Todo: Place in functions
df = pd.read_csv('data/SimLex-999.txt', sep='\t')

for i, row in df.iterrows():
    df.at[i, 'word1'] = translate(row['word1'])
    df.at[i, 'word2'] = translate(row['word2'])

df.to_csv('data/SimLex-999_urdu.txt', sep='\t', index=False)

df = pd.read_csv('data/wordsim_similarity_goldstandard.txt', sep='\t', header=None)

for i, row in df.iterrows():
    df.at[i, 0] = translate(row[0])
    df.at[i, 1] = translate(row[1])

df.to_csv('data/wordsim_similarity_goldstandard_urdu.txt', sep='\t', header=None, index=False)

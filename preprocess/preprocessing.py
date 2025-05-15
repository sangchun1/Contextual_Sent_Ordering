# 🔧 preprocessing.py
# 모듈화를 위하여 .py 생성

import pandas as pd
import itertools

## ✅ Pairwise 전처리 함수
def preprocess_pairwise(df):
    data = []
    for _, row in df.iterrows():
        sentences = [row[f'sentence_{i}'] for i in range(4)]
        answer = [row[f'answer_{i}'] for i in range(4)]
        ordered = [sentences[i] for i in answer]
        positive_pairs = [(ordered[i], ordered[i+1]) for i in range(3)]
        all_pairs = list(itertools.permutations(sentences, 2))
        for s1, s2 in all_pairs:
            label = 1 if (s1, s2) in positive_pairs else 0
            data.append({'sentence1': s1, 'sentence2': s2, 'label': label})
    return pd.DataFrame(data)

## ✅ Seq2Seq 전처리 함수
def preprocess_seq2seq(df):
    data = []
    for _, row in df.iterrows():
        sentences = [row[f'sentence_{i}'] for i in range(4)]
        answers = [row[f'answer_{i}'] for i in range(4)]
        input_text = ' [SEP] '.join(sentences)
        target_text = ' '.join(map(str, answers))
        data.append({'input_text': input_text, 'target_text': target_text})
    return pd.DataFrame(data)
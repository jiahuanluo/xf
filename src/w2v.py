import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle
import logging
import time

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def w2v(dfs, f, L=128):
    print("w2v", f)
    sentences = []
    # if f == "time":
    #     for df in dfs:
    #         for line in df[f].values:
    #             line = line.strip().replace('[', '').replace(']', '').split(',')
    #             # line = [time.strftime("%Y%m%d", time.localtime(float(t) / 1000)) for t in line]
    #             sentences.append(line)
    # else:
    for df in dfs:
        for line in df[f].values:
            line = line.strip().replace('[', '').replace(']', '').split(', ')
            sentences.append(line)
    print("Sentence Num {}".format(len(sentences)))
    w2v = Word2Vec(sentences, window=8, vector_size=L, min_count=1, sg=1, workers=20, epochs=3)
    print("save w2v to {}".format(os.path.join('data', f + ".{}d".format(L))))
    pickle.dump(w2v, open(os.path.join('data', f + ".{}d".format(L)), 'wb'))


if __name__ == "__main__":
    train_df = pd.read_csv('./data/tran_test_transform.csv')
    test_df = pd.read_csv('./data/tran_test_transform.csv')
    # 训练word2vector，维度为128
    w2v([train_df, test_df], 'tagid', L=128)
    w2v([train_df, test_df], 'sorted_tagid', L=128)
    w2v([train_df, test_df], 'time', L=128)
    w2v([train_df, test_df], 'sorted_time', L=128)

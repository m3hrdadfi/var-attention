import os
import gensim
import numpy as np
from nltk.tokenize import WhitespaceTokenizer
from utils import data_utils

W2V_DIR = './outputs/data/'
DATA_DIR = './data/'


def main():
    if not os.path.exists(W2V_DIR):
        os.makedirs(W2V_DIR)

    data = load_data(num_samples=200, preprocessing=True)
    data = data['title'] + ' ' + data['content']
    create_w2v(data)
    print('Word2Vec created successfully.')


def load_data(num_samples=None, preprocessing=True):
    data_sources = [
        os.path.join(DATA_DIR, 'articles1.csv'),
        os.path.join(DATA_DIR, 'articles2.csv'),
        os.path.join(DATA_DIR, 'articles3.csv'),
    ]

    data = data_utils.create_news_data(data_sources, num_samples=num_samples, preprocessing=preprocessing)

    return data


def create_w2v(sentences):
    np.random.shuffle(sentences)
    sentences = [WhitespaceTokenizer().tokenize(s) for s in sentences]
    w2v_model = gensim.models.Word2Vec(
        sentences,
        size=300,
        min_count=1,
        iter=50)
    w2v_model.save(W2V_DIR + 'w2v_model_news' + '.pkl')


if __name__ == '__main__':
    main()

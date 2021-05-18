import itertools
from typing import DefaultDict

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    # 참고: 딥러닝 기반 자연어 언어모델 BERT
    # 참고: https://www.kaggle.com/naim99/text-classification-tf-idf-vs-word2vec-vs-bert
    # 참고: https://www.quora.com/What-is-the-difference-between-using-word2vec-vs-one-hot-embeddings-as-input-to-classifiers
    # 참고: https://medium.com/swlh/differences-between-word2vec-and-bert-c08a3326b5d1
    # 참고: https://ratsgo.github.io/from%20frequency%20to%20semantics/2017/03/11/embedding/
    # 참고: https://littlefoxdiary.tistory.com/42
    # 참고: http://doc.mindscale.kr/km/unstructured/11.html
    # 참고: https://wikidocs.net/21694

    # 문장을 숫자로.
    # 단어를 숫자로.
    #   단어를 숫자로 하려면 전체 단어 어휘에 대한 Numbering이 필요하다.
    #   One-Hot 인코딩 - 단어를 vector로 만들되, numbering에 대해서 1로 mark (Sparse 하다는 단점)
    #   Embedding - Sparse 하다는 단점을 극복하고자 Dense 하게 Vector를 만든다

    encoding_basic()
    encoding_basic_keras()

    # limit = 5
    # df = pd.DataFrame(columns=['sentense', 'words'])

    # dataset_file = 'nsmc/ratings_test.txt'
    # reader = file_lines_reader(dataset_file, limit=limit, skip=1)

    # for line in tqdm(reader):
    #     lines = line.rstrip().split('\t')
    #     if len(lines) == 3:
    #         df = df.append({'sentense': lines[1]}, ignore_index=True)

    # print('input.len', len(df))

    # df['words'] = df['sentense'].apply(lambda s: s.replace('.', ' '))
    # df['words'] = df['words'].apply(lambda s: s.replace('"', ' ').replace("'", ' '))
    # df['words'] = df['words'].apply(lambda s: s.replace(',', ' ').replace("~", ' '))
    # df['words'] = df['words'].apply(lambda s: s.split())

    # print(df.head())

    # vocabulary = set()
    # for idx, words in tqdm(df['words'].iteritems()):
    #     vocabulary.update(words)
    # vocabulary = sorted(vocabulary)
    # print(vocabulary[int(len(vocabulary)/2):int(len(vocabulary)/2 + 10)])

    # print('vocabulary.len', len(vocabulary))
    # print('어휘의 차원', len(vocabulary))

    # for idx, row in df.head().iterrows():
    #     for word in row['words']:
    #         vocab_index = vocabulary.index(word)
    #         print(idx, word, vocab_index)

    #         word_vector = one_hot_encoding(word, vocabulary)
    #         print(word_vector)

    # input_matrix = np.zeros((0, len(vocabulary)), dtype=int)
    # for idx, row in df.head().iterrows():
    #     print(row['words'])
    #     for word in row['words']:
    #         word_vector = one_hot_encoding(word, vocabulary)
    #         input_matrix = np.append(input_matrix, [word_vector], axis=0)
    #     break

    # print(input_matrix)


def encoding_basic():
    sentences = [
        '국경의 긴 터널을 빠져나오자, 설국이었다.',
        '밤의 아랫쪽이 하얘졌다.',
        '신호소에 기차가 멈춰 섰다.',
        '국경의 기차가 섰다.'
    ] # yapf: disable
    print('sentence', sentences)

    print('preprocessing')
    sentences = list(map(preprocess, sentences))
    print('sentence', len(sentences), sentences)

    print('tokenize')
    sentences = list(map(lambda it: it.split(), sentences))
    print('sentence', len(sentences), sentences)

    occurance = DefaultDict(int)
    for word in itertools.chain.from_iterable(sentences):
        occurance[word] += 1
    print('occurance', occurance)
    vocabulary = sorted([(v, k) for k, v in occurance.items()],
                        key=lambda it: it,
                        reverse=True)
    vocabulary = list(map(lambda it: it[1], vocabulary))
    print('vocabulary', len(vocabulary), vocabulary)

    for s in sentences:
        print(s, [vocabulary.index(word) for word in s])
        word_vec_list = [one_hot_encoding(word, vocabulary) for word in s]
        for word_vec in word_vec_list:
            print(' ', word_vec)


def encoding_basic_keras():
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical


    sentences = [
        '국경의 긴 터널을 빠져나오자, 설국이었다.',
        '밤의 아랫쪽이 하얘졌다.',
        '신호소에 기차가 멈춰 섰다.',
        '국경의 기차가 섰다.'
    ] # yapf: disable

    t = Tokenizer()
    t.fit_on_texts(sentences)
    print(t.word_index)


    for seq in t.texts_to_sequences(sentences):
        print(t.sequences_to_texts([seq])[0], seq)
        one_hot = to_categorical(seq)
        for word_vec in one_hot:
            print(' ', word_vec)

def preprocess(sentence):
    sentence = sentence.replace(',', ' ')
    sentence = sentence.replace('.', ' ')
    return sentence


def one_hot_encoding(word, vocabulary, dtype=int):
    word_vector = np.zeros(len(vocabulary), dtype=dtype)
    vocab_index = vocabulary.index(word)
    word_vector[vocab_index] = 1
    return word_vector


def file_lines_reader(filepath, limit=0, skip=0):
    unlimited = limit == 0
    with open(filepath) as f:
        while True:
            line = f.readline()
            if skip > 0:
                skip -= 1
                continue
            if not line:
                return
            if unlimited or (limit > 0):
                yield line
            else:
                return
            limit -= 1


if __name__ == '__main__':
    main()

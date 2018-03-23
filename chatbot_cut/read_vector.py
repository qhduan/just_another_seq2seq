"""
读取一个文本格式的，保存预训练好的embedding的文件

wiki.zh.vec

它的第一行会被忽略
第二行开始，每行是 词 + 空格 + 词向量维度0 + 空格 + 词向量维度1 + ...

参考fasttext的文本格式

https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
"""

import pickle
import numpy as np
from tqdm import tqdm


def read_vector(path='wiki.zh.vec', output_path='word_vec.pkl'):
    """
    读取文本文件 path 中的数据，并且生成一个 dict 写入到 output_path

    格式：
    word_vec = {
        'word_1': np.array(vec_of_word_1),
        'word_2': np.array(vec_of_word_2),
        ...
    }
    """
    fp = open(path, 'r')
    word_vec = {}
    first_skip = False
    dim = None
    for line in tqdm(fp):
        if not first_skip:
            first_skip = True
        else:
            line = line.strip()
            line = line.split(' ')
            if len(line) >= 2:
                word = line[0]
                vec_text = line[1:]
                vec = np.array([float(v) for v in vec_text])
                word_vec[word] = vec
                if dim is None:
                    dim = vec.shape

    # PAD_TAG = '<pad>'
    # UNK_TAG = '<unk>'
    # START_TAG = '<s>'
    # END_TAG = '</s>'

    np.random.seed(0)
    word_vec['<pad>'] = np.random.random(size=(300,)) - 0.5
    word_vec['<s>'] = np.random.random(size=(300,)) - 0.5
    word_vec['<unk>'] = np.random.random(size=(300,)) - 0.5

    pickle.dump(word_vec, open(output_path, 'wb'))

if __name__ == '__main__':
    read_vector()

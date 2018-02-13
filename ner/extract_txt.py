
"""
把tmx（xml）的数据解开，分词，然后保存到data.pkl
"""

# import re
import sys
import pickle

sys.path.append('..')

def read_txt(path):
    """读取一个txt文件的NER标注数据"""
    x_data, y_data = [], []
    x, y = [], []
    for line in open(path, 'r'):
        line = line.strip()
        line = line.split(' ')
        if len(line) == 2:
            x.append(line[0])
            y.append(line[1])
        else:
            if x and y:
                x_data.append(x)
                y_data.append(y)
            x, y = [], []
    return x_data, y_data

def main(limit=100):
    """执行程序
    Args:
        limit: 只输出句子长度小于limit的句子
    """
    from word_sequence import WordSequence

    x_data, y_data = [], []

    x, y = read_txt('train.txt')
    x_data += x
    y_data += y

    x, y = read_txt('validation.txt')
    x_data += x
    y_data += y

    x, y = read_txt('test.txt')
    x_data += x
    y_data += y

    print(len(x_data))

    print(x_data[:10])
    print(y_data[:10])

    print('tokenize')

    data = list(zip(x_data, y_data))
    data = [(x, y) for x, y in data if len(x) < limit and len(y) < limit]
    x_data, y_data = zip(*data)

    print(x_data[:10])
    print(y_data[:10])

    print(len(x_data), len(y_data))

    print('fit word_sequence')

    ws_input = WordSequence()
    ws_target = WordSequence()
    ws_input.fit(x_data, min_count=1)
    ws_target.fit(y_data, min_count=1)

    print('dump')

    pickle.dump(
        (x_data, y_data, ws_input, ws_target),
        open('ner.pkl', 'wb')
    )

    print('done')


if __name__ == '__main__':
    main()

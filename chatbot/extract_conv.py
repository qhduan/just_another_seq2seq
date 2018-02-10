"""把 dgk_shooter_min.conv 文件格式转换为可训练格式
"""

import re
import sys
import pickle
from tqdm import tqdm

sys.path.append('..')

def make_split(line):
    """构造合并两个句子之间的符号
    """
    if re.match(r'.*([，。…？！～\.,!?])$', ''.join(line)):
        return []
    return ['，']

def main(limit=15):
    """执行程序
    Args:
        limit: 只输出句子长度小于limit的句子
    """
    from word_sequence import WordSequence

    print('extract lines')
    fp = open('dgk_shooter_min.conv', 'r', errors='ignore')
    last_line = None
    groups = []
    group = []
    for line in tqdm(fp):
        if line.startswith('M '):
            line = line.replace('\n', '')
            line = line[2:].split('/')
            line = line[:-1]
            group.append(line)
        else: # if line.startswith('E'):
            last_line = None
            if group:
                groups.append(group)
                group = []
    if group:
        groups.append(group)
        group = []
    print('extract groups')
    x_data = []
    y_data = []
    for group in tqdm(groups):
        for i, line in enumerate(group):
            last_line = None
            if i > 0:
                last_line = group[i - 1]
            next_line = None
            if i < len(group) - 1:
                next_line = group[i + 1]
            next_next_line = None
            if i < len(group) - 2:
                next_next_line = group[i + 2]

            if next_line:
                x_data.append(line)
                y_data.append(next_line)
            if last_line and next_line:
                x_data.append(last_line + make_split(last_line) + line)
                y_data.append(next_line)
            if next_line and next_next_line:
                x_data.append(line)
                y_data.append(next_line + make_split(next_line) \
                    + next_next_line)

    print(len(x_data), len(y_data))
    for ask, answer in zip(x_data[:20], y_data[:20]):
        print(''.join(ask))
        print(''.join(answer))
        print('-' * 20)

    data = list(zip(x_data, y_data))
    data = [(x, y) for x, y in data if len(x) < limit and len(y) < limit]
    x_data, y_data = zip(*data)

    print('fit word_sequence')

    ws_input = WordSequence()
    ws_target = WordSequence()
    ws_input.fit(x_data)
    ws_target.fit(y_data)

    print('dump')

    pickle.dump(
        (x_data, y_data, ws_input, ws_target),
        open('chatbot.pkl', 'wb')
    )

    print('done')


if __name__ == '__main__':
    main()

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
    p1_data, q1_data, p2_data = [], [], []
    for group in tqdm(groups):
        for i, line in enumerate(group):
            last_line = None
            if i > 0:
                last_line = group[i - 1]
            next_line = None
            if i < len(group) - 1:
                next_line = group[i + 1]

            if last_line and next_line:
                p1_data.append(last_line)
                q1_data.append(line)
                p2_data.append(next_line)

    print(len(p1_data), len(q1_data), len(p2_data))
    for p1, q1, p2 in zip(p1_data[:5], q1_data[:5], p2_data[:5]):
        print(''.join(p1))
        print(''.join(q1))
        print(''.join(p2))
        print('-' * 20)

    data = list(zip(p1_data, q1_data, p2_data))
    data = [
        (p1, q1, p2)
        for p1, q1, p2 in data
        if len(p1) < limit and len(q1) < limit and len(p2) < limit
    ]
    p1_data, q1_data, p2_data = zip(*data)

    print('fit word_sequence')

    ws = WordSequence()
    ws.fit(p1_data + q1_data + p2_data)

    print('dump')

    pickle.dump(
        (p1_data, q1_data, p2_data, ws),
        open('chatbot_rl.pkl', 'wb')
    )

    print('done')


if __name__ == '__main__':
    main()

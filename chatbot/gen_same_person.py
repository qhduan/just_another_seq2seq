"""
判断两句话是不是连续上下文
"""

import sys
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle

sys.path.append('..')

def main():

    from word_sequence import WordSequence

    data, _, ws = pickle.load(open('chatbot.pkl', 'rb'))

    x1_data = []
    x2_data = []
    y_data = []

    for i, x in tqdm(enumerate(data), total=len(data)):
        for sign in ('，', '。', '；', '！', '？'):
            if sign in x:
                t = ''.join(x).split(sign)
                if len(t) == 2:
                    a, b = t
                    if len(a) >= 3 and len(b) >= 3:
                        x1_data.append(a)
                        x2_data.append(b)
                        y_data.append(1)
    print(len(x1_data))
    length = len(x1_data)

    for i in range(length):

        if np.random.random() > 0.5:
            a = x1_data[i]
        else:
            a = x2_data[i]

        j = np.random.randint(0, length)
        while j == i:
            j = np.random.randint(0, length)

        if np.random.random() > 0.5:
            b = x1_data[j]
        else:
            b = x2_data[j]

        x1_data.append(a)
        x2_data.append(b)
        y_data.append(0)

    # ws = WordSequence()
    # ws.fit(x1_data[:length])

    x1_data, x2_data, y_data = shuffle(x1_data, x2_data, y_data, random_state=0)

    pickle.dump((x1_data, x2_data, y_data, ws), open('same_person.pkl', 'wb'))

if __name__ == '__main__':
    main()


"""
把tmx（xml）的数据解开，分词，然后保存到data.pkl
"""

# import re
import sys
import pickle
import xml.etree.ElementTree as ET
import nltk
import jieba
from tqdm import tqdm

sys.path.append('..')

def main():
    """
    执行程序
    """
    from word_sequence import WordSequence

    x_data, y_data = [], []
    tree = ET.parse('en-zh_cn.tmx')
    root = tree.getroot()
    body = root.find('body')
    for tu in tqdm(body.findall('tu')):
        en = ''
        zh = ''
        for tuv in tu.findall('tuv'):
            if list(tuv.attrib.values())[0] == 'en':
                en += tuv.find('seg').text
            elif list(tuv.attrib.values())[0] == 'zh_cn':
                zh += tuv.find('seg').text

        if en and zh:
            x_data.append(en)
            y_data.append(zh)

    print(len(x_data))

    print(x_data[:10])
    print(y_data[:10])

    print('tokenize')

    def en_tokenize(text):
        # text = re.sub('[\(（][^\)）]+[\)）]', '', text)
        return nltk.word_tokenize(text.lower())

    x_data = [
        en_tokenize(x)
        for x in tqdm(x_data)
    ]

    def zh_tokenize(text):
        # text = text.replace('，', ',')
        # text = text.replace('。', '.')
        # text = text.replace('？', '?')
        # text = text.replace('！', '!')
        # text = text.replace('：', ':')
        # text = re.sub(r'[^\u4e00-\u9fff,\.\?\!…《》]:', '', text)
        # text = text.strip()
        text = jieba.lcut(text.lower())
        return text

    y_data = [
        zh_tokenize(y)
        for y in tqdm(y_data)
    ]

    data = list(zip(x_data, y_data))
    data = [(x, y) for x, y in data if len(x) < 15 and len(y) < 15]

    x_data, y_data = [x[0] for x in data], [x[1] for x in data]

    print(x_data[:10])
    print(y_data[:10])

    print(len(x_data), len(y_data))

    print('fit word_sequence')

    ws_input = WordSequence()
    ws_target = WordSequence()
    ws_input.fit(x_data)
    ws_target.fit(y_data)

    print('dump')

    pickle.dump(
        (x_data, y_data, ws_input, ws_target),
        open('en-zh_cn.pkl', 'wb')
    )

    print('done')


if __name__ == '__main__':
    main()

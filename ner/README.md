
# NER 测试

中文 命名实体识别(Named Entity Recognizer) 测试

## 1、下载数据

下载页面

https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset

下载链接：

wget "https://raw.githubusercontent.com/lancopku/Chinese-Literature-NER-RE-Dataset/master/ner/train.txt"

wget "https://raw.githubusercontent.com/lancopku/Chinese-Literature-NER-RE-Dataset/master/ner/test.txt"

wget "https://raw.githubusercontent.com/lancopku/Chinese-Literature-NER-RE-Dataset/master/ner/validation.txt"

## 2、预处理数据

运行 `extract_txt.py` 得到 `ner.pkl`

## 3、训练数据

#### seq2seq 模型

运行 `train.py` 训练（默认到`/tmp/s2ss_ner.ckpt`）

### 或者 crf 模型

运行 `train_crf.py` 训练（默认到`/tmp/s2ss_ner_crf.ckpt`）

## 4、测试数据（测试翻译）

#### seq2seq 模型

运行 `test.py` 查看测试结果

### 或者 crf 模型

运行 `test_crf.py` 查看测试结果

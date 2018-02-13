
*A lot of Chinese in codes and docs*

# Just another seq2seq repo

- [x] 主要是从个人角度梳理了一下seq2seq的代码
- [x] 加入了可选基本的CRF支持，loss和infer（还不确定对
- [x] 加入了一些中文注释
- [x] 相对于其他一些repo，bug可能会少一些
    - 有些repo的实现在不同参数下会有问题：例如有些支持gru不支持lstm，有些不支持bidirectional，有些选择depth > 1的时候会有各种bug之类的，这些问题我都尽量修正了，虽然不保证实现肯定是对的
- [x] 后续我可能会添加一些中文的例子，例如对联、古诗、闲聊、NER
- [ ] pretrained embedding support
- [ ] 根据本repo，我会整理一份seq2seq中间的各种trick和实现细节的坑
- [ ] 后续这个repo会作为一个基础完成一个dialog system
    - seq2seq模型至少可以作为通用NER实现（截止2018年初，最好的NER应该还是bi-LSTM + CRF）

# Known issues

Example里的例子和整个项目，虽然未经验证，但是在内存较小的电脑上（<8GB），可能会有问题。
这涉及到数据处理、数据输入、模型参数等部分，所以严格来说并不算BUG。

# Platform

作者在一台64GB内存 + GTX1070 6GB + Ubuntu 16.04电脑上运行。

内存肯定不需要这么大，不过显存如果在2GB，如果要在GPU上运行模型，可能需要调节batch_size等模型参数。

# Example

Example里面用到的数据，都是比较小且粗糙的。
作者只基本验证了可行性，所以也不可能实用了，例如英汉翻译就别期待准确率很高了，
大概意思到了就代表模型的一定有效性了。

[英汉句子翻译实例](/en2zh/)

#### 测试结果样例

我不保证能完全重复这个结果

```
Input English Sentence:go to hell
[[30475 71929 33464]] [3]
[[41337 48900 41337 44789     3]]
['go', 'to', 'hell']
['去', '地狱', '去', '吧', '</s>']
Input English Sentence:nothing, but the best for you
[[50448   467 13008 71007 10118 27982 79204]] [7]
[[ 25904 132783  90185      4  28145  81577  80498  28798      3]]
['nothing', ',', 'but', 'the', 'best', 'for', 'you']
['什么', '都', '没有', ' ', '但', '最好', '是', '你', '</s>']
Input English Sentence:i'm a bad boy
[[35437   268  4018  8498 11775]] [5]
[[ 69313  80498  21899  49069 100342      3     -1]]
['i', "'m", 'a', 'bad', 'boy']
['我', '是', '个', '坏', '男孩', '</s>', '<unk>']
Input English Sentence:i'm really a bad boy
[[35437   268 58417  4018  8498 11775]] [6]
[[ 69313 103249  80498  17043  49069 100342      3      3      3      3
       3      3]]
['i', "'m", 'really', 'a', 'bad', 'boy']
['我', '真的', '是', '一个', '坏', '男孩', '</s>', '</s>', '</s>', '</s>', '</s>', '</s>']
```

[Chatbot实例](/chatbot/)

[NER实例](/ner/)

# TensorFlow alert

test only tensorflow == 1.4.1

TensorFlow的API总是变，不能保证后续的更新兼容

# Related work

As mention in the head of `sequence_to_sequence.py`,
At beginning, the code is heavily borrow from [here](https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py)

I have modified a lot of code, some ***Chinese comments*** in the code.
And fix many bugs, restructure many things, add more features.

Code was borrow heavily from:

https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py

Another wonderful example is:

https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm

Official sequence2sequence tutorial

https://www.tensorflow.org/tutorials/seq2seq

Official sequence2sequence project:

https://github.com/tensorflow/nmt

Another official sequence2sequence model:

https://github.com/tensorflow/tensor2tensor

Another seq2seq repo:

https://github.com/ematvey/tensorflow-seq2seq-tutorials


# pylint

pylintrc from [here](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc)

changed indent from 2 to 4

PS. 谷歌的lint一般建议indent是2，相反百度的lint很多建议indent是4，
个人怀疑这里面有“中文”的问题，也许是因为从小习惯作文空两格？（就是四个英文空格了）

我个人是习惯4个的

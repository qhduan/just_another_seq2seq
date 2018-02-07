

# Just another seq2seq repo

*A lot of Chinese in codes and docs*

- [x] 主要是从个人角度梳理了一下seq2seq的代码
- [x] 加入了可选基本的CRF支持，loss和infer（还不确定对
- [x] 加入了一些中文注释
- [x] 相对于其他一些repo，bug可能会少一些
    - 有些repo的实现在不同参数下会有问题：例如有些支持gru不支持lstm，有些不支持bidirectional，有些选择depth > 1的时候会有各种bug之类的，这些问题我都尽量修正了，虽然不保证实现肯定是对的
- [ ] pretrained embedding support
- [ ] 根据本repo，我会整理一份seq2seq中间的各种trick和实现细节的坑
- [ ] 后续我可能会添加一些中文的例子，例如对联、古诗、闲聊、NER
- [ ] 后续这个repo会作为一个基础完成一个dialog system
    - seq2seq模型至少可以作为通用NER实现（截止2018年初，最好的NER应该还是bi-LSTM + CRF）

# TensorFlow alert

test only tensorflow == 1.4.1

TensorFlow的API总是变，不能保证后续的更新兼容

# Related work

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

Another:

https://github.com/ematvey/tensorflow-seq2seq-tutorials


# pylint

pylintrc from [here](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc)

changed indent from 2 to 4

PS. 谷歌的lint一般建议indent是2，相反百度的lint很多建议indent是4，
个人怀疑这里面有“中文”的问题，也许是因为从小习惯作文空两格？（就是四个英文空格了）

我个人是习惯4个的

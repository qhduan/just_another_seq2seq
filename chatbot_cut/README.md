
# Chatbot测试

## 1、下载数据

Subtitle data from [here](https://github.com/fateleak/dgk_lost_conv)

```
wget https://github.com/fateleak/dgk_lost_conv/raw/master/dgk_shooter_min.conv.zip
```

输出：`dgk_shooter_min.conv.zip`

## 2、解压缩

```
unzip dgk_shooter_min.conv.zip
```

输出：`dgk_shooter_min.conv`


## 3、下载训练好的fasttext的embedding

https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

注意是文本格式的

```
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.zh.vec
```

得到 `wiki.zh.vec` 文件

## 4、改变embedding格式

运行

```
python3 read_vector.py
```

得到 `word_vec.pkl`文件在目录下

## 5、预处理数据（前面两步embedding部分必须执行完）

```
python3 extract_conv.py
```

输出：`chatbot.pkl`

## 6、训练数据

运行 `python3 train.py` 训练（默认到`./s2ss_chatbot.ckpt`）

或者！

运行 `python3 train_anti.py` 训练抗语言模型（默认到`./s2ss_chatbot_anti.ckpt`）

## 7、测试数据（测试对话）

运行 `python3 test.py` 查看测试结果，需要提前训练普通模型

或者！

运行 `python3 test_anti.py` 查看抗语言模型的测试结果，需要提前训练抗语言模型

或者！

运行 `python3 test_compare.py` 查看普通模型和抗语言模型的对比测试结果，
需要提前训练两个模型

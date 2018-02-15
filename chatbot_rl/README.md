
# Chatbot强化学习训练测试

***注意`chatbot_re`的`extract_conv.py`和`chatbot`的不一样***

Paper: [Deep Reinforcement Learning for Dialogue Generation](https://arxiv.org/abs/1606.01541)

这个Paper应该修订版在[这里](https://aclweb.org/anthology/D16-1127)

Other Example:

https://github.com/Marsan-Ma/tf_chatbot_seq2seq_antilm/blob/master/lib/seq2seq_model.py

https://github.com/brianhuang1019/RL-Chatbot/blob/master/python/RL/train.py

## 1、下载数据

ssubtitle data from [here](https://github.com/fateleak/dgk_lost_conv)

```
wget https://github.com/fateleak/dgk_lost_conv/raw/master/dgk_shooter_min.conv.zip
```

输出：`dgk_shooter_min.conv.zip`

## 2、解压缩

```
unzip dgk_shooter_min.conv.zip
```

输出：`dgk_shooter_min.conv`

## 3、预处理数据

```
python3 extract_conv.py
```

输出：`chatbot.pkl`

## 4、训练数据

运行 `train.py` 训练（默认到`/tmp/s2ss_chatbot.ckpt`）

## 5、测试数据（测试翻译）

运行 `test.py` 查看测试结果


# Chatbot Adversarial

Paper: [Adversarial Learning for Neural Dialogue Generation](https://arxiv.org/abs/1701.06547)

[作者Jiwei Li的 repo](https://github.com/jiweil/Neural-Dialogue-Generation)

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

## 3、预处理数据

```
python3 extract_conv.py
```

输出：`chatbot.pkl`

## 4、训练数据

按顺序：

训练普通的模型

运行 `train_forward.py` 训练（默认到`./s2ss_chatbot_forward.ckpt`）

训练一个 discriminator

运行 `train_discriminative.py` 训练（默认到`./s2ss_chatbot_discriminative.ckpt`）

训练 Adversarial 模型

运行 `train_ad.py` 训练（默认到`./s2ss_chatbot_ad.ckpt`）

## 5、测试数据（测试对话）

运行 `test.py` 查看测试结果

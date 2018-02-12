
# 英汉翻译测试

## 1、下载数据

下载页面

http://opus.nlpl.eu/OpenSubtitles2018.php

下载链接：

wget -O "en-zh_cn.tmx.gz" "http://opus.nlpl.eu/download.php?f=OpenSubtitles2018/en-zh_cn.tmx.gz"

## 2、解压数据

这个数据是`英文-中文`的平行语聊

解压缩：

gunzip -k en-zh_cn.tmx.gz

下载并解压数据，然后重命名为 `en-zh_zh.tmx` （如果有有必要）

这应该是一个xml格式（在`linux`下可以用`head`命令查看下是否正确）

## 3、预处理数据

运行 `extract_tmx.py` 得到 `data.pkl`

## 4、训练数据

运行 `train.py` 训练（默认到`/tmp/s2ss_en2zh.ckpt`）

## 5、测试数据（测试翻译）

运行 `test.py` 查看测试结果

## 测试结果样例

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

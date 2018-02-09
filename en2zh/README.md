
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

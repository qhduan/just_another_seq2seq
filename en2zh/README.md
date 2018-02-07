
# 英汉翻译测试

下载页面

http://opus.nlpl.eu/OpenSubtitles2016.php

下载链接（不知道能不能直接用）：

http://opus.nlpl.eu/download.php?f=OpenSubtitles2016/en-zh_zh.tmx.gz

这个数据是`英文-中文`的平行语聊

下载并解压数据，然后重命名为 `en-zh_zh.tmx`

这应该是一个xml格式（在`linux`下可以用`head`命令查看下是否正确）

运行 `extract_tmx.py` 得到 `data.pkl`

运行 `train.py` 训练（默认到`/tmp/s2ss_en2zh`目录）

运行 `test.py` 查看测试结果

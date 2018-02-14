# Text Classification Framework

文本分类框架，可以完成:

 - 预处理。包括构建词表、label表，从预训练文件构建word embedding;
 - 训练。训练句子分类模型，模型包括CNNs等;
 - 测试。对新的句子进行标注。

另外，该框架还利用pytorch实现了高效的数据加载模块，包括以下特性:

 - 对于大数据量，不需要将其一次性加载到内存中，而是在需要的时候才进行读取;
 - 可以设置batch_size对数据进行批量读取;
 - 设置比例划分训练集和开发集;
 - 可以对整个数据集进行shuffle;
 - 多线程读取数据。

## 1. 预处理

### 1.1 预处理训练文件

训练文件的预处理包括:

 - 构建词表，即词->id的映射表，以及label表，以`dict`格式存放在`pkl`文件中;
 - 构建embedding表，根据所提供的预训练词向量文件，抽取所需要的向量，对于不存在于预训练文件中的词，则随机初始化。结果以`np.array`的格式存放在`pkl`文件中;
 - 将训练数据按顺序编号，每个实例写入单独的文件中，便于高效加载；
 - 统计句子长度，输出句子长度的[90, 95, 98, 100]百分位值;

**运行方式:**

    $ python3 preprocessing.py -l --pd ./data/train.txt --ri ./data/train_idx/ --rv ./res/voc/ --re ./res/embed/ --pe ./path_to_embed_file

### 1.2 预处理测试文件

**运行方式:**

    $ python3 preprocessing.py --pd ./data/test.txt --ri ./data/test_idx/

**表. 参数说明**

|参数|类型|默认值|说明|
| ------------ | ------------ | ------------ | ------------ |
|l|bool|False|label，是否带有标签(标志是否是训练集)|
|pd|str|./data/train.txt|path_data，训练(测试)数据路径|
|ri|str|./data/train_idx/|root_idx，训练数据索引文件根目录|
|rv|str|./res/voc/|root_voc，词表、label表根目录|
|re|str|./res/embed/|root_embed，embed文件根目录|
|pe|str|None|path_embed，预训练的embed文件路径，`bin`或`txt`；若不提供，则随机初始化|
|pt|int|98|percentile，构建词表时的百分位值|

也可以运行`python3 preprocessing.py -h`打印出帮助信息。

## 2. 训练

若预处理时`root_idx`等参数使用的是默认值，则在训练时不需要设定。

**运行方式:**

    $ CUDA_VISIBLE_DEVICES=0,1 python3 train.py --nc 2 --ml 40 --fs 3,4,5 --fn 400,300,200 --wd 64 --bs 256 -g

**参数说明**

|参数|类型|默认值|说明|
| ------------ | ------------ | ------------ | ------------ |
|ri|str|./data/train_idx/|root_idx，训练数据索引文件根目录|
|rv|str|./res/voc/|root_voc，词表、label表根目录|
|re|str|./res/embed/|root_embed，embed文件根目录|
|ml|int|50|max_len，句子最大长度|
|ds|float|0.2|dev_size，开发集占比|
|nc|int|无|nb_classes，分类类别数量|
|wd|int|50|word_dim，词向量维度|
|fs|str|2,3,4|filter_size，卷积核尺寸|
|fn|str|256,256,256|filter_num，卷积核大小|
|dp|float|0.5|dropout_rate，dropout rate|
|ne|int|100|nb_epoch，迭代次数|
|mp|int|5|max_patience，最大耐心值，即开发集上性能超过mp次没有提示，则终止训练|
|rm|str|./model/|root_model，模型根目录|
|bs|int|64|batch_size，batch size|
|g|bool|False|是否使用GPU加速|
|nw|int|4|num_worker，加载数据时的线程数|

也可以运行`python3 train.py -h`打印出帮助信息。

## 3. 测试

**运行方式:**

    $ CUDA_VISIBLE_DEVICES=0,1 python3 test.py --bs 256 -g --pr ./result.txt

|参数|类型|默认值|说明|
| ------------ | ------------ | ------------ | ------------ |
|ri|str|./data/train_idx/|root_idx，训练数据索引文件根目录|
|rv|str|./res/voc/|root_voc，词表、label表根目录|
|re|str|./res/embed/|root_embed，embed文件根目录|
|ml|int|50|max_len，句子最大长度|
|pm|str|无|path_model，模型路径|
|bs|int|64|batch_size，batch size|
|g|bool|False|是否使用GPU加速|
|nw|int|4|num_worker，加载数据时的线程数|
|pr|str|./result.txt|预测结果存放路径|

也可以运行`python3 test.py -h`打印出帮助信息。

## 4. Requirements

 - gensim==2.3.0
 - numpy==1.13.1
 - torch==0.2.0.post3
 - torchvision==0.1.9

from collections import Counter

import math
import numpy as np
import tensorflow as tf

STOP_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
# 句子最大长度
MAX_LEN = 64
# 最小词频
MIN_WORD_FREQUENCY = 8
# 训练的batch size
BATCH_SIZE = 16
# 数据集路径
DATASET_PATH = './poems.txt'
# 共训练多少个epoch
TRAIN_EPOCHS = 20


def get_dataset(file_path=DATASET_PATH):
    lines = []
    with open(file_path, 'rb') as fd:
        lines = fd.readlines()
        lines = [line.decode('utf-8').replace(":", ":").replace("，", ",").replace("。", ".") for line in lines]

    poems = []
    counter = Counter()
    for line in lines:
        if line.count(":") != 1:
            continue
        _, last_part = line.split(":")
        ignore_flag = 0
        for word in last_part:
            if word in STOP_WORDS:
                ignore_flag = 1
                break

        if ignore_flag == 1:
            continue

        if len(last_part) > MAX_LEN - 2:
            continue

        line = line.replace("\n", "")
        counter.update(line)
        poems.append(line)

    _tokens_count = sorted([(token, count) for token, count in counter.items() if count > MIN_WORD_FREQUENCY],
                           key=lambda x: x[-1])

    _tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + [token for token, count in _tokens_count]

    _tokens_dict = dict(zip(_tokens, range(len(_tokens))))

    text_class = TextClass(_tokens_dict)

    np.random.shuffle(poems)

    return text_class, poems


class TextClass:
    def __init__(self, word_dict):
        """
        this function is build a TextTokenizer
        :param dict:
        """
        self.word_dict = word_dict
        self.token_dict = {value: key for key, value in self.word_dict.items()}
        self.vocab_size = len(self.token_dict)

    def id_to_token(self, token_id):
        """
        this function is use id to get word
        :param token_id:
        :return:
        """
        return self.token_dict[token_id]

    def token_to_id(self, token):
        """
        this function is use token to get its words
        :param token:
        :return:
        """
        return self.word_dict.get(token, self.word_dict['[UNK]'])

    def encode(self, tokens):
        res = [self.token_to_id('[CLS]')]
        for token in tokens:
            res.append(self.token_to_id(token))
        res.append(self.token_to_id('[SEP]'))
        return res

    def decode(self, token_ids):
        stop_word = ['[CLS]', '[SEP]']
        res = []
        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if token in stop_word:
                continue
            res.append(token)
        return ''.join(res)


tokenizer, poems = get_dataset('./poems.txt')


class PoetryDataGenerator:
    """
    古诗数据集生成器
    """

    def __init__(self, data, random=False):
        # 数据集
        self.data = data
        # batch size
        self.batch_size = BATCH_SIZE
        # 每个epoch迭代的步数
        self.steps = int(math.floor(len(self.data) / self.batch_size))
        # 每个epoch开始时是否随机混洗
        self.random = random


    def sequence_padding(self, data, length=None, padding=None):
        """
        将给定数据填充到相同长度
        :param data: 待填充数据
        :param length: 填充后的长度，不传递此参数则使用data中的最大长度
        :param padding: 用于填充的数据，不传递此参数则使用[PAD]的对应编号
        :return: 填充后的数据
        """
        # 计算填充长度
        if length is None:
            length = max(map(len, data))
        # 计算填充数据
        if padding is None:
            padding = tokenizer.token_to_id('[PAD]')
        # 开始填充
        outputs = []
        for line in data:
            padding_length = length - len(line)
            # 不足就进行填充
            if padding_length > 0:
                outputs.append(np.concatenate([line, [padding] * padding_length]))
            # 超过就进行截断
            else:
                outputs.append(line[:length])
        return np.array(outputs)

    def __len__(self):
        return self.steps

    def __iter__(self):
        total = len(self.data)
        # 是否随机混洗
        if self.random:
            np.random.shuffle(self.data)
        # 迭代一个epoch，每次yield一个batch
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_data = []
            # 逐一对古诗进行编码
            for single_data in self.data[start:end]:
                batch_data.append(tokenizer.encode(single_data))
            # 填充为相同长度
            batch_data = self.sequence_padding(batch_data)
            # yield x,y
            yield batch_data[:, :-1], tf.one_hot(batch_data[:, 1:], tokenizer.vocab_size)
            del batch_data

    def for_fit(self):
        """
        创建一个生成器，用于训练
        """
        # 死循环，当数据训练一个epoch之后，重新迭代数据
        while True:
            # 委托生成器
            yield from self.__iter__()


def build_model(text_class):
    model = tf.keras.Sequential([
        tf.keras.layers.Input((None,)),
        tf.keras.layers.Embedding(input_dim=text_class.vocab_size, output_dim=128),
        tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),
        tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(text_class.vocab_size, activation='softmax'))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy)

    return model


class Evaluate(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        # 给loss赋一个较大的初始值
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 在每个epoch训练完成后调用
        # 如果当前loss更低，就保存当前模型参数
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
        # 随机生成几首古体诗测试，查看训练效果
        print()
        for i in range(5):
            print(generate_random_poetry(tokenizer, model))


def generate_random_poetry(tokenizer, model, s=''):
    """
    随机生成一首诗
    :param tokenizer: 分词器
    :param model: 用于生成古诗的模型
    :param s: 用于生成古诗的起始字符串，默认为空串
    :return: 一个字符串，表示一首古诗
    """
    # 将初始字符串转成token
    token_ids = tokenizer.encode(s)
    # 去掉结束标记[SEP]
    token_ids = token_ids[:-1]
    while len(token_ids) < MAX_LEN:
        # 进行预测，只保留第一个样例（我们输入的样例数只有1）的、最后一个token的预测的、不包含[PAD][UNK][CLS]的概率分布
        _probas = model.predict([token_ids, ])[0, -1, 3:]
        # print(_probas)
        # 按照出现概率，对所有token倒序排列
        p_args = _probas.argsort()[::-1][:100]
        # 排列后的概率顺序
        p = _probas[p_args]
        # 先对概率归一
        p = p / sum(p)
        # 再按照预测出的概率，随机选择一个词作为预测结果
        target_index = np.random.choice(len(p), p=p)
        target = p_args[target_index] + 3
        # 保存
        token_ids.append(target)
        if target == 3:
            break
    return tokenizer.decode(token_ids)

generation = PoetryDataGenerator(data=poems, random=True)

model = build_model(tokenizer)

    # 创建数据集
data_generator = PoetryDataGenerator(poems, random=True)
    # 开始训练
model.fit_generator(data_generator.for_fit(), steps_per_epoch=data_generator.steps, epochs=TRAIN_EPOCHS,
                        callbacks=[Evaluate()])

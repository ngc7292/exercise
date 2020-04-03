from __future__ import division
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import time
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 超参数设置
def add_arguments(parser):
    parser.add_argument('--train_file', default='oliver.txt')
    parser.add_argument('--seq_size', type=int, default=32, help='sequence length')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--embedding_size', type=int, default=128, help='embedding hidden size')
    parser.add_argument('--lstm_size', type=int, default=128, help='lstm hidden size')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.7, help='lstm dropout keep probability')
    parser.add_argument('--gradients_norm', type=int, default=5, help='norm to clip gradients')
    parser.add_argument('--initial_words', type=list, default=['I', 'am'],
                        help='Initial words to start prediction from')
    parser.add_argument('--predict_top_k', type=int, default=5, help='top k results to sample word from')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--checkpoint_path', default='checkpoint', help='directory to store trained weights')
    parser.add_argument('--buffer_size', type=int, default=100, help='buffer size to use')


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()


# 加载数据
def get_data_from_file(train_file, batch_size, seq_size):
    with open(train_file, encoding='utf-8') as f:
        text = f.read()
    
    text = text.split()
    
    # 获取频繁词
    word_counts = Counter(text)
    # 按key降序排列
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # 创建词汇表
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    # 词汇表大小
    n_vocab = len(int_to_vocab)
    
    print('Vocabulary size', n_vocab)
    
    # 输入文本
    int_text = [vocab_to_int[w] for w in text]
    # 训练总需的批次数
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    
    # 后移
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    
    in_text = np.reshape(in_text, (-1, seq_size))
    out_text = np.reshape(out_text, (-1, seq_size))
    
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text


# 构建模型，这里使用的是单层的LSTM
class RNNModule(tf.keras.Model):
    def __init__(self, n_vocab, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.lstm_size = lstm_size
        
        # embedding shape:n_vocab * embedding_size
        self.embedding = tf.keras.layers.Embedding(n_vocab, embedding_size)
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_state=True, return_sequences=True)
        self.dense = tf.keras.layers.Dense(n_vocab)
    
    # x shape: batch_size * seq_size
    def call(self, x, prev_state):
        embed = self.embedding(x)
        output, state_h, state_c = self.lstm(embed, prev_state)
        
        logits = self.dense(output)
        preds = tf.nn.softmax(logits)
        
        return logits, preds, (state_h, state_c)
    
    # 状态初始化
    def zero_state(self, batch_size):
        return [tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size])]


# chekpoint设置
checkpoint_dir = './training_chekpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)


# 预测
def predict(model, vocab_to_int, int_to_vocab, n_vocab):
    # 根据概率选择新的词
    def get_word(int_pred, n_vocab):
        p = np.squeeze(int_pred)
        p[p.argsort()][:-5] = 0
        p = p / np.sum(p)
        word = np.random.choice(n_vocab, 1, p=p)[0]
        
        return word
    
    val_state = model.zero_state(1)
    words = args.initial_words
    # 生成文本
    for word in words:
        int_word = tf.convert_to_tensor(
            [[vocab_to_int[word]]], dtype=tf.float32)
        _, int_pred, val_state = model(int_word, val_state)
    int_pred = int_pred.numpy()
    int_word = get_word(int_pred, n_vocab)
    words.append(int_to_vocab[int_word])
    # 保证生成的文本长度为50
    for _ in range(50):
        int_word = tf.convert_to_tensor(
            [[int_word]], dtype=tf.float32)
        _, int_pred, val_state = model(int_word, val_state)
        int_pred = int_pred.numpy()
        int_word = get_word(int_pred, n_vocab)
        words.append(int_to_vocab[int_word])
    
    print('Generated sentences are: {}'.format(' '.join(words)))


@tf.function
def train_func(inputs, targets, model, state, loss_func, optimizer):
    with tf.GradientTape() as tape:
        logits, _, state = model(inputs, state)
        
        loss = loss_func(targets, logits)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss


# 训练
def train():
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(args.train_file, args.batch_size,
                                                                                args.seq_size)
    # 文本长度
    len_data = in_text.shape[0]
    # 每个epoch训练的步数
    steps_per_epoch = len_data // args.batch_size
    
    # 创建dataset对象
    dataset = tf.data.Dataset.from_tensor_slices((in_text, out_text)).shuffle(args.buffer_size)
    dataset = dataset.batch(16, drop_remainder=True)
    
    # 加载模型、Encoder状态初始化
    model = RNNModule(n_vocab, args.embedding_size, args.lstm_size)
    state = model.zero_state(args.batch_size)
    
    # 选择优化器和损失项
    optimizer = tf.keras.optimizers.Adam()
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # 开始
    for epoch in range(args.num_epochs):
        start = time.time()
        total_loss = list()
        state = model.zero_state(args.batch_size)
        
        for (batch, (inputs, targets)) in enumerate(dataset.take(steps_per_epoch)):
            
            # print ('Inputs shape: {}'.format(inputs.shape))
            # print ('targets shape: {}'.format(targets.shape))
            # Inputs shape: (16, 32)
            # targets shape: (16, 32)
            
            loss = train_func(inputs, targets, model, state, loss_func, optimizer)
            total_loss.append(loss)
            
            if batch % 100 == 0:
                print('Epoch: {}/{}'.format(epoch, args.num_epochs),
                      'Batch--> {}'.format(batch),
                      'Loss--> {}'.format(loss.numpy()))
            
            if batch % 300 == 0:
                predict(model, vocab_to_int, int_to_vocab, n_vocab)
                model.save_weights(checkpoint_prefix.format(epoch=epoch))
        
        plot_loss(total_loss)
        print('Total time of this epoch is {}:'.format(time.time() - start))


# 绘制损失变化图
def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    train()
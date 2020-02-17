import numpy as np
import tensorflow as tf
import collections
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, optimizers, datasets

start_token = 'bos'
end_token = 'eos'


def process_dataset(fileName):
    examples = []
    with open(fileName, 'r',encoding='utf-8') as fd:
        for line in fd:
            outs = line.strip().split(':')
            content = ''.join(outs[1:])
            ins = [start_token] + list(content) + [end_token]
            if len(ins) > 200:
                continue
            examples.append(ins)
    
    counter = collections.Counter()
    for e in examples:
        for w in e:
            counter[w] += 1
    
    sorted_counter = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*sorted_counter)
    words = ('PAD', 'UNK') + words[:len(words)]
    word2id = dict(zip(words, range(len(words))))
    id2word = {word2id[k]: k for k in word2id}
    
    indexed_examples = [[word2id[w] for w in poem]
                        for poem in examples]
    seqlen = [len(e) for e in indexed_examples]
    
    instances = list(zip(indexed_examples, seqlen))
    
    return instances, word2id, id2word


def poem_dataset():
    instances, word2id, id2word = process_dataset('poems.txt')
    ds = tf.data.Dataset.from_generator(lambda: [ins for ins in instances],
                                        (tf.int64, tf.int64),
                                        (tf.TensorShape([None]), tf.TensorShape([])))
    ds = ds.shuffle(buffer_size=10240)
    ds = ds.padded_batch(100, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
    ds = ds.map(lambda x, seqlen: (x[:, :-1], x[:, 1:], seqlen - 1))
    return ds, word2id, id2word

if __name__ == '__main__':
    train_ds, word2id, id2word = poem_dataset()

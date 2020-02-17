import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


def get_mnist_data():
    """
    get mnist data from tf.keras.datasets
    :return:
    """
    (train_data, train_label), (test_data, test_label) = datasets.mnist.load_data()
    print(train_data.shape)
    print(test_data.shape)
    train_data = train_data.reshape((60000, 28, 28, 1))
    test_data = test_data.reshape((10000, 28, 28, 1))
    return train_data, train_label, test_data, test_label


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (7, 7), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    print(model.summary())
    
    return model


if __name__ == '__main__':
    train_data, train_label, test_data, test_label = get_mnist_data()
    
    model = build_model()
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_data,train_label,batch_size=100,epochs=5)

    test_loss, test_acc = model.evaluate(test_data, test_label)
    
    print(test_acc)
    
    
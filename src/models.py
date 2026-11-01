"""Model architectures (Keras), in the style of the practical classes (P03)."""

import keras
from keras import layers, regularizers

from .config import CLASSES


def build_cnn(input_shape=(32, 251, 1), filters=(16, 32, 64, 128), dropout=0.3, l2=1e-4):
    """2D CNN on log-mel spectrograms (mel bands x time frames x 1 channel).

    Each block: Conv2D 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2, doubling the number of feature maps
    while halving both axes. GlobalAveragePooling2D replaces Flatten + Dense, which keeps the network
    small (~100k parameters, adequate for ~750 training patients) and makes the output less
    sensitive to *where* in the window a heart sound or murmur occurs."""
    model = keras.Sequential(name="cnn")
    model.add(keras.Input(shape=input_shape))
    for f in filters:
        model.add(layers.Conv2D(f, kernel_size=3, padding="same", use_bias=False,
                                kernel_regularizer=regularizers.l2(l2)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(len(CLASSES), activation="softmax"))
    return model


def build_rnn(input_shape=(126, 39), units=64, dropout=0.3):
    """Bidirectional GRU on sequences of MFCC vectors (time steps x coefficients).

    GRU instead of a vanilla RNN to avoid vanishing gradients over 126 time steps (4 s); bidirectional
    because the whole window is available, and the context after a sound (e.g. S2 after a systolic
    murmur) is as informative as the context before it. The first layer returns the whole sequence
    so the second one can build on it; the second returns only its final state, a summary of the window."""
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Bidirectional(layers.GRU(units, return_sequences=True)),
        layers.Bidirectional(layers.GRU(units)),
        layers.Dropout(dropout),
        layers.Dense(len(CLASSES), activation="softmax"),
    ], name="rnn")

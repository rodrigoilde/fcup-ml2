"""Model definitions (Keras)."""
import keras
from keras import layers

from . import config


def build_cnn1d(win=config.WIN, n_channels=len(config.BANDS), blocks=((16, 4), (32, 4), (64, 4), (128, 2), (128, 2)),
                kernel=9, dropout=0.5, n_classes=3):
    """1D CNN on the multi-band waveform. Each block: Conv1D -> BatchNorm -> ReLU -> MaxPool.
    With the default blocks the receptive field of the last conv layer is ~1 s, i.e. about one cardiac cycle."""
    inp = keras.Input(shape=(win, n_channels), name="waveform")
    x = inp
    for filters, pool in blocks:
        x = layers.Conv1D(filters, kernel, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(pool)(x)
    x = layers.GlobalAveragePooling1D()(x)          # murmurs repeat in every beat: average over time
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="murmur")(x)
    return keras.Model(inp, out, name="cnn1d")


def receptive_field(blocks=((16, 4), (32, 4), (64, 4), (128, 2), (128, 2)), kernel=9):
    """Receptive field (in samples) of the last conv layer of build_cnn1d."""
    rf, jump = 1, 1
    for i, (_, pool) in enumerate(blocks):
        rf += (kernel - 1) * jump
        if i < len(blocks) - 1:
            rf += (pool - 1) * jump
            jump *= pool
    return rf


def build_bigru_mt(n_mels=config.N_MELS, units=64, n_layers=2, dropout=0.3, n_states=len(config.STATES),
                   n_classes=3):
    """Bidirectional GRU with two heads:
    - "seg":    frame-wise heart-sound state (auxiliary task, trained with the .tsv annotations);
    - "murmur": recording-level class, from attention pooling over the frames.
    The .tsv files are only needed for training: at inference only the "murmur" output is used."""
    inp = keras.Input(shape=(None, n_mels), name="logmel")
    x = inp
    for _ in range(n_layers):
        x = layers.Bidirectional(layers.GRU(units, return_sequences=True, dropout=0.1))(x)
    seg = layers.Dense(n_states, activation="softmax", name="seg")(x)

    att = layers.Dense(1)(x)                          # one relevance score per frame
    att = layers.Softmax(axis=1, name="attention")(att)
    pooled = layers.Flatten()(layers.Dot(axes=1)([att, x]))   # weighted sum of the frames
    h = layers.Dropout(dropout)(pooled)
    h = layers.Dense(32, activation="relu")(h)
    murmur = layers.Dense(n_classes, activation="softmax", name="murmur")(h)
    return keras.Model(inp, {"murmur": murmur, "seg": seg}, name="bigru_mt")

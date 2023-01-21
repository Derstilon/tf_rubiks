from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Dense, Reshape, Flatten

def build_basic_scrambler():
    model = Sequential()

    model.add(InputLayer(input_shape=(6, 3, 3,)))
    model.add(Flatten())
    # TODO: add smarter architecture
    model.add(Dense(128, activation='relu'))

    # Output layer
    model.add(Dense(13))

    return model

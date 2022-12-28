from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Dense, Reshape, Flatten

def build_basic_scrambler(n=6):
    model = Sequential()

    model.add(InputLayer((10)))

    # TODO: add smarter architecture
    model.add(Dense(128, activation='relu'))

    # Output layer
    model.add(Dense(13*n))
    model.add(Reshape((13, n)))

    return model

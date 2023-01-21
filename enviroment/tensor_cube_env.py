from enviroment import cube
from random import randint
import tensorflow as tf


cube_move_dict = {
    0: 'f',
    1: 'r',
    2: 'l',
    3: 'u',
    4: 'd',
    5: 'b',
    6: '.f',
    7: '.r',
    8: '.l',
    9: '.u',
    10: '.d',
    11: '.b',
    12: '',
}

tile_color_dict = {
    'R': 0,
    'O': 1,
    'Y': 2,
    'G': 3,
    'B': 4,
    'W': 5,
}


class TensorCubeEnv():
    def __init__(self):
        self.cube = None
        self.orderNum = 3
        self.reset()
        self.solved_tensor = tf.constant([[[i, i, i] for _ in range(self.orderNum)] for i in [5, 2, 1, 0, 3, 4]])

    def reset(self):
        self.cube = cube.Cube(order=self.orderNum)

    def get_tensor(self):
        values = [tile_color_dict[i] for i in self.cube.constructVectorState()]
        # reshape to 3x3x6 tensor
        return tf.reshape(values, (1, 6, self.orderNum, self.orderNum))

    def get_loss(self):
        tensor = self.get_tensor()
        # count the number of tiles that are different from the solved state
        values = tf.math.count_nonzero(tf.math.not_equal(tensor, self.solved_tensor)).numpy()
        # normalize the loss
        return tf.constant(values / (6 * (self.orderNum ** 2 - 1)))

    def render(self, isColor=True):
        self.cube.displayCube(isColor=isColor)

    def perform_move(self, move_tensor):
        move = tf.math.argmax(move_tensor).numpy()

        self.cube.minimalInterpreter(cube_move_dict[move])

    def get_random_move(self):
        return tf.one_hot(randint(0, 11), 13, dtype=tf.dtypes.float32)

    def get_oposite_move(self, move_tensor):
        # use tensor operations to get the opposite move
        move = tf.math.argmax(move_tensor).numpy()
        if(move < 6):
            move += 6
        elif(move < 12):
            move -= 6
        # generate tensor of opposite move
        oposite_tensor = tf.one_hot(move, 13, dtype=tf.dtypes.float32)

        return tf.reshape(oposite_tensor, [1,13])
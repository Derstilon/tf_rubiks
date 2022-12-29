import numpy
from tensorflow.python.keras.models import Model
import tensorflow as tf
from enviroment.tensor_cube_env import TensorCubeEnv
from solvers import build_basic_solver
from scramblers import build_basic_scrambler
import numpy as np
class CubeGAN(Model):
    def __init__(self, scrambler, solver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_loss = None
        self.g_loss = None
        self.d_opt = None
        self.g_opt = None
        self.scrambler = scrambler
        self.solver = solver
        self.cube_env = TensorCubeEnv()

    def get_moves(self, moves):
        result = []
        for move in moves.numpy()[0]:
            row_max = [index for index, item in enumerate(move) if item == max(move)]
            result.append(row_max[0])
        return result

    def train_step(self):

        self.cube_env.reset()

        moves = self.scrambler(np.random.rand(1, 10))
        self.cube_env.performMoves(self.get_moves(moves))

        tensor = self.cube_env.getTensor()
        tensor = tf.reshape(tensor, (6, 3, 3))

        # TODO ustalic shape'y
        print(tensor)

        result = self.solver(tensor) # TODO jaki result

        # w zaleznosci od resulta albo

        if result:
            total_loss = self.cube_env.getLoss()
        else:
            self.cube_env.performMoves(moves)
            total_loss = self.cube_env.getLoss()

        # jeden model dostaje kare na podstawie lossa

        # drugi dostaje nagrode

        # Apply backpropagation
        scrambler_gradient = tf.GradientTape().gradient(total_loss,self.scrambler.trainable_variables)
        solver_gradient = tf.GradientTape().gradient(1 - total_loss, self.solver.trainable_variables)
        opt = tf.keras.optimizers.experimental.SGD(learning_rate=0.1)
        opt.apply_gradients(zip(scrambler_gradient, self.scrambler.trainable_variables))
        opt.apply_gradients(zip(solver_gradient, self.solver.trainable_variables))

        return {"scrambler_loss": total_loss, "solver_loss": 1 - total_loss}



if __name__ == "__main__":

    solver = build_basic_solver()
    scrambler = build_basic_scrambler()
    cubeGAN = CubeGAN(scrambler, solver)

    cubeGAN.train_step()
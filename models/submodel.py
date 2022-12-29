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

    def train_step(self):

        self.cube_env.reset()
        with tf.GradientTape() as scrambler_tape:
            random_moves = self.scrambler(np.random.rand(1, 10))
            self.cube_env.performMoves(tf.math.argmax(random_moves, 2).numpy()[0])

        tensor = self.cube_env.getTensor()
        tensor = tf.reshape(tensor, (1, 6, 3, 3))

        with tf.GradientTape() as solver_tape:
            solve_moves = self.solver(tensor)
            self.cube_env.performMoves(tf.math.argmax(solve_moves, 2).numpy()[0])

        total_loss = self.cube_env.getLoss()

        # Apply backpropagation
        scrambler_gradient = scrambler_tape.gradient(1 - total_loss, self.scrambler.trainable_variables)
        solver_gradient = solver_tape.gradient(total_loss, self.solver.trainable_variables)

        opt = tf.keras.optimizers.experimental.SGD(learning_rate=0.1)
        opt.apply_gradients(zip(scrambler_gradient, self.scrambler.trainable_variables))
        opt.apply_gradients(zip(solver_gradient, self.solver.trainable_variables))

        return {"scrambler_loss": 1 - total_loss, "solver_loss": total_loss}



if __name__ == "__main__":

    solver = build_basic_solver()
    scrambler = build_basic_scrambler()
    cubeGAN = CubeGAN(scrambler, solver)
    cubeGAN.train_step()
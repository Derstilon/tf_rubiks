from tensorflow.python.keras.models import Model
import tensorflow as tf
from enviroment.tensor_cube_env import TensorCubeEnv
from models.solvers import build_basic_solver
from models.scramblers import build_basic_scrambler
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
        
    def print(self, loss):
        scrambler_loss = 1 - loss
        solver_loss = loss
        message = f'scrambler_loss: {scrambler_loss}, solver_loss: {solver_loss}'
        print(message)
        return message
    
    def train_step(self, DEBUG=False):

        self.cube_env.reset()
        with tf.GradientTape() as scrambler_tape:
            random_moves = self.scrambler(np.random.rand(1, 10))
            self.cube_env.performMoves(random_moves)

        tensor = self.cube_env.getTensor()

        with tf.GradientTape() as solver_tape:
            solve_moves = self.solver(tensor)
            self.cube_env.performMoves(random_moves)

        total_loss = self.cube_env.getLoss()

        # Apply backpropagation
        scrambler_gradient = scrambler_tape.gradient(1 - total_loss, self.scrambler.trainable_variables)
        solver_gradient = solver_tape.gradient(total_loss, self.solver.trainable_variables)

        opt = tf.keras.optimizers.experimental.SGD(learning_rate=0.1)
        try:
            opt.apply_gradients(zip(scrambler_gradient, self.scrambler.trainable_variables))
            opt.apply_gradients(zip(solver_gradient, self.solver.trainable_variables))
        except(Exception):
            if DEBUG:
                print("Gradient error, please fix this. @Petros9")
        if DEBUG:
            self.print(total_loss.numpy())
        return (total_loss.numpy(), 1- total_loss.numpy())

def build_basic_cubeGAN():
    solver = build_basic_solver()
    scrambler = build_basic_scrambler()
    cubeGAN = CubeGAN(scrambler, solver)
    return cubeGAN

if __name__ == "__main__":

    solver = build_basic_solver()
    scrambler = build_basic_scrambler()
    cubeGAN = CubeGAN(scrambler, solver)
    for i in range(100):
        cubeGAN.train_step(DEBUG=True)
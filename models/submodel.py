from tensorflow.python.keras.models import Model
import tensorflow as tf
from enviroment.tensor_cube_env import TensorCubeEnv
from models.solvers import build_basic_solver
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizers import adam_v2
import numpy as np
from random import random
from progress.bar import Bar
from matplotlib import pyplot as plt


class CubeGAN(Model):
    def __init__(self, solver, learning_rate=0.00001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = BinaryCrossentropy(from_logits=True)
        self.opt = adam_v2.Adam(learning_rate=learning_rate)
        self.solver = solver
        self.cube_env = TensorCubeEnv()
        self.cube_env.reset()

    def print(self, loss): # ????
        solver_loss = loss
        message = f'solver_loss: {solver_loss}'
        print(message)
        return message

    def train_step(self, DEBUG=False):
        random_move = self.cube_env.get_random_move()

        # Randomly reset the env. The cube will get scrambled by the training steps and 
        # sometimes randomly reset to expose the model to low scrable scenarios.
        if random() > 0.95:
            self.cube_env.reset()

            # Sometimes skip the move to teach the solver that the cube is solved.
            if random() > 0.91:
                # No move
                random_move = tf.one_hot(12, 13, dtype=tf.dtypes.float32)

        
        self.cube_env.perform_move(random_move)
        expected_move = self.cube_env.get_oposite_move(random_move)

        cube_tensor = self.cube_env.get_tensor()

        with tf.GradientTape() as solver_tape:
            solve_move = self.solver(cube_tensor)

            solver_loss = self.loss(expected_move, solve_move)

        # Apply backpropagation
        solver_gradient = solver_tape.gradient(solver_loss, self.solver.trainable_variables)
        self.opt.apply_gradients(zip(solver_gradient, self.solver.trainable_variables))

        return solver_loss.numpy(), tf.argmax(expected_move, axis=1).numpy()[0] == tf.argmax(solve_move, axis=1).numpy()[0]



if __name__ == "__main__":
    EPOCHS = 20000
    PRINT_N = 100

    bar = Bar('Training', max=EPOCHS)

    solver = build_basic_solver()
    cubeGAN = CubeGAN(solver)

    losses = []

    for i in range(EPOCHS):
        loss, accurate_pred = cubeGAN.train_step(DEBUG=True)
        if i%PRINT_N == 0:
            losses.append(loss)
        bar.next()

    plt.plot(losses)
    plt.grid()
    plt.xlabel(f'Epoch (every {PRINT_N}th)')
    plt.ylabel('Loss')

    plt.savefig('result.png')

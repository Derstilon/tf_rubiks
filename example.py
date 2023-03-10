from enviroment.tensor_cube_env import TensorCubeEnv

env = TensorCubeEnv()
print(env.get_tensor())
print(env.get_loss())
env.render()
env.performMoves([0, 2, 7, 12, 12, 1, 6, 8, 9, 10, 11, 5])
print(env.get_tensor())
print(env.get_loss())
env.render()

# from models.solvers import build_basic_solver
# from models.scramblers import build_basic_scrambler

# solver = build_basic_solver()

# solver.summary()

# scrambler = build_basic_scrambler()

# scrambler.summary()

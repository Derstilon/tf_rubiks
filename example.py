from enviroment.tensor_cube_env import TensorCubeEnv

env = TensorCubeEnv()
print(env.getTensor())
print(env.getLoss())
env.render()
env.performMoves([0, 2, 7, 12, 12, 1, 6, 8, 9, 10, 11, 5])
print(env.getTensor())
print(env.getLoss())
env.render()

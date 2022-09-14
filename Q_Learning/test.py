from string import printable
import numpy as np

def rand_Pair():
    row = np.random.randint(0, 4)
    col = np.random.randint(0, 4)

    return row, col

np.random.seed(1)
size_env = 5
array = np.random.randint(low=1, high=100, size=size_env)
print(array)

pieces = []
for i in range(5):
    np.random.seed(array[i])
    pos = rand_Pair()
    pieces.append(pos)

print(pieces)
    
import sys

import deepmimo

print("python:", sys.executable)
print("sys.path[0]:", sys.path[0])
print("deepmimo:", deepmimo.__file__)
print("has download:", hasattr(deepmimo, "download"))

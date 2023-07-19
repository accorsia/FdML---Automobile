import os
import random

import numpy as np




if __name__ == "__main__":

    a = random.randint(0, 100)
    b = random.randint(0, 100)

    serialize("spz/tester.npz", a=a, b=b)

    data = np.load('spz/tester.npz')
    a2 = data['a']
    b2 = data['b']

    print(f"a = {a}")
    print(f"b = {b}")
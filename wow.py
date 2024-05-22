import numpy as np
import random

num1 = random.randrange(1,11)
num2 = random.randrange(1,11)

a = np.zeros(1, dtype=int)
b = np.zeros(1, dtype=int)

for i in range(10):
    if num1 % 2 == 0 and num2 % 2 == 0:
        a = num1
        b = num2
        a_saved = a
        b_saved = b
    else:
        a = a_saved
        b = b_saved

    print(a, b)


import numpy as np
import random



a_saved = []
b_saved = []
c = []
d = []


for i in range(10):
    num1 = np.float32(random.randrange(1, 11))
    num2 = np.array(random.randrange(1, 11))
    print(num1, num2)
    if num1 % 2 == 0 or num2 % 2 == 0:
        a = num1
        b = num2
        a_saved = a
        b_saved = b
        c.append(a)
        d.append(b)
    else:
        a = a_saved
        b = b_saved
        c.append(a)
        d.append(b)
print(c, d, num1.dtype, num2.dtype)


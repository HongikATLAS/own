import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a/sum_exp_a
#     return y

def softmax(a): # softmax 함수 값들의 합은 항상 1이다.
    c = np.max(a)
    exp_a = np.exp(a-c) # 오버플로우를 방지하기 위함이다.
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y

x = np.array([1.0, 0.5])
w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])
w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])

a1 = np.dot(x,w1) + b1
z1 = sigmoid(a1)
a2 = np.dot(z1, w2) + b2
z2 = sigmoid(a2)
a3 = np.dot(z2, w3) + b3
print(a3, softmax(a3), np.sum(softmax(a3)))






# 활성화 함수란 입력 신호의 총합을 출력신호로 변환하는 함수 즉 입력 신호의 총합이 활성화를 일으키는지를 정하는 역할
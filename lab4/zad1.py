import numpy as np


def activate(x):
    return 1 / (1 + np.exp(-x))


def forwardPass(wiek, waga, wzrost):
    hidden1 = wiek * (-0.46122) + waga * 0.97314 + wzrost * (-0.39203) + 0.80109
    hidden1_act = activate(hidden1)

    hidden2 = wiek * 0.78548 + waga * 2.10584 + wzrost * (-0.57847) + 0.43529
    hidden2_act = activate(hidden2)

    output = hidden1_act * -0.81546 + hidden2_act * 1.03775 - 0.2368
    return output


# print(forwardPass(23, 75, 176))

# wiek / waga / wzrost
dane = [
    [23, 75, 176],
    [25, 67, 180],
    [28, 120, 175],
    [22, 65, 165],
    [46, 70, 187],
    [50, 68, 180],
    [48, 97, 178],
]


for d in dane:
    print(forwardPass(d[0], d[1], d[2]))


# 0.7985341880063129
# 0.8009499165011525
# -0.0145099999999998
# 0.8009329715279782
# 0.8009499999938147
# 0.8009499999978288
# 0.01518239867249338

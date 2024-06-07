import numpy as np
import matplotlib.pyplot as plt

max_range = [50, 340]
random_range = np.random.randint(max_range[0], max_range[1])
g = 9.81
v = 50
h = 100
tries = 0


def projectile_range(angle):
    angle_rad = np.deg2rad(angle)
    distance = (
        (v * np.sin(angle_rad) + np.sqrt(v**2 * np.sin(angle_rad) ** 2 + 2 * g * h))
        * v
        * np.cos(angle_rad)
        / g
    )
    return distance


def draw_trajectory(angle, d):
    angle_rad = np.radians(angle)
    vx = v * np.cos(angle_rad)
    vy = v * np.sin(angle_rad)
    t_flight = d / vx

    t = np.linspace(0, t_flight, num=100)
    x = vx * t
    y = vy * t - 0.5 * g * t**2 + h

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="Trajektoria pocisku")
    plt.title("Trajektoria pocisku Warwolf")
    plt.xlabel("Odległość")
    plt.ylabel("Wysokość")
    plt.grid(True)
    plt.legend()
    plt.savefig("trajektoria.png")
    plt.show()


while True:
    aim_angle = input("Podaj kąt strzału: ")

    try:
        aim_angle = float(aim_angle)
        if 0 < aim_angle < 90:
            d = projectile_range(aim_angle)
            print("d: ", d)
            print("random_range: ", random_range)
            diff = abs(random_range - d)

            if diff < 5:
                print(f"Trafiłeś! Odległość od celu: {diff:.2f}m")
                draw_trajectory(aim_angle, d)
                print(f"Za {tries + 1} próbą")
                break
            else:
                print(f"Pudło! Odległość od celu: {random_range - d:.2f}m")
                tries += 1
        else:
            print("Kąt musi być z zakresu (0, 90)")
    except ValueError:
        print("Podaj poprawną wartość kąta")

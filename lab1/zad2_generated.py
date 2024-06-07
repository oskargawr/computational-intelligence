import numpy as np
import matplotlib.pyplot as plt


def random_target_range(min_range, max_range):
    return np.random.randint(min_range, max_range)


tries = 0


def projectile_range(angle, velocity, height, gravity):
    angle_rad = np.deg2rad(angle)
    distance = (
        (
            velocity * np.sin(angle_rad)
            + np.sqrt(velocity**2 * np.sin(angle_rad) ** 2 + 2 * gravity * height)
        )
        * velocity
        * np.cos(angle_rad)
        / gravity
    )
    return distance


def draw_trajectory(angle, velocity, height, gravity, target_distance):
    angle_rad = np.radians(angle)
    vx = velocity * np.cos(angle_rad)
    vy = velocity * np.sin(angle_rad)
    t_flight = target_distance / vx

    t = np.linspace(0, t_flight, num=100)
    x = vx * t
    y = vy * t - 0.5 * gravity * t**2 + height

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="Projectile trajectory")
    plt.title("Projectile Trajectory")
    plt.xlabel("Distance")
    plt.ylabel("Height")
    plt.grid(True)
    plt.legend()
    plt.savefig("trajectory.png")
    plt.show()


def main():
    max_range = [50, 340]
    gravity = 9.81
    velocity = 50
    height = 100

    target_distance = random_target_range(max_range[0], max_range[1])

    while True:
        aim_angle = input("Enter the firing angle: ")

        try:
            aim_angle = float(aim_angle)
            if 0 < aim_angle < 90:
                distance = projectile_range(aim_angle, velocity, height, gravity)
                print("Projectile range: ", distance)
                print("Target range: ", target_distance)
                diff = abs(target_distance - distance)

                if diff < 5:
                    print(f"You hit the target! Distance from target: {diff:.2f}m")
                    draw_trajectory(aim_angle, velocity, height, gravity, distance)
                    print(f"Number of tries: {tries + 1}")
                    break
                else:
                    print(f"Miss! Distance from target: {diff:.2f}m")
                    tries += 1
            else:
                print("The angle must be in the range (0, 90)")
        except ValueError:
            print("Please enter a valid angle value")


if __name__ == "__main__":
    main()

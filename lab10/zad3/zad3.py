import gymnasium as gym
import simpful as sf
import numpy as np

FS = sf.FuzzySystem()

# zmienna lingwistyczna - kąt wahadła 
angle_negative = sf.FuzzySet(
    function=sf.Triangular_MF(a=-np.pi, b=-np.pi / 2, c=0), term="negative"
)
angle_zero = sf.FuzzySet(
    function=sf.Triangular_MF(a=-np.pi / 2, b=0, c=np.pi / 2), term="zero"
)
angle_positive = sf.FuzzySet(
    function=sf.Triangular_MF(a=0, b=np.pi / 2, c=np.pi), term="positive"
)
FS.add_linguistic_variable(
    "Angle",
    sf.LinguisticVariable(
        [angle_negative, angle_zero, angle_positive],
        universe_of_discourse=[-np.pi, np.pi],
    ),
)

# zmienna lingwistyczna - prędkość kątowa
angular_velocity_negative = sf.FuzzySet(
    function=sf.Triangular_MF(a=-8, b=-4, c=0), term="negative"
)
angular_velocity_zero = sf.FuzzySet(
    function=sf.Triangular_MF(a=-4, b=0, c=4), term="zero"
)
angular_velocity_positive = sf.FuzzySet(
    function=sf.Triangular_MF(a=0, b=4, c=8), term="positive"
)
FS.add_linguistic_variable(
    "AngularVelocity",
    sf.LinguisticVariable(
        [angular_velocity_negative, angular_velocity_zero, angular_velocity_positive],
        universe_of_discourse=[-8, 8],
    ),
)

# zmienna lingwistyczna - moment siły
torque_negative = sf.FuzzySet(
    function=sf.Triangular_MF(a=-2, b=-1, c=0), term="negative"
)
torque_zero = sf.FuzzySet(function=sf.Triangular_MF(a=-1, b=0, c=1), term="zero")
torque_positive = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=1, c=2), term="positive")
FS.add_linguistic_variable(
    "Torque",
    sf.LinguisticVariable(
        [torque_negative, torque_zero, torque_positive], universe_of_discourse=[-2, 2]
    ),
)

R1 = (
    "IF (Angle IS negative) AND (AngularVelocity IS negative) THEN (Torque IS positive)"
)
R2 = (
    "IF (Angle IS negative) AND (AngularVelocity IS positive) THEN (Torque IS negative)"
)
R3 = "IF (Angle IS zero) THEN (Torque IS zero)"
R4 = (
    "IF (Angle IS positive) AND (AngularVelocity IS negative) THEN (Torque IS negative)"
)
R5 = (
    "IF (Angle IS positive) AND (AngularVelocity IS positive) THEN (Torque IS positive)"
)
FS.add_rules([R1, R2, R3, R4, R5])

FS.plot_variable("Angle")
FS.plot_variable("AngularVelocity")
FS.plot_variable("Torque")

env = gym.make("Pendulum-v1", render_mode="human")
state, _ = env.reset()

for _ in range(200):
    env.render()

    cos_theta, sin_theta, angular_velocity = state
    angle = np.arctan2(sin_theta, cos_theta) 

    FS.set_variable("Angle", angle)
    FS.set_variable("AngularVelocity", angular_velocity)

    torque = FS.Mamdani_inference(["Torque"])["Torque"]

    state, reward, done, _, _ = env.step([torque])
    if done:
        break

env.close()

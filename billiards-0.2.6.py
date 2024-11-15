import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

a, b = 2, 1
attraction_point = [0.0, 0.0]
attraction_radius = 0.5
gravity = 0.0

initial_conditions = [
    ([0.0, 1.5], np.pi / 3),
    ([1.5, 0.5], np.pi / 4),
    ([-1.5, 0.5], np.pi / 6),
    ([0.5, -1.5], 2 * np.pi / 3)
]

fig, (ax_phase, ax_position_angle) = plt.subplots(1, 2, figsize=(15, 5))
colors = ['b', 'g', 'r', 'm']

ax_phase.set_xlim(-a, a)
ax_phase.set_ylim(0, 1.0)
ax_phase.set_xlabel("x-coordinate")
ax_phase.set_ylabel("tangent velocity")

ax_position_angle.set_xlim(-a - 1, 3 * a + 1)
ax_position_angle.set_ylim(0, np.pi) 
ax_position_angle.set_xlabel("x-coordinate")
ax_position_angle.set_ylabel("Angle (radians)")

def check_collision(x, y, a, b):
    return (x ** 2 / a ** 2 + y ** 2 / b ** 2) >= 1

def is_in_field(x, y):
    return math.sqrt((x - attraction_point[0]) ** 2 + (y - attraction_point[1]) ** 2) <= attraction_radius

def reflect_off_ellipse(x, y, vx, vy, a, b):
    normal_x = 2 * x / a ** 2
    normal_y = 2 * y / b ** 2
    normal_len = np.sqrt(normal_x ** 2 + normal_y ** 2)
    normal_x /= normal_len
    normal_y /= normal_len
    dot_product = vx * normal_x + vy * normal_y
    vx_reflected = vx - 2 * dot_product * normal_x
    vy_reflected = vy - 2 * dot_product * normal_y
    return vx_reflected, vy_reflected

def calculate_tangent_velocity(x, y, vx, vy, a, b):
    tangent_x = -b ** 2 * x
    tangent_y = a ** 2 * y
    tangent_len = np.sqrt(tangent_x ** 2 + tangent_y ** 2)
    tangent_x /= tangent_len
    tangent_y /= tangent_len
    tangent_velocity = vx * tangent_x + vy * tangent_y
    return np.abs(tangent_velocity)

def calculate_reflection_position_angle(x, y, vx, vy, a, b):
    A = (vx ** 2 / a ** 2) + (vy ** 2 / b ** 2)
    B = 2 * ((x * vx) / a ** 2 + (y * vy) / b ** 2)
    C = (x ** 2 / a ** 2) + (y ** 2 / b ** 2) - 1
    t_collision = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
    x_new = x + vx * t_collision
    y_new = y + vy * t_collision
    return x_new, y_new, t_collision

for idx, (initial_position, initial_angle) in enumerate(initial_conditions):
    position = initial_position
    velocity = [np.cos(initial_angle), np.sin(initial_angle)]
    phase_data = []
    position_angle_data = []

    n_reflection = 10000
    for _ in range(n_reflection):
        x_0, y_0 = position
        vx, vy = velocity

        x_new, y_new, t_collision = calculate_reflection_position_angle(x_0, y_0, vx, vy, a, b)
        if t_collision > 0:
            vx, vy = reflect_off_ellipse(x_new, y_new, vx, vy, a, b)
            velocity = [vx, vy]
            tangent_velocity = calculate_tangent_velocity(x_0, y_0, vx, vy, a, b)
            phase_data.append((x_new, tangent_velocity))

            angle = np.arctan2(vy, vx)
            if y_new > 0:
                position_angle_data.append((x_new + 2*a, abs(angle)))
            else:
                position_angle_data.append((x_new, abs(angle)))

            position = [x_new, y_new]

    if phase_data:
        phase_x, phase_y = zip(*phase_data)
        ax_phase.plot(phase_x, phase_y, 'o', color=colors[idx], markersize=3, label=f'Initial condition {idx + 1}')

    if position_angle_data:
        pa_x, pa_y = zip(*position_angle_data)
        ax_position_angle.plot(pa_x, pa_y, 'o', color=colors[idx], markersize=3, label=f'Initial condition {idx + 1}')

ax_phase.legend()
ax_position_angle.legend()

plt.show()

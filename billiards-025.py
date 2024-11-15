import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

a, b = 2, 1
step = 0.075
attraction_point = [0.0, 0.0]
attraction_radius = 0.5
gravity = 0.0

# List of initial positions and angles
initial_conditions = [
    ([0.0, 1.0], np.pi / 4),
    ([1.0, 0.5], np.pi / 3),
    ([-1.0, 0.5], np.pi / 6),
    ([0.5, -1.0], 3 * np.pi / 4)
]

fig, (ax_phase, ax_position_angle) = plt.subplots(1, 2, figsize=(15, 5))
colors = ['b', 'g', 'r', 'm']

ax_phase.set_xlim(-a, a)
ax_phase.set_ylim(0, 1.0)
ax_phase.set_xlabel("x-coordinate (at bounce)")
ax_phase.set_ylabel("tangent velocity (at bounce)")

ax_position_angle.set_xlim(-a - 1, 3 * a + 1)
ax_position_angle.set_ylim(-np.pi, np.pi)  # Angle range in radians
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

for idx, (initial_position, initial_angle) in enumerate(initial_conditions):
    position = initial_position
    velocity = [np.cos(initial_angle), np.sin(initial_angle)]
    phase_data = []
    position_angle_data = []

    n_reflection = 10000
    for _ in range(n_reflection):
        x_0, y_0 = position
        vx, vy = velocity
        x_new = x_0 + vx * step
        y_new = y_0 + vy * step

        if is_in_field(x_new, y_new):
            down = [attraction_point[0] - x_new, attraction_point[1] - y_new]
            magnitude = np.hypot(down[0], down[1])
            vx += (down[0] / magnitude) * gravity
            vy += (down[1] / magnitude) * gravity
            velocity = [vx, vy]
            x_new = x_0 + vx * step
            y_new = y_0 + vy * step

        if check_collision(x_new, y_new, a, b):
            vx, vy = reflect_off_ellipse(x_0, y_0, vx, vy, a, b)
            x_new = x_0 + vx * step
            y_new = y_0 + vy * step
            velocity = [vx, vy]
            tangent_velocity = calculate_tangent_velocity(x_0, y_0, vx, vy, a, b)
            phase_data.append((x_0, tangent_velocity))

            angle = np.arctan2(vy, vx)
            if y_new > 0:
                position_angle_data.append((x_new + 2*a, angle))
            else:
                position_angle_data.append((x_new, angle))

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


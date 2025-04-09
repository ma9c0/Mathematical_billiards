import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, Circle

# Ellipse parameters and initial conditions
a, b = 4, 2
step = 0.01
attraction_point = [0.0, 0.0]
attraction_radius = 0.5
gravity = 0.005

# Initial position and angle
position = [0.5, -1.5]
angle = 2 * np.pi / 3
velocity = [np.cos(angle), np.sin(angle)]

# Setting up the plots
ellipse = Ellipse([0, 0], 2 * a, 2 * b, edgecolor='b', fc='None')
a_point = Circle([attraction_point[0], attraction_point[1]], attraction_radius, edgecolor='r', fc='r', alpha=0.3)

fig, (ax, ax_phase, ax_position_angle) = plt.subplots(1, 3, figsize=(15, 5))
ax.set_xlim(-a-1, a+ 1)
ax.set_ylim(-b-1, b+1)
ax.add_patch(ellipse)
ax.add_patch(a_point)
ball, = ax.plot([], [], 'ro')
trajectory_line, = ax.plot([], [], 'b-', linewidth=0.5)
trajectory = []

# Phase space setup
ax_phase.set_xlim(-a, a)
ax_phase.set_ylim(0, 1.0)
ax_phase.set_xlabel("x-coordinate (at bounce)")
ax_phase.set_ylabel("tangent velocity (at bounce)")
phase_points, = ax_phase.plot([], [], 'bo', markersize=3)
phase_data = []

# Position vs. Angle plot setup
ax_position_angle.set_xlim(-a - 1, a + 1)
ax_position_angle.set_ylim(-np.pi, np.pi)  # Angle range in radians
ax_position_angle.set_xlabel("x-coordinate")
ax_position_angle.set_ylabel("Angle (radians)")
position_angle_points, = ax_position_angle.plot([], [], 'go', markersize=1.5)
position_angle_data = []

# Check if collision occurs
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

def ball_movements(frame):
    global position, velocity

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
        phase_x, phase_y = zip(*phase_data)
        phase_points.set_data(phase_x, phase_y)
        angle = np.arctan2(vy, vx)
        position_angle_data.append((x_new, angle))
        pa_x, pa_y = zip(*position_angle_data)
        position_angle_points.set_data(pa_x, pa_y)
        

    position = [x_new, y_new]
    trajectory.append((x_new, y_new))
    trajectory_x, trajectory_y = zip(*trajectory)
    trajectory_line.set_data(trajectory_x, trajectory_y)
    ball.set_data([x_new], [y_new])

    # Calculate the angle of the velocity vector
    # angle = np.arctan2(vy, vx)
    # position_angle_data.append((x_new, angle))
    # pa_x, pa_y = zip(*position_angle_data)
    # position_angle_points.set_data(pa_x, pa_y)

    return ball, trajectory_line, phase_points, position_angle_points

ani = FuncAnimation(fig, ball_movements, frames=2000, interval=1, blit=True)
plt.show()

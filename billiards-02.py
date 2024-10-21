# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.patches import Ellipse

# a, b = 50, 5

# # Initial position and angle
# position = [0.0, 5.0]
# angle = np.pi / 4

# ellipse = Ellipse([0, 0], 2 * a, 2 * b, edgecolor='b', fc='None')

# fig, (ax, ax_phase) = plt.subplots(1, 2)
# ax.set_xlim(-a - 1, a + 1)
# ax.set_ylim(-a - 1, a + 1)

# ax.add_patch(ellipse)

# ball, = ax.plot([], [], 'ro')
# trajectory_line, = ax.plot([], [], 'b-', linewidth=0.5)
# trajectory = []

# ellipse_perimeter = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
# ax_phase.set_xlim(0, ellipse_perimeter / 2)
# ax_phase.set_ylim(0, np.pi / 2)
# ax_phase.set_xlabel("position")
# ax_phase.set_ylabel('angle')
# phase_points, = ax_phase.plot([], [], markersize=1)
# phase_data = []


# def check_collision(x, y, a, b):
#     return np.abs(x ** 2 / a ** 2 + y ** 2 / b ** 2 - 1) <= 1 * 10 ** (-2)


# def get_tangent_slope(x_0, y_0, a, b):
#     def tangent_line(x):
#         return -x_0 * b ** 2 / (y_0 * a ** 2) * (x - x_0) + y_0
#     return tangent_line, -x_0 * b ** 2 / (y_0 * a ** 2)


# def get_reflection_line_point(x_0, y_0, a, b, theta):
#     tangent_line, s1 = get_tangent_slope(x_0, y_0, a, b)
#     slope = (s1 + np.tan(theta)) / (1 - s1 * np.tan(theta))
#     b_i = y_0 - slope * x_0
#     x = np.sqrt((1 - b_i ** 2 / b ** 2) / (1 / a ** 2 + slope ** 2 / b ** 2))
#     y = slope * x + b_i
#     return slope, b_i, [x, y]


# def update_position(x_0, y_0, theta, a, b):
#     dx = np.cos(theta)
#     dy = np.sin(theta)
#     x_new = x_0 + dx
#     y_new = y_0 + dy

#     # Check if the ball collides with the ellipse boundary
#     if check_collision(x_new, y_new, a, b):
#         theta = -theta  # Reflect angle if collision
#     return x_new, y_new, theta


# def ball_movements(frame):
#     global position, angle

#     # Update the ball position
#     x_0, y_0 = position
#     x_new, y_new, angle = update_position(x_0, y_0, angle, a, b)

#     position = [x_new, y_new]

#     # Update the trajectory
#     trajectory.append((x_new, y_new))
#     trajectory_x, trajectory_y = zip(*trajectory)
#     trajectory_line.set_data(trajectory_x, trajectory_y)
#     ball.set_data([x_new], [y_new])  # Update ball position with x and y as separate sequences

#     # Update phase diagram data
#     phase_data.append((ellipse_perimeter / 2 * (x_new / a), np.abs(angle)))
#     phase_x, phase_y = zip(*phase_data)
#     phase_points.set_data(phase_x, phase_y)

#     return ball, trajectory_line, phase_points


# # Create the animation
# ani = FuncAnimation(fig, ball_movements, frames=200, interval=100, blit=True)

# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse

a, b = 2, 1

# Initial position and angle
position = [0.0, 1.0]
angle = np.pi / 4  # Initial direction angle
velocity = [np.cos(angle), np.sin(angle)]  # Initial velocity vector

ellipse = Ellipse([0, 0], 2 * a, 2 * b, edgecolor='b', fc='None')

fig, (ax, ax_phase) = plt.subplots(1, 2)
ax.set_xlim(-a - 1, a + 1)
ax.set_ylim(-a - 1, a + 1)

ax.add_patch(ellipse)

ball, = ax.plot([], [], 'ro')
trajectory_line, = ax.plot([], [], 'b-', linewidth=0.5)
trajectory = []

ax_phase.set_xlim(-a, a)  # X-axis limits as per the ellipse's x-axis
ax_phase.set_ylim(0, 1.0)  # Adjust Y-axis for tangent velocity
ax_phase.set_xlabel("x-coordinate (at bounce)")
ax_phase.set_ylabel('tangent velocity (at bounce)')
phase_points, = ax_phase.plot([], [], 'bo', markersize=3)
phase_data = []


def check_collision(x, y, a, b):
    # Check if the point is on or outside the ellipse
    return (x ** 2 / a ** 2 + y ** 2 / b ** 2) >= 1


def reflect_off_ellipse(x, y, vx, vy, a, b):
    # Compute the normal at the point (x, y) on the ellipse
    normal_x = 2 * x / a ** 2
    normal_y = 2 * y / b ** 2
    normal_len = np.sqrt(normal_x ** 2 + normal_y ** 2)
    normal_x /= normal_len
    normal_y /= normal_len

    # Reflect the velocity vector based on the normal
    dot_product = vx * normal_x + vy * normal_y
    vx_reflected = vx - 2 * dot_product * normal_x
    vy_reflected = vy - 2 * dot_product * normal_y

    return vx_reflected, vy_reflected


def calculate_tangent_velocity(x, y, vx, vy, a, b):
    # Tangent vector at the current position on the ellipse
    tangent_x = -b ** 2 * x
    tangent_y = a ** 2 * y
    tangent_len = np.sqrt(tangent_x ** 2 + tangent_y ** 2)
    tangent_x /= tangent_len
    tangent_y /= tangent_len

    # Project the velocity onto the tangent vector to get the tangent velocity
    tangent_velocity = vx * tangent_x + vy * tangent_y

    return np.abs(tangent_velocity)  # Use the absolute value of the tangent velocity


def ball_movements(frame):
    global position, velocity

    # Update the ball position
    x_0, y_0 = position
    vx, vy = velocity

    x_new = x_0 + vx
    y_new = y_0 + vy

    # Check if the ball collides with the ellipse boundary
    if check_collision(x_new, y_new, a, b):
        # Reflect the velocity vector if there is a collision
        vx, vy = reflect_off_ellipse(x_0, y_0, vx, vy, a, b)
        x_new = x_0 + vx
        y_new = y_0 + vy
        velocity = [vx, vy]

        # Calculate the tangent velocity just after the collision
        tangent_velocity = calculate_tangent_velocity(x_0, y_0, vx, vy, a, b)

        # Append the x-coordinate and tangent velocity to the phase space data
        phase_data.append((x_0, tangent_velocity))
        phase_x, phase_y = zip(*phase_data)
        phase_points.set_data(phase_x, phase_y)

    position = [x_new, y_new]

    # Update the trajectory (draw continuous line on the left plot)
    trajectory.append((x_new, y_new))
    trajectory_x, trajectory_y = zip(*trajectory)
    trajectory_line.set_data(trajectory_x, trajectory_y)
    ball.set_data([x_new], [y_new])  # Update ball position with x and y as separate sequences

    return ball, trajectory_line, phase_points


# Create the animation
ani = FuncAnimation(fig, ball_movements, frames=200, interval=10, blit=True)

plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.patches import Ellipse

# a, b = 50, 5

# # Initial position and constant speed
# position = [0.0, 5.0]
# angle = np.pi / 14  # Initial direction angle
# speed = 1  # Constant speed (fixed magnitude of velocity)
# velocity = [speed * np.cos(angle), speed * np.sin(angle)]  # Initial velocity vector with constant magnitude

# ellipse = Ellipse([0, 0], 2 * a, 2 * b, edgecolor='b', fc='None')

# fig, (ax, ax_phase) = plt.subplots(1, 2)
# ax.set_xlim(-a - 1, a + 1)
# ax.set_ylim(-a - 1, a + 1)

# ax.add_patch(ellipse)

# ball, = ax.plot([], [], 'ro')
# trajectory_line, = ax.plot([], [], 'b-', linewidth=0.5)
# trajectory = []

# ax_phase.set_xlim(-a, a)  # X-axis limits as per the ellipse's x-axis
# ax_phase.set_ylim(0, np.pi / 2)  # Y-axis limits for angle
# ax_phase.set_xlabel("x-coordinate (at bounce)")
# ax_phase.set_ylabel('angle (at bounce)')
# phase_points, = ax_phase.plot([], [], 'bo', markersize=3)
# phase_data = []


# def check_collision(x, y, a, b):
#     # Check if the point is on or outside the ellipse
#     return (x ** 2 / a ** 2 + y ** 2 / b ** 2) >= 1


# def reflect_off_ellipse(x, y, vx, vy, a, b):
#     # Compute the normal at the point (x, y) on the ellipse
#     normal_x = 2 * x / a ** 2
#     normal_y = 2 * y / b ** 2
#     normal_len = np.sqrt(normal_x ** 2 + normal_y ** 2)
#     normal_x /= normal_len
#     normal_y /= normal_len

#     # Reflect the velocity vector based on the normal
#     dot_product = vx * normal_x + vy * normal_y
#     vx_reflected = vx - 2 * dot_product * normal_x
#     vy_reflected = vy - 2 * dot_product * normal_y

#     # Normalize the reflected velocity to maintain constant speed
#     velocity_magnitude = np.sqrt(vx_reflected ** 2 + vy_reflected ** 2)
#     vx_reflected = speed * vx_reflected / velocity_magnitude
#     vy_reflected = speed * vy_reflected / velocity_magnitude

#     return vx_reflected, vy_reflected


# def calculate_tangent_angle(x, y, vx, vy, a, b):
#     # Tangent vector at the current position on the ellipse
#     tangent_x = -b ** 2 * x
#     tangent_y = a ** 2 * y
#     tangent_len = np.sqrt(tangent_x ** 2 + tangent_y ** 2)
#     tangent_x /= tangent_len
#     tangent_y /= tangent_len

#     # Compute the angle between the velocity and the tangent vector
#     dot_product = vx * tangent_x + vy * tangent_y
#     velocity_len = np.sqrt(vx ** 2 + vy ** 2)
#     tangent_angle = np.arccos(dot_product / velocity_len)

#     # Map the angle to the range [0, pi/2]
#     tangent_angle = np.abs(np.pi / 2 - tangent_angle)

#     return tangent_angle


# def ball_movements(frame):
#     global position, velocity

#     # Update the ball position
#     x_0, y_0 = position
#     vx, vy = velocity

#     x_new = x_0 + vx
#     y_new = y_0 + vy

#     # Check if the ball collides with the ellipse boundary
#     if check_collision(x_new, y_new, a, b):
#         # Reflect the velocity vector if there is a collision
#         vx, vy = reflect_off_ellipse(x_0, y_0, vx, vy, a, b)
#         x_new = x_0 + vx
#         y_new = y_0 + vy
#         velocity = [vx, vy]

#         # Calculate the angle between velocity and the tangent vector at the point of collision
#         tangent_angle = calculate_tangent_angle(x_new, y_new, vx, vy, a, b)

#         # Append the x-coordinate and tangent angle to the phase space data
#         phase_data.append((x_new, tangent_angle))
#         phase_x, phase_y = zip(*phase_data)
#         phase_points.set_data(phase_x, phase_y)

#     position = [x_new, y_new]

#     # Update the trajectory (draw continuous line on the left plot)
#     trajectory.append((x_new, y_new))
#     trajectory_x, trajectory_y = zip(*trajectory)
#     trajectory_line.set_data(trajectory_x, trajectory_y)
#     ball.set_data([x_new], [y_new])  # Update ball position with x and y as separate sequences

#     return ball, trajectory_line, phase_points


# # Create the animation
# ani = FuncAnimation(fig, ball_movements, frames=200, interval=10, blit=True)  # Faster interval for quicker animation

# plt.show()

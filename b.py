import math
import numpy as np
import matplotlib.pyplot as plt

a, b = 2, 1  
attraction_point = [0.0, 0.0]
attraction_radius = 0.5
gravity = 0.5


initial_conditions = [
    ([0.0, 1.5], np.pi / 3),
    ([1.5, 0.5], np.pi / 4),
    ([-1.5, 0.5], np.pi / 6),
    ([0.5, -1.5], 2 * np.pi / 3)
]


fig, ax_position_angle = plt.subplots(1, 1, figsize=(10, 5))
colors = ['b', 'g', 'r', 'm']

ax_position_angle.set_xlim(-a - 1, 3 * a + 1)
ax_position_angle.set_ylim(0, np.pi)
ax_position_angle.set_xlabel("x-coordinate")
ax_position_angle.set_ylabel("Angle (radians)")

def reflect_off_ellipse(x, y, vx, vy, a, b):
    normal_x = 2 * x / a ** 2
    normal_y = 2 * y / b ** 2
    normal_len = np.hypot(normal_x, normal_y)
    normal_x /= normal_len
    normal_y /= normal_len
    dot_product = vx * normal_x + vy * normal_y
    vx_reflected = vx - 2 * dot_product * normal_x
    vy_reflected = vy - 2 * dot_product * normal_y
    return vx_reflected, vy_reflected

def calculate_reflection_position_angle(x, y, vx, vy, a, b):
    A = (vx ** 2) / a ** 2 + (vy ** 2) / b ** 2
    B = 2 * ((x * vx) / a ** 2 + (y * vy) / b ** 2)
    C = (x ** 2) / a ** 2 + (y ** 2) / b ** 2 - 1

    discriminant = B ** 2 - 4 * A * C
    if discriminant < 0:
        return None, None, None

    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-B - sqrt_discriminant) / (2 * A)
    t2 = (-B + sqrt_discriminant) / (2 * A)

    t_list = [t for t in [t1, t2] if t > 1e-8]
    if not t_list:
        return None, None, None
    t_collision = min(t_list)

    x_new = x + vx * t_collision
    y_new = y + vy * t_collision
    return x_new, y_new, t_collision

def calculate_touching_position_angle_of_attraction_field(x, y, vx, vy, a, b, m, n, gravity):
    if gravity == 0.0:
        x_new, y_new, t_collision = calculate_reflection_position_angle(x, y, vx, vy, a, b)
        return x_new, y_new, t_collision, 'ellipse'

    x_new_e, y_new_e, t_collision_to_shape = calculate_reflection_position_angle(x, y, vx, vy, a, b)
    if t_collision_to_shape is None:
        t_collision_to_shape = float('inf')

    A = vx**2 + vy**2
    B = 2 * ((x - m) * vx + (y - n) * vy)
    C = (x - m)**2 + (y - n)**2 - attraction_radius**2
    discriminant = B ** 2 - 4 * A * C

    if discriminant >= 0:
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-B - sqrt_discriminant) / (2 * A)
        t2 = (-B + sqrt_discriminant) / (2 * A)
        positive_times = [t for t in [t1, t2] if t > 1e-8]
        if positive_times:
            t_collision_to_attraction_field = min(positive_times)
            x_new_af = x + vx * t_collision_to_attraction_field
            y_new_af = y + vy * t_collision_to_attraction_field
        else:
            t_collision_to_attraction_field = float('inf')
            x_new_af, y_new_af = None, None
    else:
        t_collision_to_attraction_field = float('inf')
        x_new_af, y_new_af = None, None

    if t_collision_to_shape <= t_collision_to_attraction_field:
        if x_new_e is not None and y_new_e is not None:
            return x_new_e, y_new_e, t_collision_to_shape, 'ellipse'
        else:
            return None, None, float('inf'), None
    else:
        if x_new_af is not None and y_new_af is not None:
            return x_new_af, y_new_af, t_collision_to_attraction_field, 'attraction_field'
        else:
            return None, None, float('inf'), None

for idx, (initial_position, initial_angle) in enumerate(initial_conditions):
    position = initial_position.copy()
    velocity = [np.cos(initial_angle), np.sin(initial_angle)]
    position_angle_data = []

    n_reflection = 1000
    for _ in range(n_reflection):
        x_0, y_0 = position
        vx, vy = velocity

        result = calculate_touching_position_angle_of_attraction_field(
            x_0, y_0, vx, vy, a, b, attraction_point[0], attraction_point[1], gravity
        )
        x_new, y_new, t_collision, collision_type = result

        if np.isfinite(t_collision) and x_new is not None and y_new is not None:
            if collision_type == 'ellipse':
                vx, vy = reflect_off_ellipse(x_new, y_new, vx, vy, a, b)
                angle = np.arctan2(vy, vx)
                if y_new > 0:
                    position_angle_data.append((x_new + 2 * a, abs(angle)))
                else:
                    position_angle_data.append((x_new, abs(angle)))
                position = [x_new, y_new]
                velocity = [vx, vy]
            elif collision_type == 'attraction_field':
                position = [x_new, y_new]
                # Apply gravity: adjust velocity towards the attraction point
                direction_x = attraction_point[0] - x_new
                direction_y = attraction_point[1] - y_new
                distance = np.hypot(direction_x, direction_y)
                if distance != 0:
                    direction_x /= distance
                    direction_y /= distance
                    vx += gravity * direction_x
                    vy += gravity * direction_y
                velocity = [vx, vy]
                continue
        else:
            break

    if position_angle_data:
        pa_x, pa_y = zip(*position_angle_data)
        ax_position_angle.plot(pa_x, pa_y, 'o', color=colors[idx], markersize=3, label=f'Initial condition {idx + 1}')

ax_position_angle.legend()
plt.show()

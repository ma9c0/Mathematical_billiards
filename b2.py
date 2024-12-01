import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from scipy.optimize import fsolve

a, b = 2, 1
attraction_point = [0.0, 0.0]
attraction_radius = 0.5
gravity = 0.01

initial_conditions = [
    ([0.0, 1.5], np.pi / 3),
    ([1.5, 0.5], np.pi / 4),
    ([-1.5, 0.5], np.pi / 6),
    ([0.5, -1.5], 2 * np.pi / 3)
]

fig, (ax_position, ax_position_angle) = plt.subplots(1, 2, figsize=(12, 6))
colors = ['b', 'g', 'r', 'm']

ellipse = Ellipse([0, 0], 2 * a, 2 * b, edgecolor='b', fc='None')
a_point = Circle([attraction_point[0], attraction_point[1]], attraction_radius, edgecolor='r', fc='r', alpha=0.3)
ax_position.add_patch(ellipse)
ax_position.add_patch(a_point)
ax_position.set_xlim(-a - 1, a + 1)
ax_position.set_ylim(-b - 1, b + 1)
ax_position.set_xlabel('x')
ax_position.set_ylabel('y')

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

def is_inside_ellipse(x, y, a, b):
    return (x ** 2 / a ** 2) + (y ** 2 / b ** 2) <= 1

def is_inside_attraction_field(x, y, m, n, r):
    return (x - m) ** 2 + (y - n) ** 2 <= r ** 2

def compute_shm_parameters(x0, y0, vx0, vy0, m, n, gravity):
    omega = np.sqrt(gravity)
    A_x = x0 - m
    A_y = y0 - n
    B_x = vx0 / omega
    B_y = vy0 / omega
    return omega, A_x, A_y, B_x, B_y

def time_to_exit_attraction_field(x0, y0, vx0, vy0, m, n, attraction_radius, gravity):
    omega, A_x, A_y, B_x, B_y = compute_shm_parameters(x0, y0, vx0, vy0, m, n, gravity)
    P = A_x**2 + A_y**2
    S = B_x**2 + B_y**2
    Q = 2 * (A_x * B_x + A_y * B_y)
    D0 = (P + S) / 2
    D1 = (P - S) / 2
    D2 = Q / 2
    A_amp = np.hypot(D1, D2)
    phi = np.arctan2(D2, D1)
    D3 = attraction_radius**2 - D0
    if A_amp == 0:
        # Particle is at the center; stays inside attraction field
        return None
    ratio = D3 / A_amp
    if abs(ratio) > 1:
        # No real solution; particle does not exit the attraction field
        return None
    # Compute possible times when particle reaches attraction field boundary
    t1 = (np.arccos(ratio) + phi) / (2 * omega)
    t2 = (-np.arccos(ratio) + phi) / (2 * omega)
    positive_times = [t for t in [t1, t2] if t > 0]
    if positive_times:
        t_exit = min(positive_times)
        return t_exit
    else:
        return None

def shm_position_velocity(t, m, n, omega, A_x, A_y, B_x, B_y):
    x = m + A_x * np.cos(omega * t) + B_x * np.sin(omega * t)
    y = n + A_y * np.cos(omega * t) + B_y * np.sin(omega * t)
    vx = -A_x * omega * np.sin(omega * t) + B_x * omega * np.cos(omega * t)
    vy = -A_y * omega * np.sin(omega * t) + B_y * omega * np.cos(omega * t)
    return x, y, vx, vy

def time_to_collide_with_ellipse_inside_attraction_field(x0, y0, vx0, vy0, a, b, m, n, gravity):
    omega, A_x, A_y, B_x, B_y = compute_shm_parameters(x0, y0, vx0, vy0, m, n, gravity)
    # Define the function whose root we want to find
    def func(t):
        x, y, _, _ = shm_position_velocity(t, m, n, omega, A_x, A_y, B_x, B_y)
        return x ** 2 / a ** 2 + y ** 2 / b ** 2 - 1
    # Initial guess for t
    t_guess = 0.0
    # Use fsolve to find the time of collision
    t_collision = None
    try:
        t_collision = fsolve(func, t_guess, xtol=1e-6)[0]
        if t_collision <= 0:
            t_collision = None
    except:
        t_collision = None
    return t_collision

def simulate_motion(x0, y0, vx0, vy0, a, b, m, n, attraction_radius, gravity):
    position_data = []
    position_angle_data = []
    x, y = x0, y0
    vx, vy = vx0, vy0
    while True:
        # Check if inside attraction field
        inside_field = is_inside_attraction_field(x, y, m, n, attraction_radius)
        if inside_field:
            # Compute time to exit attraction field
            t_exit = time_to_exit_attraction_field(x, y, vx, vy, m, n, attraction_radius, gravity)
            # Compute time to collide with ellipse
            t_collision = time_to_collide_with_ellipse_inside_attraction_field(x, y, vx, vy, a, b, m, n, gravity)
            # Determine next event
            times = []
            if t_exit is not None:
                times.append(('exit', t_exit))
            if t_collision is not None:
                times.append(('collision', t_collision))
            if not times:
                # No event; particle stays inside attraction field indefinitely
                break
            event, t_event = min(times, key=lambda x: x[1])
            # Update position and velocity at event time
            omega, A_x, A_y, B_x, B_y = compute_shm_parameters(x, y, vx, vy, m, n, gravity)
            x, y, vx, vy = shm_position_velocity(t_event, m, n, omega, A_x, A_y, B_x, B_y)
            position_data.append((x, y))
            if event == 'collision':
                # Reflect velocity
                vx, vy = reflect_off_ellipse(x, y, vx, vy, a, b)
                angle = np.arctan2(vy, vx)
                if y > 0:
                    position_angle_data.append((x + 2 * a, abs(angle)))
                else:
                    position_angle_data.append((x, abs(angle)))
            # Proceed to next event
        else:
            # Outside attraction field; straight-line motion
            # Compute time to enter attraction field
            t_enter = time_to_enter_attraction_field(x, y, vx, vy, m, n, attraction_radius)
            # Compute time to collide with ellipse
            t_collision = time_to_collide_with_ellipse_outside(x, y, vx, vy, a, b)
            # Determine next event
            times = []
            if t_enter is not None:
                times.append(('enter', t_enter))
            if t_collision is not None:
                times.append(('collision', t_collision))
            if not times:
                # No further events
                break
            event, t_event = min(times, key=lambda x: x[1])
            # Update position and velocity at event time
            x += vx * t_event
            y += vy * t_event
            position_data.append((x, y))
            if event == 'collision':
                # Reflect velocity
                vx, vy = reflect_off_ellipse(x, y, vx, vy, a, b)
                angle = np.arctan2(vy, vx)
                if y > 0:
                    position_angle_data.append((x + 2 * a, abs(angle)))
                else:
                    position_angle_data.append((x, abs(angle)))
            # Proceed to next event
    return position_data, position_angle_data

def time_to_enter_attraction_field(x0, y0, vx0, vy0, m, n, attraction_radius):
    # Solve quadratic equation for time to enter attraction field
    dx = x0 - m
    dy = y0 - n
    a_quad = vx0**2 + vy0**2
    b_quad = 2 * (dx * vx0 + dy * vy0)
    c_quad = dx**2 + dy**2 - attraction_radius**2
    discriminant = b_quad**2 - 4 * a_quad * c_quad
    if discriminant < 0:
        return None
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b_quad - sqrt_disc) / (2 * a_quad)
    t2 = (-b_quad + sqrt_disc) / (2 * a_quad)
    positive_times = [t for t in [t1, t2] if t > 0]
    if positive_times:
        return min(positive_times)
    else:
        return None

def time_to_collide_with_ellipse_outside(x0, y0, vx0, vy0, a, b):
    # Solve quadratic equation for collision with ellipse
    A = (vx0 ** 2 / a ** 2) + (vy0 ** 2 / b ** 2)
    B = 2 * ((x0 * vx0) / a ** 2 + (y0 * vy0) / b ** 2)
    C = (x0 ** 2 / a ** 2) + (y0 ** 2 / b ** 2) - 1
    discriminant = B ** 2 - 4 * A * C
    if discriminant < 0:
        return None
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-B - sqrt_disc) / (2 * A)
    t2 = (-B + sqrt_disc) / (2 * A)
    positive_times = [t for t in [t1, t2] if t > 0]
    if positive_times:
        return min(positive_times)
    else:
        return None

for idx, (initial_position, initial_angle) in enumerate(initial_conditions):
    position = initial_position
    velocity = [np.cos(initial_angle), np.sin(initial_angle)]
    position_data, position_angle_data = simulate_motion(
        position[0], position[1], velocity[0], velocity[1],
        a, b, attraction_point[0], attraction_point[1],
        attraction_radius, gravity
    )
    if position_data:
        x_coords = [pos[0] for pos in position_data]
        y_coords = [pos[1] for pos in position_data]
        ax_position.plot(x_coords, y_coords, marker='o', color=colors[idx], label=f'Condition {idx + 1}')
    if position_angle_data:
        pa_x, pa_y = zip(*position_angle_data)
        ax_position_angle.plot(pa_x, pa_y, 'o', color=colors[idx], markersize=3, label=f'Condition {idx + 1}')

ax_position_angle.legend()
ax_position.legend()
plt.show()

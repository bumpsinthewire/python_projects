import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import argparse

# Set up parsing
parser = argparse.ArgumentParser(description="Simulate movement on inclined plane")
parser.add_argument('--velocity', '-v', type=float, nargs='+', default=[20], help="Initial velocity (m/s), space-separated for multiple")
parser.add_argument('--increments', '-i', type=int, default=50, help="Amount of time increments")
parser.add_argument('--angle', '-a', type=float, default=30, help="Incline angle (degrees, 0 to < 90)")
parser.add_argument('--gravity', '-g', type=float, default=9.81, help="Gravity (m/s**2)")
parser.add_argument('--start', '-s', type=int, default=0, help="Starting time")
parser.add_argument('--end', '-e', type=int, default=10, help="Ending time")
args = parser.parse_args()

# Set up logging
logging.basicConfig(filename='kinematics.log', level=logging.INFO)

# Parsing validation
try:
    invalid_vs = [v for v in args.velocity if v < 0]
    if invalid_vs:
        logging.error(f"Invalid velocities: {invalid_vs}. Must be 0 or above")
        raise ValueError("All velocities must be 0 or above")
    if not args.increments > 0:
        logging.error(f"Invalid time: {args.increments}. Must be greater than 0")
        raise ValueError("Time increments must be greater than 0")
    if not 0 <= args.angle < 90:
        logging.error(f"Invalid angle: {args.angle}. Must be 0 to < 90")
        raise ValueError("Angle must be 0 to < 90")
    if not args.gravity > 0:
        logging.error(f"Invalid gravity: {args.gravity}. Must be greater than 0")
        raise ValueError("Gravity must be greater than 0")
    if not args.start < args.end:
        logging.error(f"Invalid starting time: {args.start}. Must be less than ending time")
        raise ValueError("Starting time must be less than ending time")
    if not args.end > 0:
        logging.error(f"Invalid ending time: {args.end}. Must be greater than starting time")
        raise ValueError("Ending time must be greater than 0")
except TypeError as e:
    logging.error(f"Type error in simulation inputs: {e}")
    raise
except ValueError as e:
    logging.error(f"Value error in input validation: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error in simulation: {e}")
    raise

# inputs
v0_list = args.velocity
v0_array = np.array(v0_list)
num_points = args.increments
start_time = args.start
end_time = args.end
g = args.gravity
angle = args.angle
logging.info(f"Parsed {len(v0_array)} velocities: {v0_list} m/s, time increments from {start_time} to {end_time}: {num_points}, g: {args.gravity} m/s**2, angle: {args.angle} degrees")


# Incline equation
def incline(t, v0, g, angle):
    angle_rad = np.deg2rad(angle)
    a = g * np.sin(angle_rad)
    return -0.5 * a * t[:, None]**2 + v0 * t[:, None]


# Start overall logging time
overall_start_time = time.time()

try:
    # Compute time
    compute_start_time = time.time()

    # Time array
    t = np.linspace(start_time, end_time, num_points)
    logging.info(f"Time array created: {num_points} points from {t[0]:.1f} to {t[-1]:.1f} s")

    # Call the Incline function
    y = incline(t, v0_array, g, angle)

    # End compute time
    compute_end_time = time.time()

    # Set up plotting
    t_np = t
    y_np = y
    colors = ['b-', 'r-', 'g-', 'c-']
    for i in range(len(v0_array)):
        plt.plot(t_np, y_np[:, i], colors[i % len(colors)], label=f"v0={v0_array[i]:.1f} m/s")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.title("Incline simulation")
    plt.grid(True)
    plt.savefig("kinematics.png")
    plt.show()
    logging.info("Saved plot: kinematics.png")
    logging.info(f"Plotted {len(v0_array)} trajectories")

    # Save results to file
    with open('kinematics_results.txt', 'w') as f:
        np.savetxt(f, np.column_stack([t_np] + [y_np[:, j] for j in range(y_np.shape[1])]), header='Time Positions_v' + '_v'.join([f"{v:.1f}" for v in v0_list]), fmt='%.4f')
    logging.info("Save results to kinematics_results.txt")

except RuntimeError as e:
    logging.error(f"Scale test failed: {e}")
    raise
except ValueError as e:
    logging.error(f"Value error in simulation: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error in simulation: {e}")
    raise

# Finish tracking overall time
overall_end_time = time.time()
overall_compute_time = overall_end_time - overall_start_time

# Finish tracking compute time
compute_time = compute_end_time - compute_start_time

# Time tracking logging messages
logging.info(f"Overall compute time: {overall_compute_time:.4f} seconds.")
logging.info(f"Compute time: {compute_time:.4f} seconds.")

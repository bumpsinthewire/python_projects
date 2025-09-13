import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import argparse

# Set up parsing
parser = argparse.ArgumentParser(description="Simulate free fall with GPUs")
parser.add_argument('--velocity', '-v', type=float, nargs='+', default=[20], help="Initial velocity (m/s), space-separated for multiple")
parser.add_argument('--increments', '-i', type=int, default=50, help="Amount of time increments")
parser.add_argument('--start', '-s', type=int, default=0, help="Starting time")
parser.add_argument('--end', '-e', type=int, default=10, help="Ending time")
args = parser.parse_args()

# Set up logging
logging.basicConfig(filename='free_fall_cupy.log', level=logging.INFO)

# Parsing validation
try:
    invalid_vs = [v for v in args.velocity if v < 0]
    if invalid_vs:
        logging.error(f"Invalid velocities: {invalid_vs}. Must be 0 or above")
        raise ValueError("All velocities must be 0 or above")
    if not args.increments > 0:
        logging.error(f"Invalid time: {args.increments}. Must be greater than 0")
        raise ValueError("Time increments must be greater than 0")
    if not args.start < args.end:
        logging.error(f"Invalid starting time: {args.start}. Must be less than ending time")
        raise ValueError("Starting time must be less than ending time")
    if not args.end > 0:
        logging.error(f"Invalid ending time: {args.end}. Must be greater than starting time")
        raise ValueError("Time range must be greater than 0")
except TypeError as e:
    logging.error(f"Type error in simulation inputs: {e}")
    raise
except ValueError as e:
    logging.error(f"Value error in input validation: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error in simulation: {e}")
    raise

# Inputs
v0_list = args.velocity
v0_array = cp.array(v0_list)
num_points = args.increments
start_time = args.start
end_time = args.end
g = 9.81
logging.info(f"Parsed {len(v0_array)} velocities: {v0_list} m/s, time increments from {start_time} to {end_time}: {num_points}")

# Free fall equation
def free_fall(t, v0, g):
    t_reshaped = t[:, None]
    y = -0.5 * g * t_reshaped**2 + v0 * t_reshaped
    return y

# Start overall logging time
overall_start_time = time.time()

try:
    # Compute time
    compute_start_time = time.time()

    # Time array
    t = cp.linspace(start_time, end_time, num_points)
    logging.info(f"Time array created: {num_points} points from {t[0].get():.1f} to {t[-1].get():.1f} s")

    # Call the free_fall function
    y = free_fall(t, v0_array, g)

    # End compute time 
    compute_end_time = time.time()

    # Convert CuPy to NumPy for plotting
    t_np = t.get()
    y_np = y.get()

    # Set up plotting
    for i in range(len(v0_array)):
        plt.plot(t_np, y_np[:, i], "b-", label=f"v0={v0_array[i].get():.1f} m/s")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.title("Free Fall Trajectories (CuPy)")
    plt.grid(True)
    plt.savefig("freefall_cupy.png")
    plt.show()
    logging.info("Saved plot: freefall_cupy.png")
    logging.info(f"Plotted {len(v0_array)} trajectories")

    # Save results to file
    with open('free_fall_cupy_results.txt', 'w') as f:
        np.savetxt(f, np.column_stack([t_np] + [y_np[:, j] for j in range(y_np.shape[1])]), header='Time Positions_v' + '_v'.join([f"{v:.1f}" for v in v0_list]), fmt='%.4f')
    logging.info(f"Saved results to free_fall_cupy_results.txt")

except cp.cuda.memory.OutOfMemoryError as e:
    logging.error(f"GPU memory error in simulation: {e}")
    raise
except ValueError as e:
    logging.error(f"Value error in simulation: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error in simulation: {e}")
    raise

# Finish computing overall time
overall_end_time = time.time()
overall_compute_time = overall_end_time - overall_start_time

# Finish computing end time
compute_time = compute_end_time - compute_start_time

# Compute logging messages
logging.info(f"Overall compute time: {overall_compute_time:.4f} seconds.")
logging.info(f"Compute time: {compute_time:.4f} seconds.")

import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import argparse

# Set up parsing
parser = argparse.ArgumentParser(description="Simulate free fall")
parser.add_argument('--velocity', '-v', type=float, default=20, help="Initial velocity (m/s)")
parser.add_argument('--increments', '-i', type=int, default=50, help="Amount of time increments")
parser.add_argument('--start', '-s', type=int, default=0, help="Starting time")
parser.add_argument('--end', '-e', type=int, default=10, help="Ending time")
args = parser.parse_args()

# Set up logging
logging.basicConfig(filename='free_fall.log', level=logging.INFO)

# Parsing validation
try:
    if not args.velocity >= 0:
        logging.error(f"Invalid velocity: {args.velocity}. Must be 0 or above")
        raise ValueError("Velocity must be 0 or above")
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
v0 = args.velocity
num_points = args.increments
start_time = args.start
end_time = args.end
g = 9.81
logging.info(f"Parsed velocity: {v0} m/s, time increments from {start_time} to {end_time}: {num_points}")

# Free fall equation
def free_fall(t, v0, g):
    y = -1/2 * g * t**2 + v0 * t
    return y

# Start overall logging time
overall_start_time = time.time()

try:
    # Compute time
    compute_start_time = time.time()

    # Time array
    t = np.linspace(start_time, end_time, num_points)
    logging.info(f"Time array created: {num_points} points from {t[0]:.1f} to {t[-1]:.1f} s")

    # Call the free_fall function
    y = free_fall(t, v0, g)

    # End compute time 
    compute_end_time = time.time()

    # Set up plotting
    plt.plot(t, y, "b-")
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.title("Free fall")
    plt.grid(True)
    plt.savefig("freefall.png")
    plt.show()
    logging.info("Saved plot: freefall.png")
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

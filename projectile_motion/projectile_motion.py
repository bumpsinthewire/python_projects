import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse

# Set up parsing
parser = argparse.ArgumentParser(description="Simulate projectile motion")
parser.add_argument('--velocity', type=float, default=20, help="Initial velocity (m/s)")
parser.add_argument('--angle', type=float, default=45, help="Launch angle (degrees)")
parser.add_argument('--timeplots', type=int, default=50, help="Amount of time increments")
args = parser.parse_args()

# Set up logging
logging.basicConfig(filename='projectile_motion.log', level=logging.INFO)

# Validate angle
if not 0 <= args.angle <= 90:
    logging.error(f"Invalid angle: {args.angle}. Must be 0-90 degrees")
    raise ValueError("Angle must be 0-90 degrees")

# Inputs
v0 = args.velocity  # Initial velocity (m/s)
theta = args.angle  # Launch angle (degrees)
time_increments = args.timeplots
g = 9.81  # Gravity (m/s^2)
logging.info(f"Parsed velocity: {v0} m/s, angle: {theta} degrees, time increments: {time_increments}")

# Time array
t = np.linspace(0, 3, time_increments)  # 100 points from 0 to 3 seconds
logging.info(f"Time array created: {time_increments} points from 0 to {t[-1]:.1f} s")

# Convert angle to radians
theta_rad = np.deg2rad(theta)
logging.info(f"Angle converted to {theta_rad:.4f} radians")

# Horizontal position: x(t) = v0 * cos(theta) * t
x = v0 * np.cos(theta_rad) * t
logging.info(f"Computed x[0]={x[0]:.2f}, x[-1]={x[-1]:.2f} m")

# Vertical position: y(t) = v0 * sin(theta) * t - (1/2) * g * t^2
y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
logging.info(f"Computed y[0]={y[0]:.2f}, y[{time_increments//2}]={y[time_increments//2]:.2f} m")

# Plot x vs. y (trajectory)
plt.plot(x, y, "k.")
plt.xlabel("Time")
plt.ylabel("Height")
plt.title("Projectile Trajectory")
plt.grid(True)
plt.savefig("projectile.png")
plt.show()
logging.info("Saved plot: projectile.png")

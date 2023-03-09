import os
import datetime

# Define the directory where the log files are stored
log_directory = "./log_processes"

# Define the expected naming convention for the log files
file_name_pattern = "%Y-%m-%d_%H-%M-%S_*_log.txt"

# Get a list of all the log files in the log directory
log_files = [f for f in os.listdir(log_directory) if os.path.isfile(os.path.join(log_directory, f))]

# Filter out any files that do not match the expected naming convention
log_files = [f for f in log_files if datetime.datetime.strptime(f.split("_")[0], "%Y-%m-%d")]

# Sort the remaining files by their modification time, with the most recent file first
log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_directory, x)), reverse=True)

# Get the path to the most recent log file, if one exists
if log_files:
    most_recent_log_file = os.path.join(log_directory, log_files[0])
else:
    most_recent_log_file = None

print(f"Most recent log file: {most_recent_log_file}")

with open(most_recent_log_file, "r") as log_file:
    log_contents = log_file.read()
    pid2 = int(log_contents.split("\n")[1].split(":")[1])

print(f"PID of mpc_process: {pid2}")
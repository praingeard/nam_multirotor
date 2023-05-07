import subprocess
import datetime
import os

# Define the paths to the two Python files you want to start
file1_path = "./reinforcement_learning/dqn_run.py"
file2_path = "./mpc.py"

# Start the two files simultaneously using the subprocess module and get their PIDs
rl_process = subprocess.Popen(["python3", file1_path])
mpc_process = subprocess.Popen(["python3", file2_path])

# Retrieve the PIDs of the two subprocesses
rl_pid = rl_process.pid
mpc_pid = mpc_process.pid

print(f"PID of rl_process: {rl_pid}")
print(f"PID of mpc_process: {mpc_pid}")

# Get the current date and time in the desired format
now = datetime.datetime.now()
date_str = now.strftime("%Y-%m-%d")
time_str = now.strftime("%H-%M-%S")

operation_name = "rl_leader"
log_file_name = f"{date_str}_{time_str}_{operation_name}_log.txt"

# Create the log file in the /log_processes directory if it doesn't already exist
log_file_path = os.path.join("./log_processes", log_file_name)
if not os.path.exists("./log_processes"):
    os.makedirs("./log_processes")
with open(log_file_path, "w") as log_file:
    # Write the PIDs of the two subprocesses to the log file
    log_file.write(f"PID of rl_process: {rl_pid}\n")
    log_file.write(f"PID of mpc_process: {mpc_pid}\n")

print(f"Log file written to {log_file_path}")

# Wait for the two subprocesses to finish and get their return codes
if rl_pid is not None:
    try:
        rl_process_returncode = rl_process.communicate()[0]
    except Exception as e:
        print(f"Error communicating with rl_process: {e}")
        rl_process_returncode = None

if mpc_pid is not None:
    try:
        mpc_returncode = mpc_process.communicate()[0]
    except Exception as e:
        print(f"Error communicating with mpc_process: {e}")
        mpc_returncode = None

# Write the return codes of the two subprocesses to the log file
with open(log_file_path, "a") as log_file:
    if rl_pid is not None:
        log_file.write(f"rl_process return code: {rl_process_returncode}\n")
    if mpc_pid is not None:
        log_file.write(f"mpc_process return code: {mpc_returncode}\n")
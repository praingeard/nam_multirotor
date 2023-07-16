import subprocess
import threading

# List of arguments for each run
arguments_list = [
    ["python3", "dqn_runsubproc.py", "Drone1"],
    ["python3", "dqn_runsubproc.py", "Drone2"],
    ["python3", "dqn_runsubproc.py", "Drone3"],
    ["python3", "dqn_runsubproc.py", "Leader"],
    ["python3", "airgym/envs/print_traj.py"]
    # Add more argument lists as needed
]

# Function to launch the script in a separate thread
def run_script(arguments):
    subprocess.run(arguments)

# Launch the script in separate threads
threads = []
for arguments in arguments_list:
    thread = threading.Thread(target=run_script, args=(arguments,))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()
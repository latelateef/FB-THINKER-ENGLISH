import subprocess
import time

# List of files to be executed
files_to_run = [
    'script1.py',
    'script2.py',
    'script3.py',
    'script4.py',
    'script5.py',
    'script6.py',
    'script7.py',
    'script8.py',
    'script9.py',
    'script10.py'
]

# Terminal command for Windows
terminal_command = 'start cmd /k'  # Use `start powershell` if you prefer PowerShell

# Run each file in a new terminal
processes = []
for file in files_to_run:
    command = f'{terminal_command} python {file}'
    processes.append(subprocess.Popen(command, shell=True))
    time.sleep(2)  # Optional: Add delay to avoid opening too many terminals at once

# Optional: Wait for all processes to complete
for process in processes:
    process.wait()

import shutil

for i in range(1, 10):
    source_file = f'script1.py'
    destination_file = f'script{i+1}.py'
    shutil.copy(source_file, destination_file)

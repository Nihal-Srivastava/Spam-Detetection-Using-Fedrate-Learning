import subprocess
import pandas as pd
import os

port = 5002

final_results = pd.DataFrame(columns=['Client no.', 'Type', 'Epoch', 'Average Loss', 'Accuracy'])

file_path = 'final_results.csv'

if os.path.isfile(file_path):
    print('The file exists')
else:
    print('The file does not exist')
    final_results.to_csv(file_path)

client = 8  # number of times to run the file
number_of_cluster = 5
top_select = 1
filename = "client.py"  # name of the file to run
data = []

# change the number of samples
for _ in range(client):
    samples = 2000
    data.append((str(samples)))

processes = [subprocess.Popen(["python", "server.py"] + [str(port), str(number_of_cluster), str(top_select)])]
for i in range(client):
    processes.append(subprocess.Popen(["python", filename] + [str(port), data[i], str(i + 1)]))

for process in processes:
    process.wait()

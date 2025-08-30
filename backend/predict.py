import requests
import os
# Define a mapping of numbers to class names
class_map = {
    '1': 'Cataract',
    '2': 'Conjunctivitis',
    '3': 'Eyelid',
    '4': 'Normal',
    '5': 'Uveitis'
}

# Get the file path from user input
file_path = input("Enter the file path (e.g. 1/1.jpg where 1=Cataract, 2=Conjunctivitis, 3=Eyelid, 4=Normal, 5=Uveitis): ")

# Extract class number and convert to class name
parts = file_path.split('/')
if len(parts) > 1 and parts[0] in class_map:
    file_path = class_map[parts[0]] + '/' + '/'.join(parts[1:])

# Construct the full path
full_path = os.path.join('./data', file_path)
print("Full path:", full_path)

# Check if file exists
if not os.path.exists(full_path):
    print(f"Error: File {full_path} not found.")
    exit(1)

url = 'http://127.0.0.1:5000/predict'
files = {'image': open(full_path, 'rb')}
response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response text:", repr(response.text))
if response.headers.get('Content-Type','').startswith('application/json'):
    print("JSON:", response.json())

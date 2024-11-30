import json
import os

# List to hold combined data
combined_data = {}

# Loop through the 10 JSON files and load their data
for i in range(1, 11):
    file_name = f'data_{i}.json'
    if os.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
            combined_data.update(data)
    else:
        print(f"{file_name} does not exist.")

# Save the combined data to a single JSON file
with open('combined_data.json', 'w', encoding='utf-8') as combined_file:
    json.dump(combined_data, combined_file, ensure_ascii=False, indent=4)

print("Data combined and saved as 'combined_data.json'.")

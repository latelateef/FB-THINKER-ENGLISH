import json

input_file = './chinese.json'

# Load the original JSON file
try:
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
except Exception as e:
    print(f"Error reading input file: {e}")

# Ensure the data is a dictionary (key-value pairs)
if isinstance(data, dict):
    keys = list(data.keys())
    total_pairs = len(keys)
    chunk_size = total_pairs // 10

    # Split the data into 10 chunks
    for i in range(10):
        start_index = i * chunk_size
        if i == 9:  # For the last chunk, include all remaining data
            chunk_keys = keys[start_index:]
        else:
            chunk_keys = keys[start_index:start_index + chunk_size]
        
        chunk = {key: data[key] for key in chunk_keys}
        
        # Save each chunk into a separate JSON file
        with open(f'data_part_{i+1}.json', 'w', encoding='utf-8') as chunk_file:
            json.dump(chunk, chunk_file, ensure_ascii=False, indent=4)
else:
    print("The JSON data is not a dictionary of key-value pairs.")

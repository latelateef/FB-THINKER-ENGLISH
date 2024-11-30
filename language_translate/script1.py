import json
from googletrans import Translator
import time
import random
from collections import deque
import os

def translate_text(text, translator):
    # Translate text from Chinese to English.
    for attempt in range(2):  # Retry up to 2 times
        try:
            translation = translator.translate(text, src='zh-CN', dest='en')
            return translation.text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(random.uniform(1, 3))  # Sleep for 1-3 seconds before retrying
    return text  # Return the original text if all attempts fail

def translate_json(input_file, output_file):
    start_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    translator = Translator()

    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    def translate_iteratively(data):
        # Iteratively translate both keys and values in a dictionary or list.
        queue = deque([data])

        while queue:
            current = queue.popleft()
            if isinstance(current, dict):
                items_to_translate = list(current.items())
                translated_dict = {}
                
                for k, v in items_to_translate:
                    # Translate the key
                    new_key = translate_text(k, translator) if isinstance(k, str) else k

                    # Handle value translation
                    if isinstance(v, (dict, list)):
                        queue.append(v)
                        translated_dict[new_key] = v  # Add the untranslated value, it will be processed later
                    elif isinstance(v, str):
                        translated_dict[new_key] = translate_text(v, translator)
                    else:
                        translated_dict[new_key] = v

                # Replace the current dictionary with the new translated one
                current.clear()
                current.update(translated_dict)
                
            elif isinstance(current, list):
                for i, item in enumerate(current):
                    if isinstance(item, (dict, list)):
                        queue.append(item)
                    elif isinstance(item, str):
                        current[i] = translate_text(item, translator)

        return data

    translated_data = translate_iteratively(data)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(translated_data, file, ensure_ascii=False, indent=4)

    end_time = time.time()
    print('Time taken:', end_time - start_time, 'seconds')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# Get the current script's filename
file_name = os.path.basename(__file__) # 'script1.py'
i = int(file_name.split('.')[0][-1])
input_file = f'./data_part_{i}.json'

output_file = f'data_{i}.json'
translate_json(input_file, output_file)
import json
import os

def load_json(input_data):
    if os.path.isfile(input_data):  # Check if it's a file path
        with open(input_data, "r", encoding="utf-8") as f:
            return json.load(f)
    else:  # Assume it's a JSON string
        return json.loads(input_data)
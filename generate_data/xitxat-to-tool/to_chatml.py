import logging
import argparse
import json

# Set up logging configuration
logging.basicConfig(level=logging.WARN, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def read_json(data_path: str) -> tuple[list, dict]:
    logger.info("Loading dataset...")
    try:
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
    except:
        with open(data_path, 'r') as f:
            data = json.loads(f.read())

    return data

def save_json(output_path, data):
    with open(output_path, 'w', encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)  

def run(input_path, output_path):
    data = read_json(input_path)
        
    for raw in data:
        conversations = raw["conversations"]
        for conv in conversations:
            if conv["from"] == "gpt" and conv.get("tool_calls", None):
                tool_calls = conv["tool_calls"]
                value = conv["value"].strip()
                for tool_call in tool_calls:
                    value += "<tool_call>\n"
                    value += json.dumps(tool_call, ensure_ascii=False)
                    value += "\n</tool_call>"
                
                conv.pop("tool_calls")
                conv["value"] = value
            elif conv["from"] == "tool":
                conv["value"] = json.dumps(conv["value"], ensure_ascii=False)
    save_json(output_path, data)





if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Distributed processing script.")
    parser.add_argument("--input", type=str, required=True, help="Input path.")
    parser.add_argument("--output", type=str, default="./output.json", help="Path to save output files.")

    args = parser.parse_args()
    
    input = args.input
    output = args.output
    
    run(input, output)



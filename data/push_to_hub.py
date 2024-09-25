from datasets import load_dataset, Dataset, DatasetDict
import json
from dotenv import load_dotenv
import sys
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from huggingface_hub import upload_file

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.retrieve_context import get_openai_tools

env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(env_path)
tools = get_openai_tools()

HF_TOKEN = os.environ["HF_TOKEN"]

if __name__ == "__main__":
    json_conversations_list = {}
    try:
        with open("tmp.json", "r") as file:
            json_conversations_list = json.loads(file.read())
    except:
        pass

    dataset = load_dataset("projecte-aina/RAG_Multilingual", token=HF_TOKEN)
    train_data = dataset.data["train"].to_pandas()

    # train_data['tools'] = tools
    train_data['conversations'] = None

    for id in json_conversations_list:
        index_to_update = train_data[train_data['id'] == id].index[0]
        train_data.at[index_to_update, 'conversations'] = json_conversations_list[id]
        
    train_data_cleaned = train_data.dropna(subset=['conversations'])
    train_data_cleaned['tools'] = [tools] * len(train_data_cleaned)
    train_data_cleaned = train_data_cleaned.drop(columns=['instruction', 'context', 'response'])
    train_data_90, validation_data_10 = train_test_split(train_data_cleaned, test_size=0.1, random_state=42)

    # train_dataset = Dataset.from_pandas(train_data_90)
    # validation_dataset = Dataset.from_pandas(validation_data_10)
    
    train_data_90.to_json("train.json", orient="records", indent=4, force_ascii=False)
    validation_data_10.to_json("validation.json", orient="records", indent=4, force_ascii=False)

    # Create Dataset objects from JSON files
    train_dataset = Dataset.from_json("train.json")
    validation_dataset = Dataset.from_json("validation.json")


    new_dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset
    })

    upload_file(
        path_or_fileobj="train.json",  # Path to your JSON file
        path_in_repo="train.json",   # Path in the repo
        repo_type="dataset",
        repo_id="BSC-LT/RAG_Multilingual_Conversational_Fastchat",  # Your repository name
        token=HF_TOKEN  # Authentication token
    )

    upload_file(
        path_or_fileobj="validation.json",  # Path to your JSON file
        path_in_repo="validation.json",   # Path in the repo
        repo_type="dataset",
        repo_id="BSC-LT/RAG_Multilingual_Conversational_Fastchat",  # Your repository name
        token=HF_TOKEN  # Authentication token
    )
    

    # new_dataset.push_to_hub("BSC-LT/RAG_Multilingual_Conversational_Fastchat", token=HF_TOKEN)






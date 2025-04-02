from abc import ABC, abstractmethod
import os
import sys
import importlib.util
import logging
from typing import Optional, Type, Union
import pandas as pd
import json
import glob
from generate_dataset.utils import timer
from generate_dataset.utils.logger import setup_logger
from generate_dataset.utils.class_manager import ClassManager
from typing import List, Dict, Protocol, Union, Any
import re

class MessagesType(Dict[str, str]):
    """Defines the message format with 'role' and 'content' keys."""
    role: str
    content: str


class GetLLMResponseType(Protocol):
    """A callable type for generating LLM responses.

    Args:
        messages (List[MessagesType]): A list of message dictionaries, each containing 'role' and 'content'.
        wait_for_connection (bool): Whether to wait until a connection is available.

    Returns:
        str: The generated AI response in text format.
    """
    def __call__(self, messages: List[MessagesType], wait_for_connection: bool) -> str:
        pass



logger = setup_logger(__name__)

class MethodManager(ClassManager):
    # This will inherit ClassManager and be used specifically for methods
    pass

class BaseMethod(ABC):
    def __init__(self, input: str, output: str, global_rank:int, messages_list: List[MessagesType], unique_key, output_keys, output_types):
        self.input = input
        self.output = output
        self.global_rank = global_rank
        self.messages_list = messages_list
        self.unique_key = unique_key
        self.output_keys = output_keys
        self.output_types = output_types
        self._detect_file_type(self.output)
        self.data = self._read_file(self.input)
        print(self.data)
        self._check_data(self.data, self.unique_key, self.output_keys, self.output_types, self.messages_list)
        self.processed_data_ids = self.extract_unique_key_values(self.output_path_pattern, self.unique_key)
        
        self.output_path_pattern, self.output_rank_path = self._generate_temporal_path(self.output, self.global_rank)

    @staticmethod
    def _check_data(df, unique_key, output_keys, output_types, messages_list):
        existing_keys = [key for key in output_keys if key in df.columns]
        if existing_keys:
            raise KeyError(f"These keys '{str(existing_keys)}' already exists in dataset.")
        
        if unique_key not in df.columns:
            raise ValueError(f"The unique key '{unique_key}' does not exist in the dataset.")
        
        if not df[unique_key].is_unique:
            raise ValueError(f"The unique key '{unique_key}' exists but its values are not unique in the dataset.")
        
        not_permitted_types = set(output_types) - ["json", "str"]
        if not_permitted_types:
            raise TypeError(f"Output types '{str(not_permitted_types)}' are not permitted")

        #TODO: Check messages format


    def get_unique_id(self, index):
        return self.data.at[index, self.unique_key]

    def generate_messages(self, json_data: Dict[str, Any], index: int = -1) -> List[MessagesType]:
            """Substitutes placeholders in messages with values from json_data.

            Args:
                json_data (Dict[str, Any]): Dictionary containing values to replace placeholders.
                index (int, optional): If -1, all messages are processed. Otherwise, only messages[index] is processed.

            Returns:
                List[Dict[str, str]]: The modified messages with placeholders replaced.
            
            Raises:
                KeyError: If a placeholder is found but not in json_data.
                IndexError: If index is out of range.
                ValueError: If no changes were made after substitution.
            """

            changed = False

            def substitute_placeholders(text: str) -> str:
                """Replaces placeholders using str.format(), raising an error if a key is missing."""
                try:
                    new_text = text.format(**json_data)  # Uses str.format() for substitution
                    if not changed and new_text != text:
                        changed = True
                    return new_text
                
                except KeyError as e:
                    raise KeyError(f"Missing key in json_data: {e}")

            if 0 <= index < len(self.messages_list):
                msg = self.messages_list[index]
                messages = [{"role": msg["role"], "content": substitute_placeholders(msg["content"])}]
            else:
                raise IndexError("Invalid index: out of range.")

            if not changed:
                raise ValueError("No substitutions made: The content is the same as before.")
            
            return messages

    @abstractmethod
    def generate_data(self, index, get_llm_response: GetLLMResponseType):
        """Function to be implemented by subclasses"""
        pass

    def is_done(self, index):
        return self.data.at[index, self.unique_key] in self.processed_data_ids

    @staticmethod
    def _generate_temporal_path(path: str, number: int) -> str:
        directory, filename = os.path.split(path)  # Separate path and filename
        name, _ = os.path.splitext(filename)  # Split filename and extension
        new_filename = f"._{name}_{number}.jsonl"  # Insert number before extension
        pattern = f"._{name}_*.jsonl"  # Insert number before extension
        return pattern, os.path.join(directory, new_filename)  # Reconstruct full path


    @staticmethod
    def _detect_file_type(path):
        """Detect file format based on extension."""
        _, ext = os.path.splitext(path.lower())
        if ext == ".json":
            return "json"
        elif ext in (".jsonl", ".ndjson"):
            return "jsonl"
        elif ext == ".csv":
            return "csv"
        elif ext == ".parquet":
            return "parquet"
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    @staticmethod
    def _read_file(path, file_type=None):
        if not file_type:
            file_type = BaseMethod._detect_file_type(path)
        
        """Reads a file and returns a Pandas DataFrame and its type."""
        if file_type == "json":
            df = pd.read_json(path)
        elif file_type == "jsonl":
            df = pd.read_json(path, lines=True)
        elif file_type == "csv":
            df = pd.read_csv(path)
        elif file_type == "parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        return df
    
    @staticmethod
    def _get_all_records(files_pattern):
        files = sorted(glob.glob(files_pattern))  # Find all matching JSONL files
        all_data = []

        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                all_data.extend(json.loads(line) for line in f)  # Load line-by-line

        return pd.DataFrame(all_data)
    
    @staticmethod
    def extract_unique_key_values(file_pattern, key):
        unique_values = set()  # To store unique values of the key

        # Get all file paths matching the pattern
        file_paths = glob.glob(file_pattern)

        for file_path in file_paths:
            df = BaseMethod._read_file(file_path)  # Read JSONL file into a DataFrame

            # Add unique values from the current file to the set
            unique_values.update(df[key].dropna().unique())

        return unique_values 

    def set_record(self, record, index):
        """Append JSON line by line"""
        with open(self.output_rank_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save_all(self):
        file_type = BaseMethod._detect_file_type(self.output)
        df = BaseMethod._get_all_records(self.output_path_pattern)
    
        if file_type == "json":
            df.to_json(self.output, orient="records", indent=4)
        elif file_type == "jsonl":
            df.to_json(self.output, orient="records", lines=True)
        elif file_type == "csv":
            df.to_csv(self.output, index=False)
        elif file_type == "parquet":
            df.to_parquet(self.output, index=False)
        
        [os.remove(file) for file in sorted(glob.glob(self.output_path_pattern))]
        logging.info(f"Saved: {self.output}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.data[index]



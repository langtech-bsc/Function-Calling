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

logger = setup_logger(__name__)

class MethodManager(ClassManager):
    # This will inherit ClassManager and be used specifically for methods
    pass

class BaseMethod(ABC):
    def __init__(self, input: str, output: str, global_rank):
        self.input = input
        self.output = output
        self._global_rank = global_rank

        self._detect_file_type(self.output)
        self.data = self._read_file(self.input)

        self.local_path_pattern, self.local_path = self._add_number_to_path(self.output, self._global_rank)

    @abstractmethod
    def messages(self, index):
        """Function to be implemented by subclasses"""
        pass

    @abstractmethod
    def is_done(self, index):
        pass

    @staticmethod
    def _add_number_to_path(path: str, number: int) -> str:
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
    def _remove_paths_by_pattern(files_pattern: str):
        """Removes all files matching the pattern."""
        [os.remove(file) for file in sorted(glob.glob(files_pattern))]
            
    
    def set_record(self, record, index):
        """Append JSON line by line"""
        with open(self.local_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logging.info(f"Saved: {self.local_path}")


    def save_all(self):
        file_type = BaseMethod._detect_file_type(self.output)
        df = BaseMethod._get_all_records(self.local_path_pattern)
    
        if file_type == "json":
            df.to_json(self.output, orient="records", indent=4)
        elif file_type == "jsonl":
            df.to_json(self.output, orient="records", lines=True)
        elif file_type == "csv":
            df.to_csv(self.output, index=False)
        elif file_type == "parquet":
            df.to_parquet(self.output, index=False)
        
        self._remove_paths_by_pattern(self.local_path_pattern)
        logging.info(f"Saved: {self.output}")

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.data[index]



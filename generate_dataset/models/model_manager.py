from abc import ABC, abstractmethod
import logging
from generate_dataset.utils.class_manager import ClassManager
import json
from types import SimpleNamespace

logger = logging.getLogger(__name__)

class ModelManager(ClassManager):
    # This will inherit ClassManager and be used specifically for methods
    pass

class BaseModel(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_model_name(self):
        pass
    def _get_params(self, params, default_params, restricted_keys = {}):
        found_keys = restricted_keys.intersection(params)
        if found_keys:
            raise ValueError(f"Can't pass '{', '.join(found_keys)}' in model-params, please remove them.")
        
        new_parasm = {}
        new_parasm.update(default_params)
        new_parasm.update(params)
        return new_parasm
    
    def print_args(self):
        attr = {"Model API": self.__class__.__name__}
        public_attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        attr.update(public_attrs)
        return SimpleNamespace(**attr)

    @abstractmethod
    def get_response(self, messages, wait_for_connection=False):
        """Function to be implemented by subclasses"""
        pass


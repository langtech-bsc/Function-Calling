from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools import tool
from typing import List
import json

# Define the retrieve_context function with an adjusted docstring
@tool
def retrieve_context(prompt: str) -> str:
    """
    Given a prompt, retrieves the context from vectorstore.
    """
    return "context"


def _get_functions_dict():
    return  {
        "retrieve_context": retrieve_context
    }
    

def get_openai_tools_name(tools=None) -> List[str]:
    return _get_functions_dict().keys()

def get_openai_tools(tools=None) -> List[dict]:
    
    if tools:
        return [convert_to_openai_tool(_get_functions_dict()[tool]) for tool in tools]
    else: 
        return [convert_to_openai_tool(f) for f in _get_functions_dict().values()]

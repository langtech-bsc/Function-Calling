from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools import tool
from typing import List
import json
from gradio_client import Client


client = Client("BSC-LT/VECTOR_STORE_EADOP")




# Define the retrieve_context function with an adjusted docstring
@tool
def retrieve_context(prompt: str) -> str:
    """
    Given a prompt, retrieves the context from vectorstore.
    """
    result = client.predict(
		prompt=prompt,
		num_chunks=2,
		api_name="/get-eadop-rag"
    )
    return result


@tool
def get_context(query: str) -> str:
    """
    Given a query, retrieves the context.
    """
    result = client.predict(
		prompt=query,
		num_chunks=2,
		api_name="/get-eadop-rag"
    )
    return result

@tool
def get_documents(query: str, n_chunks:int) -> str:
    """
    Given a query, retruns the documents relevants to the query.
    """
    result = client.predict(
		prompt=query,
		num_chunks=2,
		api_name="/get-eadop-rag"
    )
    return result


@tool
def sum(n1: int, n2: int) -> int:
    """
    Given two number return the sum of it.
    """
    return n1 + n2

@tool
def div(n1: int, n2: int) -> int:
    """
    Given two numbers, return their quotient..
    """
    return n1 / n2


def _get_functions_dict():
    return  {
        # "retrieve_context": retrieve_context,
        # "get_context": get_context,
        "get_documents": get_documents,
        # "sum": sum,
        # "div": div,
    }
    

def get_openai_tools_name(tools=None) -> List[str]:
    return _get_functions_dict().keys()

def get_tool_by_name(name):
    tools = _get_functions_dict()
    for key, tool in tools.items():
        if key == name:
            return tool
    return None

def get_openai_tools(tools=None) -> List[dict]:
    
    if tools:
        return [convert_to_openai_tool(_get_functions_dict()[tool]) for tool in tools]
    else: 
        return [convert_to_openai_tool(f) for f in _get_functions_dict().values()]

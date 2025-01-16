from datasets import load_dataset, Dataset, DatasetDict
from openai import OpenAI
from multiprocessing import Pool, Manager, cpu_count
from dotenv import load_dotenv
import os
from tools.retrieve_context import get_openai_tools, get_tool_by_name
from transformers import AutoTokenizer
import json
load_dotenv(".env")
HF_TOKEN = os.environ["HF_TOKEN"]
BASE_URL = os.environ["BASE_URL"]


client = OpenAI(
    base_url="http://localhost:8080/v1/", 
    api_key=HF_TOKEN
)

models = client.models.list()
model = models.data[0].id
print(model)
messages = [
#   {
#     "role": "system", 
#     "content": """Use a function to get documents or context relevate to the questions, if you have"""
#     },
    {
    "role": "user", 
    "content": """Tell me somthing about barcelona"""
    }]

tools = get_openai_tools()
chat_completion = client.chat.completions.create(
        model="tgi",
        messages=messages,
        stream=True,
        max_tokens=2000,
        temperature=0.4,
        tool_choice="auto",
        tools=tools,
        frequency_penalty=2.0
    )

arguments = ""
name = ""
text = ""

for chunk in chat_completion:
    if len(chunk.choices) > 0 and chunk.choices[0].delta.tool_calls and len(chunk.choices[0].delta.tool_calls) > 0 :
        call = chunk.choices[0].delta.tool_calls[0]
        if call.function.name:
            name=call.function.name
        if call.function.arguments:
            arguments += call.function.arguments

    elif chunk.choices[0].delta.content:
        text += chunk.choices[0].delta.content

print("Text:", text)
print("name, arguments:", name, arguments)
if name:
    if not arguments:
        arguments = "{}"

    json_arguments = json.loads(arguments)
    tool = get_tool_by_name(name=name)
    tool_ans = tool.invoke(input=json_arguments)
    content = f'[{{{name}: {tool_ans}}}]'

    print("Content:", content)
    messages.append({"role":"assistant", "tool_calls": [{"id": "call_FthC9qRpsL5kBpwwyw6c7j4k","function": {"arguments": arguments,"name": name},"type": "function"}]})
    messages.append({"role":"tool", "content": content, "tool_call_id": "call_FthC9qRpsL5kBpwwyw6c7j4k"})
    chat_completion = client.chat.completions.create(
            model="tgi",
            messages=messages,
            stream=True,
            max_tokens=2000,
            # tool_choice="auto",
            temperature=0.3,
            # tools=tools
        )

    text = ""
    arguments = ""
    name = ""
    for chunk in chat_completion:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.tool_calls and len(chunk.choices[0].delta.tool_calls) > 0 :
            call = chunk.choices[0].delta.tool_calls[0]
            if call.function.name:
                name=call.function.name
            if call.function.arguments:
                arguments += call.function.arguments

        elif chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content

    print("==============================")
    print("name, arguments:", name, arguments)
    print("text:", text)

# chat_completion = client.chat.completions.create(
#         model="tgi",
#         messages=messages,
#         stream=False,
#         max_tokens=2000,
#         temperature=0.3,
#         tools=tools,
#         tool_choice="auto",
#     )
# print(chat_completion.choices[0].message.tool_calls[0])

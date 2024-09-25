from datasets import load_dataset, Dataset, DatasetDict
from openai import OpenAI
from multiprocessing import Pool, Manager, cpu_count
from dotenv import load_dotenv
import os
from tools.retrieve_context import get_openai_tools
from transformers import AutoTokenizer

load_dotenv(".env")
HF_TOKEN = os.environ["HF_TOKEN"]
BASE_URL = os.environ["BASE_URL"]

tools = get_openai_tools()

client = OpenAI(
    base_url=BASE_URL + "/v1/", 
    api_key=HF_TOKEN
)



messages = [{"role": "user", "content": "¿Donde está barcelona?"}]

tokenizer = AutoTokenizer.from_pretrained("BSC-LT/checkpoint_7b_4epoch_rag_conversational_func_call")

prompt = tokenizer.apply_chat_template(conversation=messages, tools=tools, add_generation_prompt=True, tokenize=False)
print(prompt)
chat_completion = client.completions.create(
        model="tgi",
        prompt=prompt,
        stream=False,
        max_tokens=2000,
        temperature=0.3,
    )

print(chat_completion.choices[0].text)

# print("===================")

# chat_completion = client.chat.completions.create(
#         model="tgi",
#         messages=messages,
#         stream=False,
#         max_tokens=2000,
#         temperature=0.3,
#         tools=tools,
#         tool_choice="auto"
#     )

# print(chat_completion.choices[0].message)
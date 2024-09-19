from datasets import load_dataset, Dataset, DatasetDict
from openai import OpenAI
from multiprocessing import Pool, Manager
import json
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
from tools.retrieve_context import get_openai_tools

env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(env_path)
HF_TOKEN = os.environ["HF_TOKEN"]
BASE_URL = os.environ["BASE_URL"]

client = OpenAI(
    base_url=BASE_URL + "/v1/", 
    api_key=HF_TOKEN
)

tools = get_openai_tools()

_prompt = """
You are tasked with converting the following structured data into a conversation that uses function calling to retrieve relevant information from a tool. The conversation must follow this pattern:

1. The user asks a question.
2. The assistant makes a function call to retrieve relevant context from a vector store, providing an optimized prompt.
3. The tool returns the context based on the function call.
4. The assistant provides the response to the user, based on the returned context.
5. The conversation continues with additional questions from the user and answers from the assistant, some based on the context and others outside of it.
6. The language must preserve, if user askes question in spanish, all other questions and answers must be in spanish, same with other languages.

### Instructions:
1. Use the following placeholders:
   - **INSTRUCTION**: The user's initial question or instruction.
   - **CONTEXT**: The relevant context returned from the vector store, which helps the assistant answer.
   - **RESPONSE**: The assistant’s answer based on the context.
2. The output should be structured as a list of conversation turns, where each turn is a dictionary with the keys `role` (which can be "user", "assistant", "tool", or "system") and `value` (which can be plain text or a tool call or response).

### Conversation Structure Example:
1. The user asks a question.
2. The assistant makes a function call using `<tool_call>` with the function name and it's arguments.
3. The tool returns the context inside `<tool_response>`.
4. The assistant provides the response based on the returned context.
5. The conversation then continues with 5–10 additional question-answer pairs, where some questions are directly related to the context, while others are not.
6. The conversation should end with the assistant’s response.
7. Ensure that the JSON data used and generated throughout the process is valid and properly formatted.

Here you have data:

INSTRUCTION: {instruction}

CONTEXT: {context}

RESPONSE: {response}
---

### Format:
This is just an example; you should generate your data based on the following format.
- Do not include additional keys—only role and value should be used.
- The conversation should include 4–10 additional question-answer pairs.
- Ensure that at least one random question not unrelated to the context, and respond to it using your knowledge.
    Categories: [Fact-based, Opinion-based, Creative Thinking, Problem-Solving, Hypothetical, Trivia, Ethical, Riddle, Personal Reflection, Goal-Oriented, People, Rivers, Food, Mountains, Animals, Countries, Continents, Inventions, Literature, Movies, Music, Science, Sports, Languages, History, Technology, Space, Currencies, Landmarks, Flags]
    For random questions not unrelated to the context, please randomly select a category, then randomly generate a question from that category.
- The json format must be correct.

[
    {{
        "role": "user", 
        "value": "{{INSTRUCTION}}"
    }},
    {{
        "role": "assistant", 
        "value": "<tool_call>\n{{'arguments': {{'prompt': '{optimized_instruction}', 'function_name': 'retrieve_context'}}}}\n</tool_call>"
    }},
    {{
        "role": "tool", 
        "value": "<tool_response>{{CONTEXT}}</tool_response>"
    }},
    {{
        "role": "assistant", 
        "value": "{{RESPONSE}}"
    }},
    {{
        "role": "user", 
        "value": "Follow-up question based on the response or context"
    }},
    {{
        "role": "assistant", 
        "value": "Assistant answer based on context"
    }},
    {{
        "role": "user", 
        "value": "Follow-up question based on the response or context, or unrelated to question or context"
    }},
     {{
        "role": "assistant", 
        "value": "Assistant answer related or not related to the context, depending on the question"
    }},
    .
    .
    .
    {{
        "role": "user",
        "value": "Unrelated or related question"
    }},
    {{
        "role": "assistant", 
        "value": "Assistant answer related or not related to the context, depending on the question"
    }}
]
"""



_prompt2 = """
### You are tasked with improving a user query to improve its effectiveness in retrieving relevant documents from a vector store. Follow these guidelines:

# Context: {context}

# Follow these instruction to improve the query:
- Use the provided context solely as a reference to improve the query, not as a basis for generating new questions.
- Do not add extra infromation in the query from the context or based on your knowladge.
- Ensure that all important words, names, and technical terms (key tokens) are preserved. Do not replace or modify these tokens unless it significantly improves the query. If necessary, improve the phrasing around these tokens while keeping their meaning intact.
- If the original query lacks context that could improve document retrieval, add appropriate context without changing the core meaning or intent.
- Ensure that the improved query conveys the same purpose and meaning as the original. The original query’s focus and user intent should remain unchanged.
- Do not over-optimize by modifying words unnecessarily. Focus on clarity, but keep key terms untouched unless truly needed for improving the query.
- The language must match the original query’s language (e.g., [En quant cresqué el consum de cigarrets per capita a Alemanya], the language is Catalan)
- If the query is ambiguous or unclear, clarify it while preserving the original meaning, ennsuring it doesn't mislead or change the intent.

Orignal query: {instruction}

Follow this format (Give the improved query, but nothing else, without any extra lines or tags):
Improved_query: <Here the imporved query> 
"""

def save(json_data: dict):
    with open("tmp.json", "w") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

def get_conversations(context, response, instruction):
    prompt = _prompt2.format(instruction=instruction, context=context)
    messages = [{"role": "user", "content": prompt}]

    try:
        chat_completion = client.chat.completions.create(
        model="tgi",
        messages=messages,
        stream=False,
        max_tokens=500,
        temperature=0.3,
        # presence_penalty=1.2,
    )
    except Exception as err:
        print(err)
        raise err
    
    optimized_instruction = chat_completion.choices[0].message.content.split("Improved_query:")[-1].strip()
    prompt = _prompt.format(context=context, response=response, instruction=instruction, optimized_instruction=optimized_instruction)
    messages = [{"role": "user", "content": prompt}]
    try:
        chat_completion = client.chat.completions.create(
        model="tgi",
        messages=messages,
        stream=False,
        max_tokens=4000,
        temperature=0.3,
        # presence_penalty=1.2,
    )
    except Exception as err:
        print(err)
        raise err
    
    text = chat_completion.choices[0].message.content

    try:
        data = json.loads(text)
        data[0]["value"] = instruction
        data[2]["value"] = context
        data[3]["value"] = response
    except Exception as err:
        print(text)
        print("^" * 20)
        print(err)
        return None
    
    return data

def process_row(row, shared_dict, lock):
    conversations = get_conversations(row["context"], row["response"], row["instruction"])
    if conversations:
        with lock:
            shared_dict[row["id"]] = conversations
            if len(shared_dict.keys()) % 10 == 0:
                print("done: ", len(shared_dict.keys()))
                save(dict(shared_dict))  # Convert to regular dict for saving
    

def process_split(pdata, shared_dict, lock):
    rows = [row for _, row in pdata.iterrows() if row["id"] not in shared_dict]
    print(len(rows))
    with Pool(processes=10) as pool:
        pool.starmap(process_row, [(row, shared_dict, lock) for row in rows])

    save(dict(shared_dict))  # Convert to regular dict for saving
    pdata['conversations'] = pdata['id'].map(pd.Series(dict(shared_dict)))
    pdata['tools'] = tools
    return Dataset.from_pandas(pdata)

if __name__ == "__main__":
    json_conversations_list = {}
    try:
        with open("tmp.json", "r") as file:
            json_conversations_list = json.loads(file.read())
    except:
        pass
    new_dataset = {}
    dataset = load_dataset("projecte-aina/RAG_Multilingual", token=HF_TOKEN)

    with Manager() as manager:
        shared_dict = manager.dict()
        shared_dict.update(json_conversations_list)
        lock = manager.Lock()  # Use manager's Lock

        for split in ["train", "validation", "test"]:
            pdata = dataset.data[split].to_pandas()
            print(split, end=": ")
            new_dataset[split] = process_split(pdata, shared_dict, lock)

    new_dataset = DatasetDict(new_dataset)

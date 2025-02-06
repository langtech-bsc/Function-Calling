import gradio as gr
from gradio import ChatMessage

import json
from openai import OpenAI
from tools import tools, oitools
from dotenv import load_dotenv
import os
load_dotenv(".env")
HF_TOKEN = os.environ["HF_TOKEN"]
BASE_URL = os.environ["BASE_URL"]

SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant designed to assist users with a hotel booking and information system. Your role is to provide detailed and accurate information about the hotel, including available accommodations, facilities, dining options, and reservation services. You can check room availability, assist with bookings, modify or cancel reservations, and answer general inquiries about the hotel.  

Maintain clarity, conciseness, and relevance in your responses, ensuring a seamless user experience. Always respond in the same **language as the user’s query** to preserve their preferred language.
"""

client = OpenAI(
	    base_url=BASE_URL + "/v1", 
	    api_key=HF_TOKEN
    )


def complation(history, model, system_prompt, tools=None):
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        if type(msg) == dict:
            msg = ChatMessage(**msg)
        if msg.role == "assistant" and len(msg.options) > 0 and msg.options[0]["label"] == "tool_calls":
            tools_calls = json.loads(msg.options[0]["value"])
            messages.append({"role": "assistant", "tool_calls": tools_calls})
            messages.append({"role": "tool", "content": msg.content})
        else:
            messages.append({"role": msg.role, "content": msg.content})
    
    if not tools:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=1000,
            temperature=0.4,
            frequency_penalty=1,
            # stop=["<|em_end|>"],
            extra_body = {
                "repetition_penalty": 1.1,
            }
        )
    return client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=1000,
            temperature=0.4,
            tool_choice="auto",
            tools=tools,
            frequency_penalty=1,
            # stop=["<|em_end|>"],
            extra_body = {
                "repetition_penalty": 1.1,
            }
        )

def respond(
    message:any,
    history:any,
    additional_inputs,
):
    try:   
        models = client.models.list()
        model = models.data[0].id
    except Exception as err:
        gr.Warning("The model is initializing. Please wait; this may take 5 to 10 minutes ⏳.", duration=20)
        raise err

    response = ""
    arguments = ""
    name = ""
    history.append(
        ChatMessage(
            role="user",
            content=message,
        )
    )
    completion = complation(history=history,  tools=oitools, model=model, system_prompt=additional_inputs)
    appended = False
    for chunk in completion:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.tool_calls and len(chunk.choices[0].delta.tool_calls) > 0 :
            call = chunk.choices[0].delta.tool_calls[0]
            if call.function.name:
                name=call.function.name
            if call.function.arguments:
                arguments += call.function.arguments

        elif chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
            if not appended:
                history.append(
                   ChatMessage(
                        role="assistant",
                        content="",
                    )
                )
                appended = True
                
            history[-1].content = response
            yield history[-1]
    
    if not arguments:
        arguments = "{}"
    if name:
        json_arguments = json.loads(arguments)
        result = f"💥 Error using tool {name}, tools doesn't exists"
        if tools.get(name):
            result = str(tools[name].invoke(input=json_arguments))
            result = json.dumps({name: result}, ensure_ascii=False)
        history.append(
                ChatMessage(
                    role="assistant",
                    content=result,
                    metadata= {"title": f"🛠️ Used tool '{name}', arguments: {json.dumps(json_arguments, ensure_ascii=False)}"},
                    options=[{"label":"tool_calls", "value": json.dumps([{"id": "call_FthC9qRpsL5kBpwwyw6c7j4k","function": {"arguments": arguments,"name": name},"type": "function"}])}]
                )
            )
        yield history[-1]

        completion = complation(history=history, tools=oitools, model=model, system_prompt=additional_inputs)
        result = ""
        appended = False
        for chunk in completion:
            result += chunk.choices[0].delta.content
            if not appended:
                history.append(
                   ChatMessage(
                        role="assistant",
                        content="",
                    )
                )
                appended = True
                
            history[-1].content = result
            yield history[-2:]

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
if __name__ == "__main__":
    system_prompt = gr.Textbox(label="System propmt", value=SYSTEM_PROMPT_TEMPLATE, lines=3)
    demo = gr.ChatInterface(respond, type="messages", additional_inputs=[system_prompt])
    demo.launch()

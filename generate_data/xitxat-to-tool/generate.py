import json
import os
import torch.distributed as dist
from openai import OpenAI
import random
import re
import argparse
from concurrent.futures import ThreadPoolExecutor
import torch
import time
import threading

META_DIRECTORY = "xitxat_metadata"

SYSTEM_TOOLS="""
You are an expert AI system responsible for generating structured synthetic data for function calling in an LLM model chat, where user information will be gathered directly by the LLM, eliminating the need to create functions for this purpose.  
Your primary goal is to analyze a given topic and create a set of **10 to 20 distinct functions** representing actionable operations.  

**Guidelines for Generating Functions:**  

1. **Function Characteristics:**  
    - Each function must be **clear, unambiguous**, and represent a specific action or intent.  
    - Avoid redundancy by ensuring all functions are distinct and serve a unique purpose.  
    - Do not create trivial or overly generic functions such as `end_conversation`, `start_conversation`, or `additional_assistance`.  
    - Avoid creating functions solely to gather user information; the LLM will handle this directly.  
    - Include meaningful and specific parameters to enhance function utility.  
    - Functions without parameters can be included if they retrieve static information or execute self-contained actions.  
    - Do not create functions solely for collecting user information, as this should be obtained directly through the chat.

2. **Domain-Specific Context:**
    - Functions must capture all significant exchanges in the conversation.
    - Avoid creating functions for questions that can be directly answered by an LLM.
    - Reflect the domain-specific purpose (e.g., customer service, telecom, and internet setup)
    - Design functions to maintain continuity across multi-turn dialogues.

3. **Use of Enums:**  
    - If a parameter accepts a specific, finite set of values, use an `enum` to define those values as valid inputs.  
    - Do not create enums for parameters accepting arbitrary or free-form data (e.g., user-provided text or unbounded numerical values).  

4. **Function Naming and Clarity:**  
    - Use descriptive, purpose-driven names (e.g., `configure_lighting_schedule`, `activate_security_alarm`).  
    - Ensure descriptions clearly explain the function’s purpose in one to two sentences.  

5. **Parameterization:**  
    - Define input parameters in the `properties` section with their type, description, and example values.  
    - Clearly indicate required parameters using the `required` array.  
    - Set `additionalProperties` to `false` to prevent unspecified inputs.  

6. **Output Structure:**  
    - Use this JSON structure for each function:  
      {
        "type": "function",
        "function": {
          "name": "<function_name>",
          "description": "<brief description of the function's purpose>",
          "parameters": {
            "type": "object",
            "properties": {
              "<parameter_name>": {
                "type": "<parameter_type>",
                "description": "<parameter description>",
                "enum": ["<value1>", "<value2>", ...]  // Optional: Include only when necessary
              }
            },
            "required": ["<parameter1>", "<parameter2>", ...],
            "additionalProperties": false
          }
        }
      }
      

7. **Quality Standards:**  
    - Functions must have practical utility in the specified topic domain. Avoid nonsense or generic functions.  
    - Ensure no significant user intents or actions are overlooked.  
    - Validate all JSON for correctness and completeness.  

8. **Output Requirements:**  
    - Generate 10 to 20 unique functions based solely on the given topic.  
    - The response must be strictly JSON with no additional explanations, headers, or formatting.  

9. **Output Requirements:**
    - Ensure no significant intent or action is overlooked.
    - Only generate functions that require additional user input or actions that cannot be inferred automatically by the LLM.

10. **Example Input:**
    ```
    **Conversation**:
    agent: Bon dia! Com el/la puc ajudar?
    usuari: Hola mira es que acabo d'arribar a un pis nou, i aquí no hi ha internet ni hi ha res. Volia saber a veure quines ofertes teníeu per posar wifi aquí a casa.
    agent: Cap problema. Tenim tres opcions: telèfon mòbil + internet (30 euros mensuals), telèfon mòbil + telèfon fixe + internet (40 euros mensuals) o televisió + internet (45 euros mensuals). L'interessaria cap d'aquestes combinacions?
    usuari: Ah doncs mira teniu bones ofertes veig. Això dels 45 euros per la tele i l'internet quant de temps seria, vull dir, hi ha uns mesos mínim que ho hagi de contractar o puc donar-me de baixa quan sigui?
    agent: Sí. Serien 3 mesos mínim de permanència.
    usuari: Està molt bé això, m'interessa si si.
    agent: Quin és el seu nom i cognoms?
    usuari: soc l'Anna Tur Marí.
    ...

    **Topic**:
    Smart Home Automation (Lighting, Temperature, and Security Control)
    ```

11. **Example Output:**

[
    {
        "type": "function",
        "function": {
            "name": "play_song",
            "description": "Plays a song from a specified music library or platform.",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {
                        "type": "string",
                        "description": "The name of the song to be played."
                    },
                    "artist_name": {
                        "type": "string",
                        "description": "The name of the artist who performed the song. Optional.",
                        "default": ""
                    },
                    "platform": {
                        "type": "string",
                        "description": "The music platform to play the song from (e.g., Spotify, YouTube, Apple Music).",
                        "enum": ["spotify", "youtube", "apple_music", "amazon_music"],
                        "default": "spotify"
                    }
                },
                "required": ["song_name"],
                "additionalProperties": false
            }
        }
    }
]

Generate 10 to 20 unique functions based solely on the provided topic. The example conversation is included only to demonstrate the flow of dialogue and should not influence the function generation process.

Return **only** JSON adn must be complete. Do not include explanations, feedback, or any additional text, not even code block (```json)..
"""

USER_TOOLS = """
**Conversation**:
{conversation}

**Topic**:
{topic}
"""

SYSTEM_TOPICS = """You are an advanced AI system tasked with generating **creative and distinct conversation topics** for synthetic data creation, based on a **provided sample conversation**. Your goal is to analyze the conversation and generate **10 distinct and unambiguous topics** inspired by the conversation's context.
These topics should explore a **variety of user needs and scenarios** but should not directly replicate or mimic the example topics provided below.


**Follow these guidelines:**

1. **Uniqueness:**
    - Ensure **each topic is unique** with no overlap or redundancy.
    - Topics should be **distinct from one another**, exploring different user intents, actions, and needs.
    - Avoid topics that simply restate the conversation or offer the same semantic meaning in different phrasing.

2. **Topic Diversity:**
    - Suggest topics that explore **a variety of user needs and scenarios**, avoiding repetitive focus on the same actions or inquiries.
    - Explore different user scenarios that could use the provided conversation for inspiration, but make sure each topic has a **distinct angle** or purpose.

3. **Extend Beyond the Conversation:**
    - While the topics should be inspired by the provided conversation, feel free to **extend the scope** and consider other relevant situations or user actions that could emerge in a **real-world scenario** related to the conversation's domain.

4. **Output Format:**
    - Provide the topics as a **list** of concise descriptions or questions without any numbering.
    - Ensure no topic is repeated or too similar in purpose.
    - Don't provide anything else other than topics.

5. **Domain Uniqueness:**
    - Ensure that each topic corresponds to a **unique domain**, with no domain being repeated for multiple topics. However, it is acceptable to have **the same domain for two topics** if the specific scenarios or user needs differ.

**Input Structure:**
    
agent: Bon dia! Com el/la puc ajudar?
    usuari: Hola mira es que acabo d'arribar a un pis nou, i aquí no hi ha internet ni hi ha res. Volia saber a veure quines ofertes teníeu per posar wifi aquí a casa.
    agent: Cap problema. Tenim tres opcions: telèfon mòbil + internet (30 euros mensuals), telèfon mòbil + telèfon fixe + internet (40 euros mensuals) o televisió + internet (45 euros mensuals). L'interessaria cap d'aquestes combinacions?
    usuari: Ah doncs mira teniu bones ofertes veig. Això dels 45 euros per la tele i l'internet quant de temps seria, vull dir, hi ha uns mesos mínim que ho hagi de contractar o puc donar-me de baixa quan sigui?
    agent: Sí. Serien 3 mesos mínim de permanència.
    usuari: Està molt bé això, m'interessa si si.
    agent: Quin és el seu nom i cognoms?
    usuari: soc l'Anna Tur Marí.


**Example Topics (for reference only, do not directly generate from these):**
```
    Spotify (Music Recommendations and Playback Control)
    Smart Home Automation (Lighting, Temperature, and Security Control)
    E-commerce (Product Recommendations and Order Tracking)
    Travel and Booking (Flight and Hotel Reservation)
    Healthcare (Appointment Scheduling and Medication Reminders)
    Customer Support (Ticket Generation and Inquiry Resolution)
    Financial Services (Bank Transactions and Investment Advice)
    Learning and Education (Personalized Learning Paths and Assessments)
    News and Media (Real-time News Updates and Summaries)
    Social Media (Post Creation, Scheduling, and Notifications)
``` 

Generate 10 topics based on the provided conversation that meet the above requirements, considering the domain and extending beyond the conversation to related real-world scenarios.
Ensure that each topic corresponds to a **unique domain**.
"""


USER_TOPICS = """
{conversation}
"""

SYSTEM_CONVERSATION = """
You are an advanced AI system tasked with generating accurate and contextually relevant conversations based on the provided topic and functions.
Your primary goal is to create realistic and logically coherent dialogues that strictly adhere to the input structure and ensure the JSON format is always correct, complete, and error-free.


**Follow these guidelines:**
1. **Topic**: The user will provide a topic (e.g., "internet plan inquiry" or "booking installation for internet").

2. **Functions**: You will be provided with a list of functions that you need to call within the conversation.

3. **Role Transitions**:
   - After a message from the **user**, only the **agent** (gpt) can respond.
   - After a message from the **agent**, the next message can be either from the **user** or a **tool**, depending on whether a function is invoked.
   - After a message from the **tool**, the next response must come from the **agent** (gpt).
   - **Consecutive messages** from the same role (e.g., user → user or agent → agent) are **not allowed**.

4. **Function Calls**:
   - When the agent (gpt) needs to retrieve information or perform an action, it should invoke the appropriate function using a `<tool_call>` tag in its response.
   - Function calls must specify the correct function name and parameters required for the action.
   - **Missing Parameters**: 
      - If the user's message does not contain enough information to proceed with the function call, the agent must explicitly ask the user for the required details before proceeding with the function.
      - Once the required information is provided, the agent can include it in the function call.

5. **Structure**: The conversation must follow this structure for consistency:
   - User's message: `{ "from": "human", "value": "..." }`
   - Agent's message: `{ "from": "gpt", "value": "..."` (if applicable)
   - User's message: `{ "from": "human", "value": "..." }` (if applicable)
   - Agent's message: `{ "from": "gpt", "value": "", tool_calls:[{...}]` (if applicable)
   - Tool response: `{ "from": "tool", "value": {...}, }` (if applicable)
   - Agent's message: `{ "from": "gpt", "value": "..." }` (if applicable)

6. **LANGUAGE_TAG Language**: The entire conversation, including user messages and agent responses, must be in **LANGUAGE_TAG**. Tool responses could be either in English or **LANGUAGE_TAG**.

7. **Tool Calls in Context**:  
    - Use tools only when necessary to retrieve data, perform actions, or clarify user requests.  
    - Ensure tool responses are concise, result-oriented, and aligned with the tool’s purpose. They should provide structured and actionable data, not conversational or dialogue-style outputs. For example, a tool retrieving policies should return a comprehensive and well-organized list of policies, enabling the LLM to construct a clear and informative response for the user.
    - Pass parameters to tools only if they are explicitly present in the user's input or derived from the context. If key information is missing, prompt the user to provide it before calling the tool.  
    - The LLM must process tool outputs to craft responses that are clear, relevant, and actionable for the user. The final response should integrate insights from the tool with the LLM's contextual understanding, ensuring the user receives a natural and comprehensive answer based on the tool’s data.  

8. **Realism**: Ensure the conversation feels natural and follows a realistic flow based on the topic provided by the user.

9. **Formatting**: Present the entire conversation in properly formatted JSON, with clear and consistent indentation.

10. **Missing Information Handling**: 
    - Before invoking a function, verify that all required parameters are present.
    - If any parameter is missing:
    - Prompt the user to provide the missing details in a natural and contextually appropriate way.
    - Once the user responds with the necessary information, proceed with the function call.

11. **Engagement**: The conversation must contain **at least 10 user questions** across the dialogue, ensuring active user engagement.


**Example of the structure (only as a guide for the conversation, do not base your conversation on this directly):**
```json
[
    { "from": "human", "value": "Can you tell me the population of Tokyo?" },
    { "from": "gpt", "value": "", "tool_calls": [ { "arguments": { "query": "current population of Tokyo" }, "name": "browser_search" } ] },
    { "from": "tool", "value": {"browser_search": {"population": "approximately 14 million"}} },
    { "from": "gpt", "value": "The population of Tokyo is approximately 14 million." },
    { "from": "human", "value": "What's the distance between New York and Tokyo?" },
    { "from": "gpt", "value": "", "tool_calls": [ { "arguments": { "location1": "New York", "location2": "Tokyo" }, "name": "calculate_distance" } ] },
    { "from": "tool", "value": {"calculate_distance": {"distance": "6,750 miles"}} },
    { "from": "gpt", "value": "The distance between New York and Tokyo is about 6,750 miles." },
    { "from": "human", "value": "What's the time difference between these two cities?" },
    { "from": "gpt", "value": "", "tool_calls": [ { "arguments": { "city1": "New York", "city2": "Tokyo" }, "name": "time_zone_comparison.compare" } ] },
    { "from": "tool", "value": {"time_zone_comparison.compare": "14 hours ahead"} },
    { "from": "gpt", "value": "Tokyo is 14 hours ahead of New York." },
    { "from": "human", "value": "What is the current time?" },
    { "from": "gpt", "value": "Could you please specify the city you are referring to?" },
    { "from": "human", "value": "In Tokyo." },
    { "from": "gpt", "value": "", "tool_calls": [ { "arguments": { "city": "Tokyo" }, "name": "fetch_time" } ] },
    { "from": "tool", "value": {"fetch_time": "3:00 PM"} },
    { "from": "gpt", "value": "The current time in Tokyo is 3:00 PM." },
    { "from": "human", "value": "Can you tell me about the weather there right now?" },
    { "from": "gpt", "value": "", "tool_calls": [ { "arguments": { "city": "Tokyo" }, "name": "get_temprature" } ] },
    { "from": "tool", "value": {"get_temprature": {"temprature": "68°F", "condition": "partly cloudy"}} },
    { "from": "gpt", "value": "The current weather in Tokyo is 68°F and partly cloudy." },
    { "from": "human", "value": "Can you help me book a flight to Tokyo?" },
    { "from": "gpt", "value": "Sorry, I can't help with that. I don't have the ability to book flights." },
    { "from": "human", "value": "How do I get to Tokyo Tower from Shibuya?" },
    { "from": "gpt", "value": "", "tool_calls": [ { "arguments": { "from": "Shibuya", "to": "Tokyo Tower" }, "name": "get_directions" } ] },
    { "from": "tool", "value": {"get_directions": ["Take the JR Yamanote Line from Shibuya to Hamamatsucho.", "Transfer to the Oedo Line and go to Akabanebashi.", "Walk 5 minutes to Tokyo Tower."]} },
    { "from": "gpt", "value": "To get to Tokyo Tower from Shibuya: Take the JR Yamanote Line from Shibuya Station to Hamamatsucho Station, then transfer to the Oedo Line to Akabanebashi Station. Tokyo Tower is a 5-minute walk from there." },
    { "from": "human", "value": "What’s a good place to visit in Tokyo?" },
    { "from": "gpt", "value": "", "tool_calls": [ { "arguments": { "query": "popular tourist attractions in Tokyo" }, "name": "browser_search" } ] },
    { "from": "tool", "value": {"get_directions": { "place": "Tokyo Tower", "description": "An iconic landmark offering panoramic city views." }}},
    { "from": "gpt", "value": "A great place to visit in Tokyo is Tokyo Tower, an iconic landmark offering panoramic city views." },
    { "from": "human", "value": "How much does a ticket to Tokyo Tower cost?" },
    { "from": "gpt", "value": "", "tool_calls": [ { "arguments": { "query": "Tokyo Tower ticket price" }, "name": "browser_search" } ] },
    { "from": "tool", "value": {"browser_search": {"price": "$10 for adults"}} },
    { "from": "gpt", "value": "The ticket price for Tokyo Tower is $10 for adults." },
    { "from": "human", "value": "Can I book tickets online?" },
    { "from": "gpt", "value": "", "tool_calls": [ { "arguments": { "query": "Tokyo Tower online ticket booking" }, "name": "browser_search" } ] },
    { "from": "tool", "value": {"browser_search": {"booking_available": true, "link": "https://tokyotower.jp"}}},
    { "from": "gpt", "value": "Yes, you can book tickets for Tokyo Tower online at [Tokyo Tower's official website](https://tokyotower.jp)." }
]

Return **only** JSON adn must be complete. Do not include explanations, feedback, or any additional text, not even code block (```json).
"""

USER_CONVERSATION = """"
**Topic**: {topic}.
**Functions**:
```json
{functions}
```
"""

FIX_JSON_SYSTEM = """
You are an advanced AI system tasked with repairing and optimizing broken JSON data. Below, you will find a faulty JSON structure containing errors in formatting, structure, or content.

### Your Objectives:
1. **Correct Errors**: Fix all syntax, formatting, and structural issues to ensure the JSON is valid.
2. **Logical Coherence**: Analyze the content to maintain logical consistency and integrity. If any keys or values appear incomplete or misplaced, reorganize them appropriately.
3. **Completeness**: Check for missing fields or incomplete data. If applicable:
   - Remove the last row if the list is incomplete or inconsistent.
   - Ensure all remaining entries form a coherent and valid JSON structure.
4. **Enhance Structure**: Improve formatting, organization, and any logical inconsistencies while preserving the data’s intent.

Return **only** the corrected and improved JSON. Do not include explanations, feedback, or any additional text.
"""

FIX_JSON_USER = """
Broken JSON:
```
{data}
```
Return **only** the corrected and improved JSON. Do not include explanations, feedback, or any additional text.
"""


def save_metadata(json_data, id, path):
    # Create the directory path
    file_path = os.path.join(path, f"{id}.json")
    
    # Ensure all parent directories exist
    parent_dir = os.path.dirname(path)
    if parent_dir:  # Avoid trying to create directories for paths without parent directories
        os.makedirs(parent_dir, exist_ok=True)
    
    # Append the JSON data as a single line
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(json_data, indent=2, ensure_ascii=False))

def read_metadata(id, path):
    # Construct the file path
    file_path = os.path.join(path, f"{id}.json")
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            return None
        
        # Read the JSON line from the file
        with open(file_path, 'r', encoding='utf-8') as file:
            line = file.read()  # Read the single JSON line
            return json.loads(line)
    except:
        return None

def read_json(path):
    """
    Reads a JSON file or JSON Lines file and returns the data as a list of dictionaries.
    Args:
        path (str): Path to the JSON or JSONL file.
    Returns:
        list: A list of dictionaries representing the JSON data.
    """

    with open(path, 'r', encoding='utf-8') as file:
        # Try to load as a normal JSON list
        try:
            return json.loads(file.read())
        except:
            data = [json.loads(data) for data in file.readlines(file)]
            return data       
    
def generate_response(messages, client):
    chat_completion = client.chat.completions.create(
                model="tgi",
                messages=messages,
                stream=False,
                max_tokens=4000,
                temperature=0.4,
                extra_body={
                }
            )
    
    response = chat_completion.choices[0].message.content
    return response

languages = ["Catalan", "Spanish", "English"]

def generate_data_tools_and_conversation(id, topic, conversation, client):
    functions = None
    count = 0
    while (not functions):
        if count >= 4:
            return None
            # raise ValueError("Model is not capable of generating correct json.")
        count += 1
        messages = [{"role":"system", "content": SYSTEM_TOOLS}]
        messages.append({"role":"user", "content": USER_TOOLS.format(conversation=conversation, topic=topic)})
        response = generate_response(messages=messages, client=client)
        try:
            functions = json.loads(response)
        except:
            print("\tError: generating functions")

    final_conversation = None
    count = 0
    while (not final_conversation):
        if count >= 4:
            return None
        count += 1
        messages = [{"role":"system", "content": re.sub(r'LANGUAGE_TAG', random.choice(languages), SYSTEM_CONVERSATION)}]
        messages.append({"role":"user", "content": USER_CONVERSATION.format(topic=topic, functions=json.dumps(functions))})
        response = generate_response(messages=messages, client=client)
        try:
            final_conversation = json.loads(response)
        except:
            print("\tError: generating conversation")
    return functions, final_conversation
def progress_reporter(progress_tensor, data_path, stop_event):
    """
    Thread function to print progress periodically.
    """
    total_rows = int(read_json(data_path)["stats"]["total"])
    total = (total_rows*10)
    while not stop_event.is_set():
        total_progress = progress_tensor.sum().item()
        progress_percentage = int(100 * total_progress / total)

        # current_time = time.time()
        # time_took = round((current_time - initial_time)/60, 2)
        # initial_time = current_time
        
        bar_length = 50  # Length of the progress bar
        progress = int(progress_percentage / 2)  # Scale percentage to bar length
        bar = "#" * progress + "-" * (bar_length - progress)
        # Print the progress bar
        print(f"[{bar}] {progress_percentage}%", end='\r')
    
        time.sleep(2)  # Print every second


def generate_data(rows, client, metadata_dir, progress_tensor, rank):
    for j, row in enumerate(rows):
        conversation = ""
        id = row["id"]
        print(id + ": " + str(j))
        json_data = read_metadata(id, metadata_dir)
        if not json_data:
            json_data = {"id": id}
            for message in row["frases"]:
                text = message["text"]
                actor = message["actor"]
                conversation += f"{actor}: {text}\n"
            json_data["conversation"] = conversation
            save_metadata(json_data=json_data, id=id, path=metadata_dir)

        if not json_data.get("topics"):
            messages = [{"role":"system", "content": SYSTEM_TOPICS}]
            messages.append({"role":"user", "content": USER_TOPICS.format(conversation=json_data["conversation"])})
            response = generate_response(messages=messages, client=client)
            topics = [topic.strip().strip("* ").strip("- ").strip(f"{i}. ") for i, topic in enumerate(response.split("\n")) if topic.strip()]
            json_data["topics"] = topics
            save_metadata(json_data=json_data, id=id, path=metadata_dir)


        if not json_data.get("rows"):
           json_data["rows"] = {}

        topics_to_process = [topic for topic in json_data["topics"] if not json_data["rows"].get(topic)]
        if len(topics_to_process) == 0:
            print("Skiping due to already exists:\t", id)
            progress_tensor[rank] = j * 10 + (10 - len(topics_to_process))
            dist.all_reduce(progress_tensor, op=dist.ReduceOp.MAX)
            continue

        # No Concurrent
        for i, topic in enumerate(topics_to_process):
            result = generate_data_tools_and_conversation(id, topic, conversation, client)
            if result:
                tools, conversation = result  # Unpack the returned list
                json_data["rows"][topic] = { "tools": tools, "conversation": conversation }
                save_metadata(json_data=json_data, id=id, path=metadata_dir)
            
            progress_tensor[rank] = j * 10 + (10 - len(topics_to_process) + i+1)
            dist.all_reduce(progress_tensor, op=dist.ReduceOp.MAX)
            
        # Concurrent
        # with ThreadPoolExecutor() as executor:
        #     # print("Proceeding with", len(topics_to_process), "topics:\t", id)
        #     results = list(executor.map(lambda topic: generate_data_tools_and_conversation(id, topic, conversation, client), topics_to_process))
        
        # for i, result in enumerate(results):
        #     if not result:
        #         continue
        #     topic = topics_to_process[i]  # Get the topic from the filtered list
        #     tools, conversation = result  # Unpack the returned list
        #     json_data["rows"][topic] = { "tools": tools, "conversation": conversation }

        
        

def process_chunk(input_path, client, metadata_dir, rank, world_size, progress_tensor):
    """
    Process a portion of the dataset based on the worker's rank using PyTorch distributed processing.
    Args:
        file_path (str): Path to the input dataset file.
        rank (int): Rank of the current process.
        total_workers (int): Total number of workers.
    """
    # Load the dataset
    dataset = read_json(input_path)
    domains = list(dataset["stats"]["domains"].keys())
    total = len(domains)
    
    # Calculate start and end indices for this worker

    chunk_size = total // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else total

    domains = domains[start_idx:end_idx]

    for domain in domains:
        generate_data(rows=dataset["xats"][domain], client=client, metadata_dir=metadata_dir, progress_tensor=progress_tensor, rank=rank)


def metadata_to_dataset(metadata_dir, output_path):
    json_files = [f for f in os.listdir(META_DIRECTORY) if f.endswith('.json')]
    results = []
    for json_file in json_files:
        file_path = os.path.join(metadata_dir, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            for topic, row in data["rows"].items():
                results.append({"topic": topic, "tools": row["tools"], "conversation": row["conversation"]})


    with open(output_path, 'r') as f:
        json.dump(results, f, indent=2)


def run(openi_api_url, api_token, output_path, metadata_dir, xitxat_data):
    # Initialize PyTorch distributed
    dist.init_process_group(backend="gloo")  # Or "gloo" for CPU-only
    
    # Get rank and world size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("Output path:", output_path)
        print("Metadata dir:", metadata_dir)
        print("XitXat data:", xitxat_data)

    # Path to the dataset file
    client = OpenAI(
        base_url=openi_api_url, 
        api_key=api_token
    )

    progress_tensor = torch.zeros(world_size, dtype=torch.int32)
    stop_event = threading.Event()

    if rank == 0:
        progress_thread = threading.Thread(target=progress_reporter, args=(progress_tensor, xitxat_data, stop_event))
        progress_thread.start()

    # Call the process function for the current rank
    process_chunk(xitxat_data, client, metadata_dir, rank, world_size, progress_tensor)

    # Finalize distributed processing
    dist.barrier()
    if rank == 0:
        stop_event.set()
        progress_thread.join()
        parent_dir = os.path.dirname(output_path)
        if parent_dir:  # Avoid trying to create directories for paths without parent directories
            os.makedirs(parent_dir, exist_ok=True)
        metadata_to_dataset(metadata_dir, output_path)

    dist.destroy_process_group()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Distributed processing script.")
    parser.add_argument("--openi_api_url", type=str, default="http://localhost:8080/v1/", help="Openai api URL for openai client.")
    parser.add_argument("--output_path", type=str, default="./output.json", help="Path to save output files.")
    parser.add_argument("--metadata_dir", type=str, default="./metadata", help="Path to metadata files.")
    parser.add_argument("--xitxat_data", type=str, help="Path to the XitXat data file.")
    parser.add_argument("--hf_token", type=str, default="hf_xxxx", help="Hugging Face authentication token.")
    parser.add_argument("--api_token", type=str, default=None, help="OpenAI api token.")

    args = parser.parse_args()

    openi_api_url = args.openi_api_url
    output_path = args.output_path
    metadata_dir = args.metadata_dir
    xitxat_data = args.xitxat_data
    hf_token = args.hf_token
    api_token = args.api_token
    if not api_token:
        api_token = hf_token
    # Run the main function with parsed arguments
    
    run(openi_api_url, api_token, output_path, metadata_dir, xitxat_data)



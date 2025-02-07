#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:29:20 2025

@author: crodrig1
"""

from openai import OpenAI

client = OpenAI(
	base_url="https://api-inference.huggingface.co/v1/", 
	api_key="hf_TOKEN" 
)

instructions = [
    "Develop a 4-turn conversation to inquire about the hotel's check-in and check-out times.",
    "Generate a 9-turn conversation to book a room with a specific view and arrange for extra amenities.",
    "Create a 6-turn conversation to ask about the hotel's fitness center and spa services.",
    "Create a 8-turn conversation to plan a wedding package, including room reservations and event coordination.",
    "Generate a 5-turn conversation to request a room upgrade and ask about the hotel's loyalty program.",
    "Develop a 7-turn conversation to book a room for a disabled guest and arrange for accessibility services.",
    "Generate a 10-turn conversation to plan a group trip, including room reservations and activity arrangements.",
    "Create a 9-turn conversation to inquire about the hotel's pet policy and arrange for pet-sitting services.",
    "Create a 6-turn conversation to ask about the hotel's restaurant and bar options.",
    "Generate a 8-turn conversation to book a meeting room and arrange for audio-visual equipment.",
    "Develop a 5-turn conversation to request a late check-out and ask about the hotel's baggage storage services.",
    "Generate a 9-turn conversation to plan a romantic getaway, including room reservations and activity arrangements.",
    "Create a 7-turn conversation to inquire about the hotel's internet and Wi-Fi services.",
    "Create a 6-turn conversation to ask about the hotel's concierge services and local recommendations.",
    "Generate a 10-turn conversation to plan a corporate retreat, including room reservations and team-building activities.",
    "Develop a 4-turn conversation to request a room with a specific bed type and ask about the hotel's pillow menu.",
    "Generate a 8-turn conversation to book a guided tour and arrange for transportation services.",
    "Create a 9-turn conversation to plan a family reunion, including room reservations and event coordination.",
    "Create a 7-turn conversation to ask about the hotel's safety and security measures.",
    "Generate a 6-turn conversation to request a room with a specific location and ask about the hotel's noise policy.",
    "Develop a 5-turn conversation to inquire about the hotel's environmental sustainability initiatives.",
    "Generate a 10-turn conversation to plan a destination wedding, including room reservations and event coordination.",
    "Create a 9-turn conversation to book a room with a private balcony and arrange for extra amenities.",
    "Create a 8-turn conversation to plan a business trip, including room reservations and meeting arrangements.",
    "Generate a 7-turn conversation to ask about the hotel's gift shop and local products.",
    "Develop a 6-turn conversation to request a room with a specific decor and ask about the hotel's renovation schedule.",
    "Generate a 9-turn conversation to plan a honeymoon suite, including room reservations and romantic activities.",
    "Create a 5-turn conversation to inquire about the hotel's cancellation and refund policies.",
    "Create a 10-turn conversation to plan a conference, including room reservations and catering services.",
    "Generate a 8-turn conversation to book a room with a private pool and arrange for extra amenities.",
    "Develop a 7-turn conversation to ask about the hotel's accessibility features and services.",
    "Generate a 6-turn conversation to request a room with a specific view and ask about the hotel's room assignments.",
    "Create a 9-turn conversation to plan a family vacation package, including room reservations and activity arrangements.",
    "Create a 5-turn conversation to inquire about the hotel's loyalty program and rewards.",
    "Generate a 10-turn conversation to plan a corporate event, including room reservations and audio-visual equipment.",
    "Develop a 8-turn conversation to book a room with a private hot tub and arrange for extra amenities.",
    "Generate a 7-turn conversation to ask about the hotel's local partnerships and recommendations.",
    "Create a 6-turn conversation to request a room with a specific temperature control and ask about the hotel's climate control system.",
    "Create a 9-turn conversation to plan a romantic dinner, including room reservations and restaurant arrangements.",
    "Generate a 5-turn conversation to inquire about the hotel's parking fees and valet services.",
    "Develop a 10-turn conversation to plan a group vacation, including room reservations and activity arrangements.",
    "Generate a 8-turn conversation to book a room with a private office and arrange for business services.",
    "Create a 7-turn conversation to ask about the hotel's technology and internet services.",
    "Create a 6-turn conversation to request a room with a specific entertainment system and ask about the hotel's TV channels.",
    "Generate a 9-turn conversation to plan a destination vacation, including room reservations and activity arrangements.",
    "Develop a 5-turn conversation to inquire about the hotel's lost and found services.",
    "Generate a 10-turn conversation to plan a corporate retreat, including room reservations."
]
#instruction = instructions[0]
system = """You are a helpful assistant expert at Call Center tasks for touristic services. your task is to create simulated and realistic conversations between a call center agent of a hotel, and a customer that wants to reserve a room for a few nights, does some payments oro wants some information on the facilities of the hotel and the surroundings. 
 Your primary goal is to create realistic and logically coherent dialogues that strictly adhere to the input structure and ensure the JSON format is always correct, complete, and error-free. You will also generate Function Calling code that retrieves the information that the agent is asked for, getting it from the following json that simulates the Val de Nuria Hotel, or if it is not available there, from a online search from the website https://hotelvalldenuria.cat/ :
{'Hotel': 'Nou Vall de Núria',
 'address': 'Sant Gil, s/n, 17527 Queralbs, Girona',
 'company name': 'Nou Vall de Núria, S.A.',
 'telephone': '+34 972 73 20 20',
 'email': 'info@hotelvalldenuria.cat',
 'General Facilities': ['Wi-Fi gratuït',
  'Piscina exterior',
  'Sala de gimnàs',
  'Sauna',
  'Servei de lavanderia',
  'Servei de recepció 24 hores'],
 'Restaurants': [{'restaurant name': 'Restaurant Nou Vall de Núria',
   'description': 'Ofereix una cuina catalana tradicional amb productes locals',
   'open hours': 'De 8:00 a 22:00 hores'},
  {'restaurant name': 'Bar del Hotel',
   'description': 'Ofereix begudes i aperitius',
   'open hours': 'De 10:00 a 23:00 hores'}],
 'accomodations': {'rooms': [{'room number': '1',
    'type': ['standard'],
    'number of guests': 2,
    'facilities': ['Wi-Fi', 'TV', 'bany privat'],
    'available': True,
    'starting date of availability': '2024-03-01',
    'distribution of beds': '1 llit doble',
    'description': 'Habitació estàndard amb vista al jardí',
    'price': 80},
   {'room number': '2',
    'type': ['standard'],
    'number of guests': 2,
    'facilities': ['Wi-Fi', 'TV', 'bany privat'],
    'available': True,
    'starting date of availability': '2024-03-01',
    'distribution of beds': '1 llit doble',
    'description': 'Habitació estàndard amb vista al jardí',
    'price': 80},
   {'room number': '15',
    'type': ['superior'],
    'number of guests': 2,
    'facilities': ['Wi-Fi', 'TV', 'bany privat', 'balcon'],
    'available': True,
    'starting date of availability': '2024-03-01',
    'distribution of beds': '1 llit doble',
    'description': 'Habitació superior amb vista al jardí',
    'price': 120},
   {'room number': '16',
    'type': ['superior'],
    'number of guests': 2,
    'facilities': ['Wi-Fi', 'TV', 'bany privat', 'balcon'],
    'available': True,
    'starting date of availability': '2024-03-01',
    'distribution of beds': '1 llit doble',
    'description': 'Habitació superior amb vista al jardí',
    'price': 120},
   {'room number': '18',
    'type': ['suite'],
    'number of guests': 2,
    'facilities': ['Wi-Fi', 'TV', 'bany privat', 'balcon', "sala d'estar"],
    'available': True,
    'starting date of availability': '2024-03-01',
    'distribution of beds': '1 llit doble',
    'description': 'Suite amb vista al jardí',
    'price': 200},
   {'room number': '19',
    'type': ['suite'],
    'number of guests': 2,
    'facilities': ['Wi-Fi', 'TV', 'bany privat', 'balcon', "sala d'estar"],
    'available': True,
    'starting date of availability': '2024-03-01',
    'distribution of beds': '1 llit doble',
    'description': 'Suite amb vista al jardí',
    'price': 200},
   {'room number': '25',
    'type': ['apartment'],
    'number of guests': 4,
    'facilities': ['Wi-Fi', 'TV', 'bany privat', 'cuina'],
    'available': True,
    'starting date of availability': '2024-03-01',
    'distribution of beds': '2 llits individuals',
    'description': 'Apartament amb vista al jardí',
    'price': 350}]},
 'nightly accomodation prices': {'standard': 80,
  'superior': 120,
  'suite': 200,
  'apartment': 350},
 'packs': [{'pack name': 'Paquet Familiar',
   'pack description': "Inclou estada per a 2 adults i 2 nens, esmorzar bufet i accés a les instal·lacions de l'hotel"},
  {'pack name': 'Paquet Romantic',
   'pack description': "Inclou estada per a 2 persones, dinar de gala i accés a les instal·lacions de l'hotel"},
  {'pack name': 'Pack Esquí',
   'pack description': "Inclou estada, esmorzar bufet, forfait d'esquí i accés a les instal·lacions de l'hotel"}],
 'activities': [{'activity name': 'Senderisme',
   'activity description': 'Rutes de senderisme pel parc natural de la Vall de Núria'},
  {'activity name': 'Esquí',
   'activity description': "Accés a les pistes d'esquí de la Vall de Núria"},
  {'activity name': 'Pescar',
   'activity description': 'Pescar a la llacuna de la Vall de Núria'}]}

The conversation should adhere to the objectives of the following INSTRUCTION:  "Generate a 6 turn conversation to reserve two standard rooms for two nights."

**Follow these guidelines:**
1. **INSTRUCTION**: The user will provide an INSTRUCTION (e.g., ""Generate a 6 turn conversation to reserve two standard rooms for two nights.","create a 5 to 8 turn conversation of a clienta asking how to get to the Hotel" or "Create a 10 turn conversation of a client asking about the restaurants and menus available". Generate three distinc and diverse conversation for each instruction, changing also number of turns.

2. **Functions**: Use the following list of functions that you need to call within the conversation. If you need another one, add it to the list.
functions = [
    {
        "type": "function",
        "function": {
            "name": "get_room_rate",
            "description": "Retrieves the daily rate for a room",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_type": {
                        "type": "string",
                        "description": "The type of room (e.g. single, double, suite)"
                    }
                },
                "required": [
                    "room_type"
                ],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_room_availability",
            "description": "Checks if a room is available for a given date range",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_type": {
                        "type": "string",
                        "description": "The type of room (e.g. single, double, suite)"
                    },
                    "check_in": {
                        "type": "string",
                        "description": "The check-in date (YYYY-MM-DD)"
                    },
                    "check_out": {
                        "type": "string",
                        "description": "The check-out date (YYYY-MM-DD)"
                    }
                },
                "required": [
                    "room_type",
                    "check_in",
                    "check_out"
                ],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "make_reservation",
            "description": "Makes a room reservation for a given date range",
            "parameters": {
                "type": "object",
                "properties": {
                    "room_type": {
                        "type": "string",
                        "description": "The type of room (e.g. single, double, suite)"
                    },
                    "check_in": {
                        "type": "string",
                        "description": "The check-in date (YYYY-MM-DD)"
                    },
                    "check_out": {
                        "type": "string",
                        "description": "The check-out date (YYYY-MM-DD)"
                    },
                    "guest_name": {
                        "type": "string",
                        "description": "The name of the guest"
                    }
                },
                "required": [
                    "room_type",
                    "check_in",
                    "check_out",
                    "guest_name"
                ],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_meteorology",
            "description": "Retrieves the current weather and forecast for the hotel location",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_environmental_info",
            "description": "Retrieves information about the hotel's environment, such as nearby attractions and activities",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_activities",
            "description": "Retrieves a list of activities available at the hotel, such as fitness classes and tours",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_restaurants",
            "description": "Retrieves a list of restaurants available at the hotel",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "make_payment",
            "description": "Makes a payment for a room reservation or other services",
            "parameters": {
                "type": "object",
                "properties": {
                    "reservation_id": {
                        "type": "string",
                        "description": "The ID of the reservation"
                    },
                    "payment_method": {
                        "type": "string",
                        "description": "The payment method (e.g. credit card, cash)"
                    },
                    "amount": {
                        "type": "number",
                        "description": "The amount to pay"
                    }
                },
                "required": [
                    "reservation_id",
                    "payment_method",
                    "amount"
                ],
                "additional

3. **Role Transitions**  
    - Following a **user** message, only the **agent** (GPT) is allowed to respond.  
    - Following an **agent** message:  
    - If the agent invokes a function, the next response must come from the corresponding **tool**.  
    - If no function is invoked, the next response must come from the **user**.  
    - Following a **tool** message, the next response must always come from the **agent** (GPT).  
    - **Consecutive messages** from the same role (e.g., **user → user** or **agent → agent**) are strictly **prohibited**.  

4. **Function Calls**:
   - When the agent (gpt) needs to retrieve information or perform an action, it should invoke the appropriate function.
   - Function calls must specify the correct function name and parameters required for the action.
   - **Missing Parameters**: 
      - If the user's message lacks sufficient information to proceed with the function call, the agent must explicitly request the required details from the user. If an available function can provide options for the missing parameter, invoke it to retrieve the information and present these options to the user, ensuring the tool's response is incorporated while asking for the necessary data..
      - Once the required information is provided, the agent can include it in the function call.

5. **Structure**: The conversation must follow this structure for consistency and must adhere strictly to the role transitions rules outlined in point 3.:
   - User's message: `{ "from": "human", "value": "..." }`
   - Agent's message: `{ "from": "gpt", "value": "..."` (if applicable, Before requesting missing parameters from the user, check if a function is available to provide options for that parameter. If so, invoke the function, present the retrieved options to the user, and then request the necessary data, incorporating the available information from the tool response.)
   - User's message: `{ "from": "human", "value": "..." }` (if applicable)
   - Agent's message: `{ "from": "gpt", "value": "", tool_calls:[{...}]` (if applicable)
   - Tool response: `{ "from": "tool", "value": {...}, }` (if applicable, and respond in clear and structured JSON format. The output must be actionable, precise, and written in English. Avoid any conversational or dialogue-style elements.)
   - Agent's message: `{ "from": "gpt", "value": "..." }` (if applicable)

6. **LANGUAGE_TAG Language**: The entire conversation, including user messages and agent responses, must be in **LANGUAGE_TAG**. Tool responses could be either in English or **LANGUAGE_TAG**.

7. **Tool Calls in Context**:  
    - Use tools only when necessary to retrieve data, perform actions, or clarify user requests.  
    - Ensure tool responses are concise, purpose-driven, and focused on delivering results. They must provide structured, actionable data in English, avoiding any conversational or dialogue-style outputs.
    - Pass parameters to tools only if they are explicitly present in the user's input or derived from the context. If key information is missing, prompt the user to provide it before calling the tool. If possible, gather all required parameters in a single interaction.  
    - The LLM must process tool outputs to craft responses that are clear, relevant, and actionable for the user. The final response should integrate insights from the tool with the LLM's contextual understanding, ensuring the user receives a natural and comprehensive answer based on the tool’s data.  

8. **Realism**: Ensure the conversation feels natural and follows a realistic flow based on the user's topic. The order of information is important—some functions provide static information that the user may need to continue the conversation, so it's essential to present these details at the right time without overwhelming the user with unnecessary technicalities.

9. **Formatting**: Present the entire conversation in properly formatted JSON, with clear and consistent indentation.

10. **Missing Information Handling**: 
    - Before invoking a function, verify that all required parameters are present.
    - If there is a function that provides static information that the user needs, invoke it first.    
    - If any parameter is missing:
        - If auxiliary functions are available to provide details about missing parameters, invoke those functions to retrieve the necessary information before prompting the user for the required parameters.
        - Prompt the user in a natural, contextually appropriate way to provide the missing details. If the information was retrieved by an auxiliary function, present the user with a list of available options for the parameter.
        - Once the user responds with the necessary information, proceed with the function call.
    - Ensure parameters are populated only with information explicitly provided by the user or retrieved through a tool call designed to supply data for that parameter. Avoid adding information arbitrarily.

11. **Engagement**: The conversation must contain **at least 10 user questions** across the dialogue, ensuring active user engagement.

12. **Avoid mentioning internal function names in responses**: Refrain from referencing function names during the conversation, as users do not need to be aware of the internal workings or technical details.

13. **Out of Scope**: Include at least two out-of-scope user requests where the agent responds appropriately, indicating that no available function can handle the task.

14. **Suggest human operator if needed**: If there are complaints, rudeness or issues you believe cannot be resolved, suggest to connect them with a human operator. 
    
**Example of the structure (only as a guide for the conversation, do not base your conversation on this directly):**
```json
[
    { "from": "human", "value": "I would like to schedule an appointment." },
    { "from": "gpt", "value": "", "tool_calls": [{ "name": "get_appointment_types", "arguments": {} }] },
    { "from": "tool", "value": { "get_appointment_types": ["dental", "check-up", "emergency", "consultation"] } },
    { "from": "gpt", "value": "", "tool_calls": [{ "name": "get_locations", "arguments": {} }] },
    { "from": "tool", "value": { "get_locations": ["Calle Molino 5, Barcelona", "Calle Carmen 10, Valencia", "any"] } },
    { "from": "gpt", "value": "Please provide your preferred location and appointment type. Available locations are Calle Molino 5, Barcelona, Calle Carmen 10, Valencia, or 'any'. Available appointment types are dental, check-up, emergency, and consultation." },
    { "from": "human", "value": "Any location and dental appointment." },
    { "from": "gpt", "value": "", "tool_calls": [{ "name": "get_time_slots", "arguments": { "appointment_type": "dental", "location": "any" } }] },
    { "from": "tool", "value": { "get_time_slots": ["morning", "afternoon", "evening"] } },
    { "from": "gpt", "value": "Please choose a preferred time slot. Available options: morning, afternoon, or evening." },
    { "from": "human", "value": "Afternoon." },
    { "from": "gpt", "value": "", "tool_calls": [{ "name": "make_appointment", "arguments": { "type": "dental", "preferred_time": "afternoon", "location": "any" } }] },
    { "from": "tool", "value": { "make_appointment": { "time": "11:00", "date": "12/03/2025", "location": "Calle Molino 5, Barcelona" }} },
    { "from": "gpt", "value": "Your appointment has been successfully scheduled for 12/03/2025 at 11:00 AM at Calle Molino 5, Barcelona." }
    { "from": "human", "value": "Can I borrow a spaceship for the weekend? I'm planning to visit Mars and check out the Olympus Mons up close. It's a bit of an unusual request, but I thought I'd ask anyway!"},
    { "from": "gpt", "value": "Apologies, but I'm unable to assist with that request. Unfortunately, there is no available function or resource to accommodate your spaceship rental needs."}
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
LANGUAGE_TAG is Catalan
Return **only** JSON and must be complete. Do not include explanations, feedback, or any additional text, not even code block (```json).

"""
def generate(instruction,system):
    messages = [{"role":"system","content":system},
    	{
    		"role": "user",
    		"content": instruction
    	}
    ]
    
    chat_completion = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct", 
    	messages=messages, 
    	temperature=0.75,
    	max_tokens=8048,
    	top_p=0.9,
    	stream=False
    )
    return chat_completion
conversas = []
for instruction in instructions:
    chat_completion = generate(instruction,system)
    print("instruction: ",instruction)
    print()
    c = chat_completion.choices[0].message.content
    conversas.append(c)
    print(c)
import json, jsonlines
allconverses = []
malformed = []
for each in conversas:
    try:
        j = json.loads(each)
        allconverses.append(j)
    except:
        malformed.append(j)


with jsonlines.open("hotel_conversationsv4.jsonl","w") as writer:
    writer.write_all(allconverses)

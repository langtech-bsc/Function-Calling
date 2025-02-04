#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:55:20 2025

@author: crodrig1
"""
from openai import OpenAI

client = OpenAI(
	base_url="https://api-inference.huggingface.co/v1/",
	api_key=HF_TOKEN
)


json_prompt_creation = """
You are a helpful assistant that is an expert in finding information and storing it as a json file. Your task is to create a json with all the information in Catalan about a Hotel's services and facilities gathered from the website at the URL we weill provide.  
The json should contain a record for each of the rooms or apartments of the categories described, assuming a total of 25 accomodations, with the following distribution: 4 suites, 12 standard rooms, 5 double or Superior rooms, as well as 4 apartments. Each room record should contain a field with the list of facilities included in it, such as wifi, TV, etc., as well as how many guests it can accomodate.
Include information about how to arrive to the Hotel, and about the available Restaurants, Packs and organized activities. 
The required json schema is:
{"Hotel":"Nou Vall de Núria",
"address":string
"company name":string,
"telephone":string,
"email":string,
"General Facilities": list,
"Restaurants":
[
{"restaurant name":string,"description":string,"open hours":string}
]
"accomodations":
{"rooms":
[
{"room number": string,"type":["standard","superior","suite","apartment"],"number of guests":int,"facilities":list,"available":boolean,"starting date of availability":date,"distribution of beds":string,"description":string, "price":int}
]},
"nightly accomodation prices":{"standard":80,"superior":120,"suite":200,"apartment":350},
"packs": [
{"pack name"string, "pack description":string}
],
"activities":
[
{"activity name":string,"activity description"}
]
}
 The URL:
 """
Hotel="https://hotelvalldenuria.cat"
modelo = "meta-llama/Llama-3.3-70B-Instruct"

messages = [
 {"role": "system", "content": json_prompt_creation},
 	{ "role": "user", "content": Hotel }
 ]
response = client.chat.completions.create(
    model = modelo,#.split("/")[-1]#"meta-llama/Llama-3.3-70B-Instruct", 
 	messages=messages, 
 	temperature=0.1,
 	max_tokens=10048,
 	top_p=0.8,
 	stream=False
 )
 
 
eljson = response.choices[0].message.conten

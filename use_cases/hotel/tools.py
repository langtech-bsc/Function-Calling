import random
from abc import ABC, abstractmethod
import os
# from langchain.tools import tool
import json
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Dict, Optional, Union
import random
import copy



def read_json(data_path: str) -> tuple[list, dict]:
    try:
        with open(data_path, 'r', encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]
    except:
        with open(data_path, 'r', encoding="utf-8") as f:
            data = json.loads(f.read())
    return data

json_data = read_json("data/val_de_nuria.json")
reservations = {}
class ToolBase(BaseModel, ABC):
    @abstractmethod
    def invoke(cls, input: Dict):
        pass

    @classmethod
    def to_openai_tool(cls):
        """
        Extracts function metadata from a Pydantic class, including function name, parameters, and descriptions.
        Formats it into a structure similar to OpenAI's function metadata.
        """
        function_metadata = {
            "type": "function",
            "function": {
                "name": cls.__name__,  # Function name is same as the class name, in lowercase
                "description": cls.__doc__.strip(),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

        # Iterate over the fields to add them to the parameters
        for field_name, field_info in cls.model_fields.items():
            # Field properties
            field_type = "string"  # Default to string, will adjust if it's a different type
            annotation = field_info.annotation.__args__[0] if getattr(field_info.annotation, "__origin__", None) is Union else field_info.annotation 

            if annotation == int:
                field_type = "integer"
            elif annotation == bool:
                field_type = "boolean"
            
            # Add the field's description and type to the properties
            function_metadata["function"]["parameters"]["properties"][field_name] = {
                "type": field_type,
                "description": field_info.description,
            }

            # Determine if the field is required (not Optional or None)
            if field_info.is_required():
                function_metadata["function"]["parameters"]["required"].append(field_name)

            # If there's an enum (like for `unit`), add it to the properties
            if hasattr(field_info, 'default') and field_info.default is not None and isinstance(field_info.default, list):
                function_metadata["function"]["parameters"]["properties"][field_name]["enum"] = field_info.default

        return function_metadata

tools: Dict[str, ToolBase] = {}
oitools = []

def tool_register(cls: BaseModel):
    oaitool = cls.to_openai_tool()
    
    oitools.append(oaitool)
    tools[oaitool["function"]["name"]] = cls

@tool_register
class hotel_information(ToolBase):
    """Retrieves basic information about the hotel, such as its name, address, contact details, and overall description."""

    @classmethod
    def invoke(cls, input: Dict) -> str: 
        return """### **Nou Vall de Núria – Brief Description**  
Nestled in the stunning **Vall de Núria** in the Pyrenees, **Nou Vall de Núria** offers a perfect blend of comfort and adventure. Guests can enjoy breathtaking mountain views, premium accommodations, and excellent facilities, including an outdoor pool, gym, and sauna.  
The hotel features **two dining options**, serving traditional Catalan cuisine and refreshing drinks. Accommodations range from **cozy standard rooms to luxurious suites and fully equipped apartments**, catering to couples, families, and groups.  
For an unforgettable stay, guests can choose from **special packages**, including family-friendly stays, romantic getaways, ski adventures, and relaxation retreats. Outdoor enthusiasts can explore **hiking trails, ski slopes, and fishing spots** in the surrounding natural beauty.  
Whether for relaxation or adventure, **Nou Vall de Núria** promises a unique and memorable experience.""" 

@tool_register
class hotel_facilities(ToolBase):
    """Provides a list of available general facilities at the hotel, which could include amenities like a spa, pool, gym, conference rooms, etc."""

    @classmethod
    def invoke(cls, input: Dict) -> str: 
        return json_data["general_facilities"]

@tool_register
class restaurants_info(ToolBase):
    """Provides a list of available restaurants with their information."""

    @classmethod
    def invoke(cls, input: Dict) -> str: 
        """
        Play a playlist by its name, starting with the first or a random song.
        """

        return json_data["restaurants"]
    

# @tool_register
# class restaurant_details(ToolBase):
#     """Retrieves detailed information about a specific restaurant in the hotel, including its menu, ambiance, operating hours, and special features."""

#     name: str = Field(default=[res["name"] for res in json_data["restaurants"]], description="Name of the resaturant")
    
#     @classmethod
#     def invoke(cls, input: Dict) -> str: 
#         """
#         Play a playlist by its name, starting with the first or a random song.
#         """

#         instance = cls(**input)
#         name = instance.name

#         restaurante = [res for res in json_data["restaurants"] if res["name"] == name]
#         if restaurante:
#             return restaurante
#         else:
#             return f"We don't have any restaurante with the name: {name}"


@tool_register
class room_types(ToolBase):
    """
    Returns a list of room types available at the hotel (e.g., single, double, suite, deluxe) along with brief descriptions of each type.
    """
    @classmethod
    def invoke(cls, input: Dict) -> str: 
        return json_data["room_types"]


@tool_register
class check_room_availability(ToolBase):
    """
    Checks if a specified room type is available between the provided check-in and check-out dates for a given number of guests.
    """
    room_type: str = Field(default=list(json_data["room_types"].keys()), description="The type of room the user is interested in")
    check_in_date: str = Field(description="The starting date of the reservation (e.g., \"2025-04-01\")")
    check_out_date: str = Field(description="The ending date of the reservation (e.g., \"2025-04-05\").")
    guests: int = Field(description="The number of guests that will occupy the room.")
    
    @classmethod
    def invoke(cls, input: Dict) -> str: 
        
        room_type = input.get("room_type", None)
        check_in_date = input.get("check_in_date", None)
        check_out_date = input.get("check_out_date", None)
        guests = input.get("guests", None)

        missing = []
        if not room_type:
            missing.append("room_type")
        if not check_in_date:
            missing.append("check_in_date")
        if not check_out_date:
            missing.append("check_out_date")
        if not guests:
            missing.append("guests")

        if len(missing):
            value = ", ".join(missing)
            return f"Unable to check the room availability. The following required arguments are missing:{value}." 

        instance = cls(**input)
        room_type = instance.room_type
        check_in_date = instance.check_in_date
        check_out_date = instance.check_out_date
        guests = instance.guests
        
        rooms = [room for room in json_data["accomodations"]["rooms"] if room_type in room["type"]]
        if len(rooms) == 0:
            return f"There is no room exists with room type {room_type}"
        
        rooms2 = [room for room in rooms if guests >= room["number_of_guests"]]
        if len(rooms2) == 0:
            max_guests = json_data["room_types"][room_type]["number_of_guests"]
            return f"The number of guest is superior then the availibilty, maximum is {max_guests}"


@tool_register
class make_reservation(ToolBase):
    """
    Creates a new reservation for the hotel by booking a room of the specified type for the desired dates, and associating the booking with a user.
    """

    room_type: str = Field(default=list(json_data["room_types"].keys()), description="The type of room being reserved.")
    check_in_date: str = Field(description="The starting date of the reservation (e.g., \"2025-04-01\")")
    check_out_date: str = Field(description="The ending date of the reservation (e.g., \"2025-04-05\").")
    guests: int = Field(description="The number of guests for the reservation.")
    user_id: int = Field(description="The identifier for the user making the reservation.")
    
    @classmethod
    def invoke(cls, input: Dict) -> str:
        
        room_type = input.get("room_type", None)
        check_in_date = input.get("check_in_date", None)
        check_out_date = input.get("check_out_date", None)
        guests = input.get("guests", None)
        user_id = input.get("user_id", None)
        
        missing = []
        if not room_type:
            missing.append("room_type")
        if not check_in_date:
            missing.append("check_in_date")
        if not check_out_date:
            missing.append("check_out_date")
        if not guests:
            missing.append("guests")
        if not user_id:
            missing.append("user_id")

        if len(missing):
            value = ", ".join(missing)
            return f"Unable to complete the reservation. The following required arguments are missing:{value}."   
    

        instance = cls(**input)
        room_type = instance.room_type
        check_in_date = instance.check_in_date
        check_out_date = instance.check_out_date
        guests = instance.guests
        user_id = instance.user_id

        rooms = [room for room in json_data["accomodations"]["rooms"] if room_type in room["type"]]
        if len(rooms) == 0:
            return f"There is no room exists with room type {room_type}"
        
        rooms2 = [room for room in rooms if guests >= room["number_of_guests"]]
        if len(rooms2) == 0:
            max_guests = json_data["room_types"][room_type]["number_of_guests"]
            return f"The number of guest is superior then the availibilty, maximum is {max_guests}"

        room = rooms2[random.randint(0, len(rooms2))]

        rand = int(random.randint(0,10000000))
        while rand in reservations:
            rand = int(random.randint(0,10000000))
        
        tmp_data = {
            "status": "Reserved",
            "room_number": room["room_number"],
            "room_type": room_type,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "guests": guests,
            "reservation_id": rand,
            "user_id": user_id,
        }

        reservations[rand] = tmp_data

        return json.dumps(tmp_data)

@tool_register
class cancel_reservation(ToolBase):
    """Playing a specific playlist by its name."""

    user_id: int = Field(description="The identifier for the user requesting the cancellation.")
    reservation_id: int = Field(description="The unique identifier of the reservation to be canceled.")
    
    @classmethod
    def invoke(cls, input: Dict) -> str: 

        reservation_id = input.get("reservation_id", None)
        user_id = input.get("user_id", None)

        missing = []
        if not reservation_id:
            missing.append("reservation_id")
        if not user_id:
            missing.append("user_id")

        if len(missing):
            value = ", ".join(missing)
            return f"Unable to cancel the reservation. The following required arguments are missing:{value}."     
        
        instance = cls(**input)
        reservation_id = instance.reservation_id
        user_id = instance.user_id



        if reservation_id not in reservations:
            return f"There is no reservations with the id: {reservation_id}"
        
        if reservations["reservation_id"]["user_id"] != user_id:
            return "The user id is wrong, please provide same user id that was used make to reservation."
        
        reservations.pop(reservation_id)
        return f"The reservation {reservation_id} is cancled correctly"

@tool_register
class modify_reservation(ToolBase):
    """
    Allows a user to modify an existing reservation by updating the check-in/check-out dates or changing the room type, subject to availability.
    """


    new_room_type: str = Field(default=list(json_data["room_types"].keys()) + [None], description=f"The type of new room to be modified, if {None} same room will be modified.")
    new_check_in_date: str = Field(default=None, description="New check out date in format DD/MM/YYYY")
    new_check_out_date: str = Field(default=None, description="New check out date in format DD/MM/YYYY")
    guests: int = Field(default=None, description="New number of guests for the reservation.")
    user_id: int = Field(description="The identifier for the user requesting the modification.")
    reservation_id: int = Field(description="The unique identifier of the reservation to be modified.")

    @classmethod
    def invoke(cls, input: Dict) -> str: 
        
        user_id = input.get("user_id", None)
        reservation_id = input.get("reservation_id", None)

        missing = []
        if not reservation_id:
            missing.append("reservation_id")
        if not user_id:
            missing.append("user_id")

        instance = cls(**input)
        new_room_type = instance.new_room_type
        new_check_in_date = instance.new_check_in_date
        new_check_out_date = instance.new_check_out_date
        guests = instance.guests
        user_id = instance.user_id
        reservation_id = instance.reservation_id

        if len(missing):
            value = ", ".join(missing)
            return f"Unable to modify the reservation. The following required arguments are missing:{value}."     

        if not (new_room_type or new_check_in_date or new_check_out_date or guests):
            return "Unable to modify the reservation. One of the following arguments must be passed: new_room_type, new_check_in_date, new_check_out_date, guests."     


        if reservation_id not in reservations:
            return f"There is no reservations with the id: {reservation_id}"
        
        if reservations["reservation_id"]["user_id"] != user_id:
            return "The user id is wrong, please provide same user id that was used make to reservation."
        
        if new_room_type or guests:
            rooms = [room for room in json_data["restaurants"] if new_room_type in room["type"]]
            if len(rooms) == 0:
                return f"There is no room exists with room type {new_room_type}"
            
            rooms = [room for room in rooms if guests >= room["number_of_guests"]]
            if len(rooms) == 0:
                max_guests = json_data["room_types"][new_room_type]["number_of_guests"]
                return f"The number of guest is superior then the availibilty, maximum is {max_guests}"
            
            room = rooms[random.randint(0, len(rooms))]
            room_number = room["room_number"]
        else:
            room_number = reservations["reservation_id"]["room_number"]

        
        reservations["reservation_id"]["guests"] = guests if guests else reservations["reservation_id"]["guests"]
        reservations["reservation_id"]["check_in_date"] = new_check_in_date if new_check_in_date else reservations["reservation_id"]["check_in_date"]
        reservations["reservation_id"]["check_out_date"] = new_check_out_date if new_check_out_date else reservations["reservation_id"]["check_out_date"]
        reservations["reservation_id"]["room_type"] = new_room_type if new_room_type else reservations["reservation_id"]["room_type"]
        reservations["reservation_id"]["room_number"] = room_number

        tmp_data = copy.deepcopy(reservations["reservation_id"])
        tmp_data.pop("user_id")

        return f"The reservation {reservation_id} is modified correctly: {json.dumps(tmp_data)}"

@tool_register
class reservation_details(ToolBase):
    """Playing a specific playlist by its name."""

    user_id: int = Field(description="Id of user, could be passport or national Identity number")
    reservation_id: int = Field(description="Id of the reservation")

    @classmethod
    def invoke(cls, input: Dict) -> str: 
        user_id = input.get("user_id", None)
        reservation_id = input.get("reservation_id", None)
    
        missing = []
        if not reservation_id:
            missing.append("reservation_id")
        if not user_id:
            missing.append("user_id")

        if len(missing):
            value = ", ".join(missing)
            return f"Unable to get the details. The following required arguments are missing:{value}."     

        instance = cls(**input)
        user_id = instance.user_id
        reservation_id = instance.reservation_id

        if reservation_id not in reservations:
            return f"There is no reservations with the id: {reservation_id}"
        
        if reservations["reservation_id"]["user_id"] != user_id:
            return "The user id is wrong, please provide same user id that was used make to reservation."
        
        tmp_data = copy.deepcopy(reservations["reservation_id"])
        tmp_data.pop("user_id")
        return json.dumps(tmp_data)

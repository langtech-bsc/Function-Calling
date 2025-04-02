from generate_dataset.methods.method_manager import MethodManager, BaseMethod, GetLLMResponseType, MessagesType
from typing import Dict, Any, List
import json

@MethodManager.register("default")
class Default(BaseMethod):
    def __init__(self, input: str, output: str, global_rank:int, messages_list: List[MessagesType], unique_key, output_keys, output_types):
        super().__init__(input, output, global_rank, messages_list, unique_key, output_keys, output_types)

    def generate_data(self, index, get_llm_response: GetLLMResponseType) -> Dict[str, Any]:
        json_data = {}
        new_data = {}
        new_data.update(json_data)
        
        for i in range(self.messages_list):
            messages = self.generate_messages(new_data, i)
            response = get_llm_response(messages=messages, wait_for_connection=True)
            if self.output_type[i] == "json":
                response = json.dumps(response)
            new_data.update({self.output_keys[i]: response})
        return {key: new_data[key] for key in new_data if key in self.output_keys}
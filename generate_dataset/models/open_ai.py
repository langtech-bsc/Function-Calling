from generate_dataset.models.model_manager import ModelManager, BaseModel
from openai import OpenAI
import time


@ModelManager.register("openai")
class OpenAIChat(BaseModel):
    def __init__(self, api_url="http://localhost:8080/v1/", api_key="xyz", model="tgi", model_params={}):
        self.api_url = api_url
        self.model = model
        self._api_key = api_key
        self._client = OpenAI(
            base_url=self.api_url,  
            api_key=api_key
        )
        self.model_params = self._get_params(model_params, {"model", "messages", "stream"})


    def get_response(self, messages, wait_for_connection=False):
        params = {"model": self.model, "messages": messages, "stream": False}
        params.update(self.model_params)
        first = True
        
        while wait_for_connection or first:
            try:
                first = False
                response = self._client.chat.completions.create(**params)
                print(response.choices[0].message.content)
                return "TODO"
            except Exception as err:
                print("Error: openai api connection error waiting...")
                time.sleep(60)


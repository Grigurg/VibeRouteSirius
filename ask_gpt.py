# import openai
from openai import OpenAI
from loguru import logger
import json

with open("api_key", 'r') as file:
    lines_list = file.readlines()

# YANDEX_CLOUD_FOLDER = lines_list[0].removesuffix('\n')
# YANDEX_CLOUD_API_KEY = lines_list[1].removesuffix('\n')
OPENROUTER_API_KEY = "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"

class QwenChat:
    def __init__(self):
        self.start_message = "Ты -- нейросеть, которая умеет строить оптимальные маршруты с учетом пожеланий пользователя."
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

    def ask(self, messages, query, tools, model_name="qwen_235b"):
        if len(query) >= 1:
            messages.append({"role": "user", "content": query})

        # Map form model names to actual API model names
        # model_mapping = {
        #     "qwen_235b": f"gpt://{YANDEX_CLOUD_FOLDER}/qwen3-235b-a22b-fp8/latest",
        #     "gemma-3-27b-it": f"gpt://{YANDEX_CLOUD_FOLDER}/gemma-3-27b-it/latest"
        # }
        #
        # model = model_mapping.get(model_name, f"gpt://{YANDEX_CLOUD_FOLDER}/qwen3-235b-a22b-fp8/latest")

        response=self.client.chat.completions.create(
            model="qwen/qwen-2.5-72b-instruct",
            messages=messages,
            temperature=0.3,
            tools=tools,
            tool_choice="required",
            max_tokens=2000
        )

        tool_call = response.choices[0].message.tool_calls[0]
        name = tool_call.function.name
        id = tool_call.id
        args = json.loads(tool_call.function.arguments)

        answer = response.choices[0].message
        messages.append(answer)
        # self.messages.append({"role": "assistant", "content": answer})
        return name, args, id


        # return answer
    
    def clear_history(self, messages):
        messages = [
            {"role": "system", "content": self.start_message}
        ]

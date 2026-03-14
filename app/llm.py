import json
from pathlib import Path

from loguru import logger
from openai import OpenAI
from openai import AuthenticationError

from app.data_access import EmbSearch, get_points, load_database

ROOT_DIR = Path(__file__).resolve().parent.parent
API_KEY_FILE = ROOT_DIR / "api_key"

OPENROUTER_MODELS = {
    "gpt-4o-mini": {
        "label": "OpenAI GPT-4o Mini",
        "openrouter_id": "openai/gpt-4o-mini",
    },
    "claude-3.5-haiku": {
        "label": "Anthropic Claude 3.5 Haiku",
        "openrouter_id": "anthropic/claude-3.5-haiku",
    },
    "llama-3.3-70b": {
        "label": "Meta Llama 3.3 70B",
        "openrouter_id": "meta-llama/llama-3.3-70b-instruct",
    },
    "gemini-2.0-flash": {
        "label": "Google Gemini 2.0 Flash",
        "openrouter_id": "google/gemini-2.0-flash-001",
    },
}

DEFAULT_MODEL = "gpt-4o-mini"


def load_openrouter_api_key():
    if not API_KEY_FILE.exists():
        return None

    value = API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not value:
        return None

    return value


class OpenRouterChat:
    def __init__(self):
        self.start_message = "Ты -- нейросеть, которая умеет строить оптимальные маршруты с учетом пожеланий пользователя."
        api_key = load_openrouter_api_key()
        if not api_key:
            raise RuntimeError(
                "OpenRouter API key is not configured in the api_key file."
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def ask(self, messages, query, tools, model_name=DEFAULT_MODEL):
        if len(query) >= 1:
            messages.append({"role": "user", "content": query})

        selected_model = OPENROUTER_MODELS.get(model_name, OPENROUTER_MODELS[DEFAULT_MODEL])

        try:
            response = self.client.chat.completions.create(
                model=selected_model["openrouter_id"],
                messages=messages,
                temperature=0.3,
                tools=tools,
                tool_choice="required",
                max_tokens=2000,
            )
        except AuthenticationError as exc:
            raise RuntimeError(
                "OpenRouter authentication failed. Check OPENROUTER_API_KEY or the api_key file."
            ) from exc

        answer = response.choices[0].message
        messages.append(answer)
        return answer.tool_calls or []

    def clear_history(self, messages):
        messages = [{"role": "system", "content": self.start_message}]


class LLMAgent:
    def __init__(self):
        self.model = OpenRouterChat()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_places",
                    "parameters": {
                        "type": "object",
                        "description": "Функция получает строку запроса и возвращает до 9 самых подходящих точек под запрос",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "message",
                    "parameters": {
                        "type": "object",
                        "description": "В эту функцию нужно передать точки для маршрута в формате: адрес, id точки. Передавай точки в порядке убывания расстояния до финальной точки!!!",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Описание маршрута",
                            },
                            "points": {
                                "type": "array",
                                "description": "список id точек маршрута",
                                "items": {"type": "integer"},
                            },
                        },
                        "required": ["text", "points"],
                    },
                },
            },
        ]
        self.db = load_database()
        self.embs = EmbSearch(self.db, 3)

    def get_places(self, query, a, b):
        res_points, dists_to_a, dists_to_b = get_points(self.db, self.embs, a, b, query)
        ans = ""
        for index, point in enumerate(res_points):
            ans += (
                "id: "
                + str(point.id)
                + ", адрес: "
                + point.street
                + ", расстояние до начальной точки в метрах: "
                + str(dists_to_a[index])
                + ", расстояние до конечной точки в метрах: "
                + str(dists_to_b[index])
                + ", описание: "
                + point.desc
                + "\n"
            )
        return ans

    def message(self, text, points):
        return text, points

    def answer_model(self, messages, desc_ans, ans_id, a, b, prompt, model_name=DEFAULT_MODEL):
        tool_calls = self.model.ask(messages, prompt, self.tools, model_name)
        tool_results = []

        for tool_call in tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if name == "get_places":
                result = self.get_places(args["query"], a, b)
                logger.info(result)
            elif name == "message":
                desc = args["text"]
                points = args["points"]
                result = self.message(desc, points)
                desc_ans, ans_id = result
            else:
                result = None

            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        return tool_results, desc_ans, ans_id

    def get_answer(self, a, b, w_type, time, budget, extra_notes, model_name=DEFAULT_MODEL):
        dist = a.dist_between_points(b)
        type_route = ""
        if w_type == "friendly":
            type_route = "Дружеская прогулка"
        elif w_type == "romantic":
            type_route = "Романтическая прогулка"
        elif w_type == "family":
            type_route = "Прогулка с семьёй"
        elif w_type == "active":
            type_route = "Активная прогулка"
        elif w_type == "cozy":
            type_route = "Спокойная и уютная прогулка"
        elif w_type == "cultural":
            type_route = "Культурная прогулка"
        extra_notes_text = f"Дополнительные пожелания пользователя: {extra_notes}." if extra_notes.strip() else ""

        system_prompt = f"Ты находишься в России, на Федеральной территории Сириус. \
Требуется построить пешеходный маршрут для цели: \"{type_route}\", который начинается в адресе \
{a.street} и заканчивается в адресе {b.street}. Длительность маршрута должна \
быть равна примерно {time} минут. Расстояние от точки {a.street} до точки {b.street} -- {dist} метров. \
Бюджет на маршрут: {budget} рублей на человека. {extra_notes_text} Маршрут должен содержать не более 5 промежуточных \
точек и быть интересным и наиболее подходящим для данного случая. В маршруте должен быть \
либо одно кафе или ресторан, либо вообще без кафе и ресторанов, только если пользователь не попросил больше! \
Все точки должны быть уникальными (в частности, ни одна промежуточная точка не должна совпадать \
с точками {a.street} и {b.street})! Ты можешь спрашивать какие есть подходящие точки, \
сделав запрос get_places. В качестве аргумента передай, какого типа точки тебе нужны. \
Функция вернет тебе адреса, краткие описания и id всех подходящих заведений, а также их расстояния \
до начальной и конечной точек маршрута в метрах. Обязательно учитывай эти расстояния для определения порядка точек в маршруте! Не бери в маршрут \
точки, которые находятся слишком далеко, если поблизости есть аналоги (пусть и похуже)! НЕЛЬЗЯ, \
ЧТОБЫ ПРОГУЛКА ПОЛУЧИЛАСЬ ДОЛЬШЕ, ЧЕМ ХОЧЕТ ПОЛЬЗОВАТЕЛЬ! Также постарайся сделать её не сильно короче! \
Когда будешь готов добавить точку в маршрут, вызови функцию message и передай ей описание маршрута с адресами и массив id \
точек. За раз можно сделать только один запрос к функциям. Всего, прежде чем выдать ответ, сделай не \
более 5 запросов. После 5 запросов обязательно выведи ответ, если не сделал этого раньше!!! Начальную и \
конечную точку не нужно добавлять в маршрут! Также сделай чтобы выбранные тобой точки обходились в \
оптимальном порядке (ЭТО ОЧЕНЬ ВАЖНО)!!!! Сделай, чтобы суммарное расстояние в маршруте было как можно меньше, \
при этом необязательно минимизировать количество точек!! Если в маршруте есть кафе, то добавь его \
название в описание маршрута в message"

        desc_ans = "no-description"
        ans_id = []
        prompt = system_prompt
        messages = [{"role": "system", "content": self.model.start_message}]
        cnt = 0

        while cnt < 10 and len(ans_id) == 0:
            cnt += 1
            tool_results, desc_ans, ans_id = self.answer_model(
                messages, desc_ans, ans_id, a, b, prompt, model_name
            )
            messages.extend(tool_results)
            prompt = ""

        self.model.clear_history(messages)
        return desc_ans, list(ans_id)

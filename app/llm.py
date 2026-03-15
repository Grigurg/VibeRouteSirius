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
MAX_PLACE_CANDIDATES = 8
MAX_PLACE_DESCRIPTION_LENGTH = 220
PACE_PROFILES = {
    "relaxed": {
        "label": "relaxed",
        "speed_m_per_min": 40,
        "stop_share": 0.30,
    },
    "normal": {
        "label": "normal",
        "speed_m_per_min": 48,
        "stop_share": 0.22,
    },
    "active": {
        "label": "active",
        "speed_m_per_min": 58,
        "stop_share": 0.15,
    },
}


def load_openrouter_api_key():
    if not API_KEY_FILE.exists():
        return None

    value = API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not value:
        return None

    return value


class OpenRouterChat:
    def __init__(self):
        self.start_message = "You are an AI assistant that can build optimal routes based on the user's preferences."
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
                        "description": "This function takes a query string and returns up to 9 locations that best match the request.",
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
                        "description": "Pass route points to this function in the format: address, point id. Pass the points in descending order of distance to the final point.",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Route description",
                            },
                            "points": {
                                "type": "array",
                                "description": "List of route point ids",
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

    def shorten_text(self, text, limit):
        if text is None:
            return ""
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def wants_cafe_stop(self, extra_notes):
        normalized = (extra_notes or "").lower()
        keywords = ("coffee", "cafe", "cafes", "espresso", "latte", "кофе", "кафе", "кофейня")
        return any(keyword in normalized for keyword in keywords)

    def get_pace_profile(self, pace):
        return PACE_PROFILES.get(pace, PACE_PROFILES["relaxed"])

    def build_distance_budget(self, start_point, end_point, total_minutes, pace):
        profile = self.get_pace_profile(pace)
        walking_minutes = max(total_minutes * (1 - profile["stop_share"]), 30)
        computed_budget = int(walking_minutes * profile["speed_m_per_min"])
        direct_distance = start_point.dist_between_points(end_point)
        min_required_budget = int(direct_distance * 1.12)
        return {
            "pace_label": profile["label"],
            "speed_m_per_min": profile["speed_m_per_min"],
            "walking_minutes": int(round(walking_minutes)),
            "distance_budget_m": max(computed_budget, min_required_budget),
            "direct_distance_m": direct_distance,
        }

    def calculate_route_distance(self, start_point, point_ids, end_point):
        route_points = [start_point]
        for point_id in point_ids:
            if not isinstance(point_id, int) or not (0 <= point_id < len(self.db)):
                return None
            route_points.append(self.db[point_id])
        route_points.append(end_point)

        total_distance = 0
        for index in range(len(route_points) - 1):
            total_distance += route_points[index].dist_between_points(route_points[index + 1])
        return total_distance

    def normalize_point_ids(self, point_ids, end_point):
        normalized_ids = []
        seen_ids = set()

        for point_id in point_ids:
            if not isinstance(point_id, int) or not (0 <= point_id < len(self.db)):
                continue
            if point_id in seen_ids:
                continue
            normalized_ids.append(point_id)
            seen_ids.add(point_id)

        normalized_ids.sort(
            key=lambda point_id: self.db[point_id].dist_between_points(end_point),
            reverse=True,
        )
        return normalized_ids

    def build_fallback_query(self, walk_type, extra_notes):
        route_queries = {
            "friendly": "promenade, viewpoint, park, cafe, public space",
            "romantic": "seafront, scenic viewpoint, quiet park, cozy cafe",
            "family": "family-friendly park, playground, promenade, cafe",
            "active": "promenade, sports area, park, viewpoint",
            "cozy": "quiet promenade, cafe, park, scenic place",
            "cultural": "museum, art object, landmark, promenade",
        }
        base_query = route_queries.get(walk_type, route_queries["friendly"])
        if extra_notes.strip():
            return f"{base_query}. User preferences: {extra_notes.strip()}"
        return base_query

    def build_fallback_route(self, start_point, end_point, walk_type, extra_notes, constraints, lang):
        query = self.build_fallback_query(walk_type, extra_notes)
        candidates, _, _ = get_points(self.db, self.embs, start_point, end_point, query)
        direct_distance = constraints["direct_distance_m"]
        distance_budget = constraints["distance_budget_m"]
        selected_points = []
        selected_ids = []
        used_ids = set()
        require_cafe = self.wants_cafe_stop(extra_notes)

        if require_cafe:
            cafe_candidates, _, _ = get_points(
                self.db,
                self.embs,
                start_point,
                end_point,
                "cafe, coffee shop, cafe with takeaway coffee",
            )
            for candidate in cafe_candidates:
                if candidate.id in used_ids:
                    continue
                tentative_ids = [candidate.id]
                total_distance = self.calculate_route_distance(start_point, tentative_ids, end_point)
                if total_distance is None or total_distance > distance_budget:
                    continue
                selected_points.append(candidate)
                selected_ids.append(candidate.id)
                used_ids.add(candidate.id)
                break

        for candidate in candidates:
            if candidate.id in used_ids:
                continue
            detour_distance = (
                start_point.dist_between_points(candidate)
                + candidate.dist_between_points(end_point)
                - direct_distance
            )
            if detour_distance > max(distance_budget - direct_distance, 0):
                continue

            tentative_ids = selected_ids + [candidate.id]
            total_distance = self.calculate_route_distance(start_point, tentative_ids, end_point)
            if total_distance is None or total_distance > distance_budget:
                continue

            selected_points.append(candidate)
            selected_ids.append(candidate.id)
            used_ids.add(candidate.id)
            if len(selected_ids) >= 3:
                break

        selected_ids = self.normalize_point_ids(selected_ids, end_point)
        ordered_points = [self.db[point_id] for point_id in selected_ids]
        description = self.build_fallback_description(
            start_point,
            end_point,
            ordered_points,
            constraints,
            lang,
        )
        return description, selected_ids

    def build_fallback_description(self, start_point, end_point, points, constraints, lang):
        if lang == "ru":
            if points:
                point_names = ", ".join(point.name or point.street for point in points)
                return (
                    f"Маршрут начинается в {start_point.street} и ведёт к {end_point.street} "
                    f"через близкие и удобные точки: {point_names}. "
                    f"Маршрут рассчитан на спокойный темп и общую пешую дистанцию до {constraints['distance_budget_m']} м."
                )
            return (
                f"Маршрут проходит от {start_point.street} до {end_point.street} без промежуточных остановок, "
                f"чтобы сохранить спокойный темп и уложиться в пешую дистанцию до {constraints['distance_budget_m']} м."
            )

        if points:
            point_names = ", ".join(point.name or point.street for point in points)
            return (
                f"The route starts at {start_point.street} and finishes at {end_point.street}, "
                f"passing through nearby convenient stops: {point_names}. "
                f"It is tuned for a relaxed walking pace and a total walking distance of up to {constraints['distance_budget_m']} meters."
            )
        return (
            f"The route goes from {start_point.street} to {end_point.street} without intermediate stops "
            f"to keep a relaxed pace and stay within a walking distance of up to {constraints['distance_budget_m']} meters."
        )

    def get_places(self, query, a, b):
        res_points, dists_to_a, dists_to_b = get_points(self.db, self.embs, a, b, query)
        res_points = res_points[:MAX_PLACE_CANDIDATES]
        dists_to_a = dists_to_a[:MAX_PLACE_CANDIDATES]
        dists_to_b = dists_to_b[:MAX_PLACE_CANDIDATES]
        ans = ""
        for index, point in enumerate(res_points):
            short_desc = self.shorten_text(point.desc, MAX_PLACE_DESCRIPTION_LENGTH)
            ans += (
                "id: "
                + str(point.id)
                + ", address: "
                + point.street
                + ", distance to the start point in meters: "
                + str(dists_to_a[index])
                + ", distance to the end point in meters: "
                + str(dists_to_b[index])
                + ", description: "
                + short_desc
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
                logger.info("get_places returned {} candidates", result.count("\n"))
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

    def get_answer(self, a, b, w_type, time, pace, budget, extra_notes, model_name=DEFAULT_MODEL, lang="ru"):
        constraints = self.build_distance_budget(a, b, time, pace)
        dist = constraints["direct_distance_m"]
        route_types = {
            "friendly": "friendly walk",
            "romantic": "romantic walk",
            "family": "family walk",
            "active": "active walk",
            "cozy": "cozy walk",
            "cultural": "cultural walk",
        }
        output_language = "Russian" if lang == "ru" else "English"
        extra_notes_text = (
            f"Additional user preferences: {extra_notes}."
            if extra_notes.strip()
            else ""
        )

        system_prompt = f"You are planning a pedestrian route in Sirius, Russia. \
Build a route for a \"{route_types.get(w_type, 'friendly walk')}\" that starts at {a.street} and ends at {b.street}. \
The route should take about {time} minutes. The direct distance between {a.street} and {b.street} is {dist} meters. \
This is a {constraints['pace_label']} leisure walk, not fast transit walking. \
Assume an effective walking speed of about {constraints['speed_m_per_min']} meters per minute, with time reserved for short stops and browsing. \
The estimated walking time budget is about {constraints['walking_minutes']} minutes. The maximum total walking distance for the full route is {constraints['distance_budget_m']} meters, including all intermediate stops. Do not exceed this limit. \
Budget: {budget} RUB per person. {extra_notes_text} The route must contain at most 5 intermediate stops and should be interesting and relevant for the user request. \
Include either one cafe or restaurant, or none at all, unless the user explicitly asks for more. \
All stops must be unique, and no intermediate stop may be the same as the start or end point. \
Use the get_places tool to search for suitable locations. Pass a query describing the type of places you need. \
The tool returns addresses, short descriptions, ids, and distances to the start and end points in meters. \
You must use those distances to choose an efficient order. Prefer nearby good options over far away options. \
Prefer dense clusters of places near the direct path instead of stretching the route just to fill time. \
The final route must not be longer than the user requested, and should not be much shorter either. \
When you are ready, call the message tool with a route description and the array of selected point ids. \
Make only one tool call per assistant turn. Use no more than 5 tool calls before the final message. \
Do not include the start or end point ids in the final route. Minimize total walking distance while still making the route interesting. \
If the route includes a cafe, mention its name in the route description. \
Return the final route description in {output_language}. Keep place names and addresses in their original form when needed."

        desc_ans = "no-description"
        ans_id = []
        prompt = system_prompt
        messages = [{"role": "system", "content": self.model.start_message}]
        cnt = 0

        while cnt < 10:
            cnt += 1
            tool_results, desc_ans, ans_id = self.answer_model(
                messages, desc_ans, ans_id, a, b, prompt, model_name
            )
            messages.extend(tool_results)
            if ans_id:
                ans_id = self.normalize_point_ids(ans_id, b)
                total_distance = self.calculate_route_distance(a, ans_id, b)
                if total_distance is not None and total_distance <= constraints["distance_budget_m"]:
                    break

                logger.warning(
                    "Rejected route: total distance {} exceeds budget {}",
                    total_distance,
                    constraints["distance_budget_m"],
                )
                ans_id = []
                desc_ans = "no-description"
                exceeded_distance = (
                    f"{total_distance} meters"
                    if total_distance is not None
                    else "an invalid distance"
                )
                prompt = (
                    f"The previous route was invalid because its total walking distance was {exceeded_distance}, "
                    f"but the hard limit is {constraints['distance_budget_m']} meters. "
                    "Try again with fewer or closer stops, staying near the direct path."
                )
                continue

            prompt = ""

        self.model.clear_history(messages)
        if not ans_id or desc_ans == "no-description":
            logger.warning("Falling back to deterministic route generation")
            desc_ans, ans_id = self.build_fallback_route(
                a,
                b,
                w_type,
                extra_notes,
                constraints,
                lang,
            )
        return desc_ans, list(ans_id)

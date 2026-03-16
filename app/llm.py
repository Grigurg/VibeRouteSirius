import json
import re
from pathlib import Path

from loguru import logger
from openai import AuthenticationError, OpenAI

from app.data_access import canonicalize_category_name, normalize_text

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

CANONICAL_CATEGORIES = [
    "park",
    "cafe",
    "restaurant",
    "bar",
    "nightlife",
    "entertainment",
    "promenade",
    "beach",
    "viewpoint",
    "museum",
    "artwork",
    "attraction",
    "sports",
    "playground",
    "religious_site",
]

SEQUENCE_PATTERNS = (
    re.compile(r"сначала\s+(.+?)(?:,\s*|\s+а\s+потом\s+|\s+потом\s+)(.+?)(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"first\s+(.+?)(?:,\s*|\s+and then\s+|\s+then\s+)(.+?)(?:[.!?]|$)", re.IGNORECASE),
)

AVOID_PATTERNS = (
    re.compile(r"без\s+([^.,;!?]+)", re.IGNORECASE),
    re.compile(r"avoid\s+([^.,;!?]+)", re.IGNORECASE),
    re.compile(r"no\s+([^.,;!?]+)", re.IGNORECASE),
)

PREFERENCE_KEYWORDS = {
    "quiet": ["quiet", "calm", "тихо", "спокой", "без шума"],
    "scenic": ["scenic", "view", "вид", "панорам", "красив"],
    "romantic": ["romantic", "романт", "свидание"],
    "slow": ["slow", "unhurried", "медлен", "не спеша", "нетороп"],
    "cozy": ["cozy", "уют", "лампов"],
    "family": ["family", "семейн", "с детьми", "children"],
    "active": ["active", "sport", "актив", "энерг"],
    "cultural": ["cultural", "culture", "культур", "музей"],
}

CATEGORY_SYNONYMS = {
    "park": ["park", "парк", "garden", "сквер"],
    "cafe": ["cafe", "coffee", "кофе", "кафе", "кофейня"],
    "restaurant": ["restaurant", "еда", "food", "ресторан", "fast food", "fast_food"],
    "bar": ["bar", "бар", "pub", "кальян", "hookah"],
    "nightlife": ["nightclub", "club", "ночной клуб", "ночная жизнь"],
    "entertainment": ["entertainment", "развлеч", "cinema", "quest", "escape game"],
    "promenade": ["promenade", "embankment", "набереж", "променад", "seafront"],
    "beach": ["beach", "пляж"],
    "viewpoint": ["viewpoint", "смотров", "обзор"],
    "museum": ["museum", "музей"],
    "artwork": ["artwork", "арт", "скульптур", "памятник"],
    "attraction": ["attraction", "достопримеч", "theme park", "theme_park"],
    "sports": ["sports", "sport", "спорт", "stadium"],
    "playground": ["playground", "детская площадка", "children area"],
    "religious_site": ["church", "temple", "церковь", "храм"],
}

REASON_TRANSLATIONS = {
    "respects the requested stop order": {
        "ru": "сохранять запрошенный порядок остановок",
        "en": "respect the requested stop order",
    },
    "balances different types of places": {
        "ru": "сбалансировать разные типы мест",
        "en": "balance different types of places",
    },
    "keeps the route coherent and close to the direct path": {
        "ru": "сохранять логичный маршрут рядом с естественным путём",
        "en": "keep the route coherent and close to the direct path",
    },
    "avoids major detours": {
        "ru": "избегать больших крюков",
        "en": "avoid major detours",
    },
    "stays within the budget": {
        "ru": "уложиться в бюджет",
        "en": "stay within the budget",
    },
    "fills the requested walk duration reasonably well": {
        "ru": "хорошо заполнить запрошенную длительность прогулки",
        "en": "fill the requested walk duration reasonably well",
    },
    "provides the best trade-off between relevance and efficiency": {
        "ru": "сбалансировать релевантность и эффективность",
        "en": "provide the best trade-off between relevance and efficiency",
    },
    "keeps a simple direct walk between the requested endpoints": {
        "ru": "оставить простую прямую прогулку между выбранными точками",
        "en": "keep a simple direct walk between the requested endpoints",
    },
    "avoids unnecessary spending": {
        "ru": "избежать лишних трат",
        "en": "avoid unnecessary spending",
    },
    "uses the best available stops even if not every strict preference could be satisfied": {
        "ru": "использовать лучшие доступные остановки, даже если не все строгие пожелания удалось выполнить",
        "en": "use the best available stops even if not every strict preference could be satisfied",
    },
}


def load_openrouter_api_key():
    if not API_KEY_FILE.exists():
        return None

    value = API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not value:
        return None

    return value


def empty_parse_result():
    return {
        "hard_constraints": {
            "must_visit_in_order": [],
            "avoid_categories": [],
            "required_categories": [],
            "food_limit": None,
            "max_intermediate_stops": None,
            "max_total_budget_rub": None,
            "max_total_duration_min": None,
            "max_total_distance_m": None,
        },
        "soft_preferences": {
            "optional_preferences": [],
            "vibe": [],
            "preferred_categories": [],
        },
        "notes_summary": "",
    }


def normalize_constraint_term(value):
    normalized = normalize_text(value)
    if not normalized:
        return ""
    canonical = canonicalize_category_name(normalized)
    return canonical or normalized


def extract_json_object(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def detect_categories(text):
    normalized = normalize_text(text)
    if not normalized:
        return []

    detected = []
    for category, synonyms in CATEGORY_SYNONYMS.items():
        if any(synonym in normalized for synonym in synonyms):
            detected.append(category)
    return detected


def detect_preferences(text):
    normalized = normalize_text(text)
    if not normalized:
        return []

    matches = []
    for preference, keywords in PREFERENCE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            matches.append(preference)
    return matches


def localize_reason(reason, lang):
    translations = REASON_TRANSLATIONS.get(reason)
    if translations:
        return translations.get(lang, reason)
    return reason


class OpenRouterChat:
    def __init__(self):
        self.start_message = "You are an AI assistant that extracts structured route constraints and narrates routes."
        api_key = load_openrouter_api_key()
        self.available = bool(api_key)
        self.client = None
        if self.available:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )

    def complete(self, messages, model_name=DEFAULT_MODEL, temperature=0.2, max_tokens=1800):
        if not self.available or self.client is None:
            raise RuntimeError("OpenRouter API key is not configured.")

        selected_model = OPENROUTER_MODELS.get(model_name, OPENROUTER_MODELS[DEFAULT_MODEL])
        try:
            response = self.client.chat.completions.create(
                model=selected_model["openrouter_id"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except AuthenticationError as exc:
            raise RuntimeError(
                "OpenRouter authentication failed. Check OPENROUTER_API_KEY or the api_key file."
            ) from exc

        return response.choices[0].message.content or ""


class LLMAgent:
    def __init__(self):
        self.model = OpenRouterChat()

    def heuristic_parse_user_request(self, extra_notes):
        result = empty_parse_result()
        text = normalize_text(extra_notes)
        if not text:
            return result

        result["notes_summary"] = extra_notes.strip()

        for pattern in SEQUENCE_PATTERNS:
            match = pattern.search(extra_notes)
            if not match:
                continue
            ordered_terms = []
            for chunk in match.groups():
                categories = detect_categories(chunk)
                if categories:
                    ordered_terms.append(categories[0])
                else:
                    normalized = normalize_constraint_term(chunk)
                    if normalized:
                        ordered_terms.append(normalized)
            if ordered_terms:
                result["hard_constraints"]["must_visit_in_order"] = ordered_terms
                break

        avoid_categories = []
        for pattern in AVOID_PATTERNS:
            for match in pattern.findall(extra_notes):
                detected = detect_categories(match)
                if detected:
                    avoid_categories.extend(detected)
                else:
                    normalized = normalize_constraint_term(match)
                    if normalized:
                        avoid_categories.append(normalized)
        result["hard_constraints"]["avoid_categories"] = list(dict.fromkeys(avoid_categories))

        categories = detect_categories(extra_notes)
        if categories:
            preferred_categories = [
                category
                for category in categories
                if category not in result["hard_constraints"]["must_visit_in_order"]
                and category not in result["hard_constraints"]["avoid_categories"]
            ]
            result["soft_preferences"]["preferred_categories"] = list(dict.fromkeys(preferred_categories))

        preferences = detect_preferences(extra_notes)
        if preferences:
            result["soft_preferences"]["optional_preferences"] = preferences
            result["soft_preferences"]["vibe"] = [
                item for item in preferences if item in {"romantic", "cozy", "family", "active", "cultural", "slow"}
            ]

        if "не больше одного кафе" in text or "one cafe" in text or "at most one cafe" in text:
            result["hard_constraints"]["food_limit"] = 1
        elif "без кафе" in text or "no cafe" in text:
            result["hard_constraints"]["food_limit"] = 0

        max_stops_match = re.search(r"(?:максимум|не больше|at most|max)\s+(\d+)\s+(?:останов|stops?)", text)
        if max_stops_match:
            result["hard_constraints"]["max_intermediate_stops"] = int(max_stops_match.group(1))

        return result

    def parse_user_request(
        self,
        extra_notes,
        vibe,
        total_minutes,
        pace,
        budget_rub,
        lang="ru",
        model_name=DEFAULT_MODEL,
    ):
        base_result = self.heuristic_parse_user_request(extra_notes)
        if not extra_notes or not extra_notes.strip():
            return base_result

        if not self.model.available:
            return base_result

        system_prompt = (
            "You extract route planning constraints from user text. "
            "Return JSON only. Distinguish hard constraints from soft preferences carefully. "
            "If the user explicitly specifies an order like 'first park, then cafe', put it into hard_constraints.must_visit_in_order. "
            "If the user phrases something softly like 'would be nice to grab coffee', keep it in soft_preferences.preferred_categories or optional_preferences. "
            "Use canonical categories when possible. Allowed canonical categories: "
            + ", ".join(CANONICAL_CATEGORIES)
            + ". "
            "If the user mentions a place type outside that list, keep it as a lowercase keyword."
        )
        user_prompt = (
            "Parse the following route request into JSON with exactly these keys:\n"
            "{\n"
            '  "hard_constraints": {\n'
            '    "must_visit_in_order": [],\n'
            '    "avoid_categories": [],\n'
            '    "required_categories": [],\n'
            '    "food_limit": null,\n'
            '    "max_intermediate_stops": null,\n'
            '    "max_total_budget_rub": null,\n'
            '    "max_total_duration_min": null,\n'
            '    "max_total_distance_m": null\n'
            "  },\n"
            '  "soft_preferences": {\n'
            '    "optional_preferences": [],\n'
            '    "vibe": [],\n'
            '    "preferred_categories": []\n'
            "  },\n"
            '  "notes_summary": ""\n'
            "}\n\n"
            f"Explicit form context:\n"
            f"- vibe: {vibe}\n"
            f"- duration_minutes: {total_minutes}\n"
            f"- pace: {pace}\n"
            f"- budget_rub: {budget_rub}\n"
            f"- output_language: {'Russian' if lang == 'ru' else 'English'}\n\n"
            f"User text:\n{extra_notes.strip()}"
        )

        try:
            response = self.model.complete(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model_name=model_name,
                temperature=0.0,
                max_tokens=1200,
            )
            parsed = extract_json_object(response)
            if not isinstance(parsed, dict):
                logger.warning("Failed to parse LLM request JSON, using heuristic parse")
                return base_result

            merged = empty_parse_result()
            merged["notes_summary"] = parsed.get("notes_summary") or base_result["notes_summary"]

            for key in merged["hard_constraints"]:
                value = parsed.get("hard_constraints", {}).get(key)
                merged["hard_constraints"][key] = value if value not in ("", []) else merged["hard_constraints"][key]
            for key in merged["soft_preferences"]:
                value = parsed.get("soft_preferences", {}).get(key)
                merged["soft_preferences"][key] = value if value not in ("", []) else merged["soft_preferences"][key]

            for key, value in base_result["hard_constraints"].items():
                if merged["hard_constraints"].get(key) in (None, [], ""):
                    merged["hard_constraints"][key] = value
            for key, value in base_result["soft_preferences"].items():
                if merged["soft_preferences"].get(key) in (None, [], ""):
                    merged["soft_preferences"][key] = value

            for field_name in ("must_visit_in_order", "avoid_categories", "required_categories"):
                merged["hard_constraints"][field_name] = [
                    normalize_constraint_term(item)
                    for item in merged["hard_constraints"].get(field_name, [])
                    if normalize_constraint_term(item)
                ]
            for field_name in ("optional_preferences", "vibe", "preferred_categories"):
                merged["soft_preferences"][field_name] = [
                    normalize_constraint_term(item)
                    for item in merged["soft_preferences"].get(field_name, [])
                    if normalize_constraint_term(item)
                ]

            return merged
        except Exception as exc:
            logger.warning("LLM parse failed, using heuristic parse: {}", exc)
            return base_result

    def build_fallback_narration(self, request, route_plan):
        stop_names = [point.name or point.street for point in route_plan.stop_points]
        cafe_names = [
            point.name or point.street
            for point in route_plan.stop_points
            if "cafe" in point.categories or "restaurant" in point.categories
        ]

        if request.lang == "ru":
            if stop_names:
                description = (
                    f"Маршрут начинается в {request.start.street} и ведёт к {request.end.street} через "
                    + ", ".join(stop_names)
                    + ". "
                )
            else:
                description = (
                    f"Маршрут проходит напрямую от {request.start.street} до {request.end.street}. "
                )

            if route_plan.route_reasons:
                localized = [localize_reason(reason, "ru") for reason in route_plan.route_reasons]
                description += "Он подобран так, чтобы " + "; ".join(localized) + ". "
            if cafe_names:
                description += "По пути есть остановка с кофе или едой: " + ", ".join(cafe_names) + ". "
            return description.strip()

        if stop_names:
            description = (
                f"The route starts at {request.start.street} and finishes at {request.end.street}, passing through "
                + ", ".join(stop_names)
                + ". "
            )
        else:
            description = f"The route goes directly from {request.start.street} to {request.end.street}. "

        if route_plan.route_reasons:
            localized = [localize_reason(reason, "en") for reason in route_plan.route_reasons]
            description += "It was selected to " + "; ".join(localized) + ". "
        if cafe_names:
            description += "Food or coffee stops on the route: " + ", ".join(cafe_names) + ". "
        return description.strip()

    def narrate_route(self, request, route_plan, model_name=DEFAULT_MODEL):
        payload = {
            "language": request.lang,
            "start": {
                "name": request.start.name or request.start.street,
                "address": request.start.street,
            },
            "finish": {
                "name": request.end.name or request.end.street,
                "address": request.end.street,
            },
            "ordered_stops": [
                {
                    "name": point.name or point.street,
                    "address": point.street,
                    "categories": list(point.categories),
                    "description": point.desc,
                    "why_selected": route_plan.stop_reasons.get(point.id, []),
                }
                for point in route_plan.stop_points
            ],
            "route_reasons": route_plan.route_reasons,
            "total_distance_m": route_plan.total_distance_m,
            "estimated_duration_min": route_plan.estimated_duration_min,
        }

        if not self.model.available:
            return self.build_fallback_narration(request, route_plan)

        system_prompt = (
            "You narrate pedestrian routes. You may only describe the provided route. "
            "Do not add, remove, replace, or reorder stops. "
            "Do not invent any extra locations. "
            "Briefly explain why the route fits the request and mention key stops, including cafes if present."
        )
        user_prompt = (
            "Narrate this exact route in "
            + ("Russian" if request.lang == "ru" else "English")
            + ". Keep place names and addresses in their original language when needed.\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

        try:
            response = self.model.complete(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model_name=model_name,
                temperature=0.3,
                max_tokens=900,
            )
            return response.strip() or self.build_fallback_narration(request, route_plan)
        except Exception as exc:
            logger.warning("LLM narration failed, using deterministic text: {}", exc)
            return self.build_fallback_narration(request, route_plan)

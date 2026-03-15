from functools import lru_cache

from app.data_access import load_database
from app.llm import DEFAULT_MODEL, LLMAgent, OPENROUTER_MODELS
from app.models import Object
from app.route_engine import DeterministicRoutePlanner

LANGUAGES = {
    "ru": "Русский",
    "en": "English",
}

UI_TEXT = {
    "ru": {
        "html_lang": "ru",
        "title": "VibeRoute — Твой идеальный маршрут, созданный AI",
        "tagline": "Твой идеальный маршрут, созданный AI",
        "about_button": "О проекте",
        "about_text": "VibeRoute — сервис планирования прогулок и маршрутов по настроению, бюджету и времени. Powered by AI.",
        "language_label": "Язык",
        "start_label": "Откуда начать?",
        "start_placeholder": "Введите адрес...",
        "start_geo_title": "Использовать геопозицию",
        "end_label": "Куда хочешь прийти?",
        "end_placeholder": "Введите адрес...",
        "end_geo_title": "Указать конечную точку по геопозиции",
        "duration_label": "Длительность прогулки",
        "duration_hours": "ч",
        "duration_minutes": "мин",
        "pace_label": "Темп прогулки",
        "budget_label": "Бюджет на человека",
        "budget_hint": "(₽)",
        "budget_economy": "Эконом",
        "budget_medium": "Средний",
        "budget_comfort": "Комфортный",
        "budget_unlimited": "Безлимитный",
        "vibe_label": "Характер прогулки",
        "notes_label": "Свои пожелания",
        "notes_placeholder": "Опишите ваши пожелания...",
        "model_label": "Выберите модель для генерации маршрута",
        "submit_idle": "Создать Маршрут",
        "submit_loading": "Генерируем...",
        "result_title": "Твой идеальный маршрут!",
        "metric_vibe": "Тип прогулки",
        "metric_duration": "Длительность",
        "metric_pace": "Темп",
        "metric_distance": "Дистанция",
        "metric_budget": "Бюджет",
        "steps_title": "Пошаговый план маршрута",
        "route_badge": "Описание маршрута",
        "place_fallback": "Место",
        "marker_start": "Старт",
        "marker_end": "Конец",
        "marker_end_point": "Конечная точка",
        "geo_unsupported": "Геолокация не поддерживается браузером.",
        "map_not_ready": "Карта ещё не загружена. Подождите немного и попробуйте снова.",
        "geo_failed": "Не удалось получить геопозицию.",
        "geo_denied": "Доступ к геолокации запрещён в настройках браузера.",
        "geo_unavailable": "Геопозиция недоступна.",
        "geo_timeout": "Превышено время ожидания запроса геолокации.",
    },
    "en": {
        "html_lang": "en",
        "title": "VibeRoute — Your perfect AI-generated route",
        "tagline": "Your perfect AI-generated route",
        "about_button": "About",
        "about_text": "VibeRoute is a walk and route planning service based on mood, budget, and time. Powered by AI.",
        "language_label": "Language",
        "start_label": "Where do you want to start?",
        "start_placeholder": "Enter an address...",
        "start_geo_title": "Use current location",
        "end_label": "Where do you want to finish?",
        "end_placeholder": "Enter an address...",
        "end_geo_title": "Set destination from geolocation",
        "duration_label": "Walk duration",
        "duration_hours": "h",
        "duration_minutes": "min",
        "pace_label": "Walking pace",
        "budget_label": "Budget per person",
        "budget_hint": "(₽)",
        "budget_economy": "Budget",
        "budget_medium": "Standard",
        "budget_comfort": "Comfort",
        "budget_unlimited": "Unlimited",
        "vibe_label": "Walk style",
        "notes_label": "Extra preferences",
        "notes_placeholder": "Describe your preferences...",
        "model_label": "Choose a model for route generation",
        "submit_idle": "Create Route",
        "submit_loading": "Generating...",
        "result_title": "Your perfect route!",
        "metric_vibe": "Walk style",
        "metric_duration": "Duration",
        "metric_pace": "Pace",
        "metric_distance": "Distance",
        "metric_budget": "Budget",
        "steps_title": "Step-by-step route plan",
        "route_badge": "Route description",
        "place_fallback": "Place",
        "marker_start": "Start",
        "marker_end": "End",
        "marker_end_point": "Destination",
        "geo_unsupported": "Geolocation is not supported by this browser.",
        "map_not_ready": "The map is not ready yet. Please wait a bit and try again.",
        "geo_failed": "Failed to get your location.",
        "geo_denied": "Geolocation access is blocked in the browser settings.",
        "geo_unavailable": "Geolocation is unavailable.",
        "geo_timeout": "The geolocation request timed out.",
    },
}

VIBE_LABELS = {
    "friendly": {"emoji": "🤝", "ru": "Дружеская", "en": "Friendly"},
    "romantic": {"emoji": "❤️", "ru": "Романтическая", "en": "Romantic"},
    "family": {"emoji": "👨‍👩‍👧‍👦", "ru": "Семейная", "en": "Family"},
    "cultural": {"emoji": "🏛️", "ru": "Культурная", "en": "Cultural"},
    "active": {"emoji": "🚴", "ru": "Активная", "en": "Active"},
    "cozy": {"emoji": "☕", "ru": "Спокойная / Уютная", "en": "Cozy / Relaxed"},
}

VIBES = list(VIBE_LABELS.keys())
PACE_LABELS = {
    "relaxed": {"emoji": "🌿", "ru": "Спокойный", "en": "Relaxed"},
    "normal": {"emoji": "🚶", "ru": "Обычный", "en": "Normal"},
    "active": {"emoji": "⚡", "ru": "Бодрый", "en": "Active"},
}

PACE_OPTIONS = list(PACE_LABELS.keys())
PLACES = [
    {
        "name": "Шоколадница",
        "amenity": "cafe",
        "coordinates": [39.9262057, 43.4272589],
    },
    {
        "name": "O'Sullivan's Irish Pub",
        "amenity": "pub",
        "coordinates": [39.9755125, 43.3964213],
    },
]
MODEL_OPTIONS = [
    (model_key, model_config["label"])
    for model_key, model_config in OPENROUTER_MODELS.items()
]


def normalize_model(model_name):
    if model_name in OPENROUTER_MODELS:
        return model_name
    return DEFAULT_MODEL


def normalize_language(lang):
    if lang in LANGUAGES:
        return lang
    return "ru"


def normalize_pace(pace):
    if pace in PACE_OPTIONS:
        return pace
    return "relaxed"


def get_ui_text(lang):
    return UI_TEXT[normalize_language(lang)]


def get_vibes(lang):
    normalized_lang = normalize_language(lang)
    return [
        (vibe_key, vibe_data["emoji"], vibe_data[normalized_lang])
        for vibe_key, vibe_data in VIBE_LABELS.items()
    ]


def get_paces(lang):
    normalized_lang = normalize_language(lang)
    return [
        (pace_key, pace_data["emoji"], pace_data[normalized_lang])
        for pace_key, pace_data in PACE_LABELS.items()
    ]


@lru_cache(maxsize=1)
def get_agent():
    return LLMAgent()


@lru_cache(maxsize=1)
def get_database():
    return load_database()


@lru_cache(maxsize=1)
def get_route_engine():
    return DeterministicRoutePlanner(get_database())


def parse_form(req_form):
    lang = normalize_language(req_form.get("lang", "ru"))
    selected_model = normalize_model(req_form.get("model", DEFAULT_MODEL))

    return {
        "lang": lang,
        "start_addr": req_form.get("start_addr", ""),
        "end_addr": req_form.get("end_addr", ""),
        "start_lat": req_form.get("start_lat", ""),
        "start_lng": req_form.get("start_lng", ""),
        "end_lat": req_form.get("end_lat", ""),
        "end_lng": req_form.get("end_lng", ""),
        "duration_hrs": int(req_form.get("duration_hrs", 2)),
        "duration_mins": int(req_form.get("duration_mins", 0)),
        "pace": normalize_pace(req_form.get("pace", "relaxed")),
        "budget": int(req_form.get("budget", 2000)),
        "vibe": req_form.get("vibe", "friendly"),
        "extra_notes": req_form.get("extra_notes", ""),
        "model": selected_model,
        "map_lat": req_form.get("map_lat", ""),
        "map_lng": req_form.get("map_lng", ""),
        "map_zoom": req_form.get("map_zoom", ""),
    }


def get_start_location_coords(req_form):
    lat_str = req_form.get("start_lat", "")
    lng_str = req_form.get("start_lng", "")
    if lat_str and lng_str:
        try:
            return {"lat": float(lat_str), "lng": float(lng_str)}
        except (ValueError, TypeError):
            return None
    return None


def get_end_location_coords(req_form):
    lat_str = req_form.get("end_lat", "")
    lng_str = req_form.get("end_lng", "")
    if lat_str and lng_str:
        try:
            return {"lat": float(lat_str), "lng": float(lng_str)}
        except (ValueError, TypeError):
            return None
    return None


def get_places(formdata):
    global PLACES
    agent = get_agent()
    route_engine = get_route_engine()
    start_object = Object(float(formdata["start_lng"]), float(formdata["start_lat"]), formdata["start_addr"])
    end_object = Object(float(formdata["end_lng"]), float(formdata["end_lat"]), formdata["end_addr"])
    time_minutes = formdata["duration_hrs"] * 60 + formdata["duration_mins"]

    parsed_request = agent.parse_user_request(
        formdata["extra_notes"],
        formdata["vibe"],
        time_minutes,
        formdata["pace"],
        formdata["budget"],
        formdata.get("lang", "ru"),
        formdata["model"],
    )
    planning_request = route_engine.build_request(
        start_object,
        end_object,
        formdata["vibe"],
        time_minutes,
        formdata["pace"],
        formdata["budget"],
        formdata["extra_notes"],
        formdata.get("lang", "ru"),
        formdata["model"],
        parsed_request,
    )
    route_plan = route_engine.plan(planning_request)
    description = agent.narrate_route(planning_request, route_plan, formdata["model"])

    PLACES.clear()
    for point in route_plan.stop_points:
        PLACES.append(
            {
                "name": point.name,
                "amenity": point.amenity or point.primary_category,
                "coordinates": [point.x, point.y],
                "lat": point.y,
                "lng": point.x,
                "desc": point.desc,
                "budget": f"~{point.estimated_cost_rub} ₽" if point.estimated_cost_rub else "",
            }
        )
    return PLACES, description, planning_request, route_plan


def get_vibe_verbose(vibe):
    return VIBE_LABELS.get(vibe, VIBE_LABELS["friendly"])


def get_pace_verbose(pace):
    return PACE_LABELS.get(pace, PACE_LABELS["relaxed"])


def shorten_text(text, limit=180):
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_route_steps(formdata, planning_request, route_plan):
    steps = []
    lang = normalize_language(formdata.get("lang", "ru"))
    ordered_points = [planning_request.start] + route_plan.stop_points + [planning_request.end]

    for index, point in enumerate(ordered_points):
        if index == 0:
            desc = "Старт маршрута." if lang == "ru" else "Route start."
        elif index == len(ordered_points) - 1:
            desc = "Финиш маршрута." if lang == "ru" else "Route destination."
        else:
            reasons = route_plan.stop_reasons.get(point.id, [])
            reason_text = "; ".join(reasons)
            if lang == "ru":
                desc = shorten_text(point.desc)
                if reason_text:
                    desc += f" Почему выбрано: {reason_text}."
            else:
                desc = shorten_text(point.desc)
                if reason_text:
                    desc += f" Why it was selected: {reason_text}."

        budget = f"~{point.estimated_cost_rub} ₽" if point.estimated_cost_rub else ""
        map_link = f"https://yandex.ru/maps/?ll={point.x},{point.y}&z=16"
        segment = ""
        if index < len(ordered_points) - 1:
            next_point = ordered_points[index + 1]
            leg_distance_m = point.dist_between_points(next_point)
            leg_minutes = round(leg_distance_m / max(planning_request.speed_m_per_min, 1))
            if lang == "ru":
                segment = f"До следующей точки: {leg_minutes} минут пешком ({leg_distance_m / 1000:.1f} км)"
            else:
                segment = f"To the next stop: {leg_minutes} minutes on foot ({leg_distance_m / 1000:.1f} km)"

        steps.append(
            {
                "name": point.name or point.street,
                "desc": desc,
                "budget": budget,
                "img": "",
                "map_link": map_link,
                "segment": segment,
            }
        )
    return steps


def demo_tips(formdata, route_plan=None):
    used_budget = route_plan.estimated_cost_rub if route_plan else 1400
    rest = max(formdata["budget"] - used_budget, 0)
    if normalize_language(formdata.get("lang", "ru")) == "en":
        return (
            f"You can spend the remaining budget of <span style='font-weight:bold'>{rest} ₽</span> on dessert at a cafe near the final point or on souvenirs."
            "<br>Extra tip: bring a power bank so you do not miss great photo spots.<br>"
        )
    return (
        f"Остаток бюджета <span style='font-weight:bold'>{rest} ₽</span> можно потратить на десерт в кофейне у конечной точки или на покупку сувениров."
        "<br>Дополнительно: Возьмите power bank, чтобы не пропустить красивые фото!<br>"
    )


def build_summary(formdata, planning_request=None, route_plan=None):
    lang = normalize_language(formdata.get("lang", "ru"))
    steps = build_route_steps(formdata, planning_request, route_plan) if planning_request and route_plan else []
    ui = get_ui_text(lang)
    return {
        "vibe_verbose": get_vibe_verbose(formdata["vibe"])[lang],
        "pace_verbose": get_pace_verbose(formdata.get("pace", "relaxed"))[lang],
        "duration_str": (
            f"{formdata['duration_hrs']} {ui['duration_hours']} {formdata['duration_mins']} {ui['duration_minutes']}"
            if formdata["duration_mins"]
            else f"{formdata['duration_hrs']} {ui['duration_hours']}"
        ),
        "budget": formdata["budget"],
        "model": formdata["model"],
        "distance": None,
        "steps": steps,
        "tips": demo_tips(formdata, route_plan),
    }

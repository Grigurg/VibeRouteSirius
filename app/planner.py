import random
from functools import lru_cache

from app.data_access import load_database
from app.llm import DEFAULT_MODEL, LLMAgent, OPENROUTER_MODELS
from app.models import Object

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
PLACES = [
    {
        "type": "Feature",
        "properties": {"amenity": "cafe", "name": "Шоколадница"},
        "geometry": {"type": "Point", "coordinates": [39.9262057, 43.4272589]},
    },
    {
        "type": "Feature",
        "properties": {"amenity": "pub", "name": "O'Sullivan's Irish Pub"},
        "geometry": {"type": "Point", "coordinates": [39.9755125, 43.3964213]},
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


def get_ui_text(lang):
    return UI_TEXT[normalize_language(lang)]


def get_vibes(lang):
    normalized_lang = normalize_language(lang)
    return [
        (vibe_key, vibe_data["emoji"], vibe_data[normalized_lang])
        for vibe_key, vibe_data in VIBE_LABELS.items()
    ]


@lru_cache(maxsize=1)
def get_agent():
    return LLMAgent()


@lru_cache(maxsize=1)
def get_database():
    return load_database()


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
    database = get_database()
    agent = get_agent()
    start_object = Object(float(formdata["start_lat"]), float(formdata["start_lng"]), formdata["start_addr"])
    end_object = Object(float(formdata["end_lat"]), float(formdata["end_lng"]), formdata["end_addr"])
    time_minutes = formdata["duration_hrs"] * 60 + formdata["duration_mins"]

    description, indices = agent.get_answer(
        start_object,
        end_object,
        formdata["vibe"],
        time_minutes,
        formdata["budget"],
        formdata["extra_notes"],
        formdata["model"],
        formdata.get("lang", "ru"),
    )

    PLACES.clear()
    for index in indices:
        if 0 <= index < len(database):
            point = database[index]
            PLACES.append(
                {
                    "name": point.name,
                    "amenity": point.amenity,
                    "coordinates": [point.x, point.y],
                }
            )
    return PLACES, description


def get_vibe_verbose(vibe):
    return VIBE_LABELS.get(vibe, VIBE_LABELS["friendly"])


def demo_route_steps(formdata):
    points = []
    if formdata.get("start_addr"):
        points.append(formdata["start_addr"])
    if formdata.get("end_addr"):
        points.append(formdata["end_addr"])

    used = []
    steps = []
    vibe_data = get_vibe_verbose(formdata.get("vibe"))
    vibe_verbose = vibe_data[normalize_language(formdata.get("lang", "ru"))]

    for index, point_name in enumerate(points):
        if PLACES:
            available = [place for place in PLACES if place.get("name") not in used] or PLACES
            place = random.choice(available)
            title = place.get("name", get_ui_text(formdata.get("lang", "ru"))["place_fallback"])
            desc = place.get("desc", "")
            budget = place.get("budget", "")
            img = place.get("img", "https://placehold.co/300x150?text=Place")
            lat = place.get("lat")
            lng = place.get("lng")
            if lat is not None and lng is not None:
                map_link = f"https://yandex.ru/maps/?ll={lng},{lat}&z=16"
            else:
                map_link = f"https://yandex.ru/maps/?text={point_name}" if point_name else "#"
        else:
            title = ""
            desc = ""
            budget = ""
            img = ""
            map_link = f"https://yandex.ru/maps/?text={point_name}" if point_name else "#"

        used.append(title)
        if "%vibe%" in desc and vibe_verbose:
            desc = desc.replace("%vibe%", vibe_verbose)

        steps.append(
            {
                "name": point_name if point_name.strip() else title,
                "desc": desc,
                "budget": budget,
                "img": img,
                "map_link": map_link,
                "segment": (
                    (
                        f"Время в пути до следующей точки: {15 + 5 * index} минут пешком ({1.2 + 0.3 * index:.1f} км)"
                        if normalize_language(formdata.get("lang", "ru")) == "ru"
                        else f"Walking time to the next point: {15 + 5 * index} minutes ({1.2 + 0.3 * index:.1f} km)"
                    )
                    if index < len(points) - 1
                    else ""
                ),
            }
        )
    return steps


def demo_tips(formdata):
    rest = max(formdata["budget"] - 700 * 2, 0)
    if normalize_language(formdata.get("lang", "ru")) == "en":
        return (
            f"You can spend the remaining budget of <span style='font-weight:bold'>{rest} ₽</span> on dessert at a cafe near the final point or on souvenirs."
            "<br>Extra tip: bring a power bank so you do not miss great photo spots.<br>"
        )
    return (
        f"Остаток бюджета <span style='font-weight:bold'>{rest} ₽</span> можно потратить на десерт в кофейне у конечной точки или на покупку сувениров."
        "<br>Дополнительно: Возьмите power bank, чтобы не пропустить красивые фото!<br>"
    )


def build_summary(formdata):
    lang = normalize_language(formdata.get("lang", "ru"))
    steps = demo_route_steps(formdata)
    ui = get_ui_text(lang)
    return {
        "vibe_verbose": get_vibe_verbose(formdata["vibe"])[lang],
        "duration_str": (
            f"{formdata['duration_hrs']} {ui['duration_hours']} {formdata['duration_mins']} {ui['duration_minutes']}"
            if formdata["duration_mins"]
            else f"{formdata['duration_hrs']} {ui['duration_hours']}"
        ),
        "budget": formdata["budget"],
        "model": formdata["model"],
        "distance": f"{5.2 + random.randint(-1, 2) * 0.3:.1f}",
        "steps": steps,
        "tips": demo_tips(formdata),
    }

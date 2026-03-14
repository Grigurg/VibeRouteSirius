import random
from functools import lru_cache

from app.data_access import load_database
from app.llm import DEFAULT_MODEL, LLMAgent, OPENROUTER_MODELS
from app.models import Object

VIBES = [
    ("friendly", "🤝", "Дружеская"),
    ("romantic", "❤️", "Романтическая"),
    ("family", "👨‍👩‍👧‍👦", "Семейная"),
    ("cultural", "🏛️", "Культурная"),
    ("active", "🚴", "Активная"),
    ("cozy", "☕", "Спокойная / Уютная"),
]
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


@lru_cache(maxsize=1)
def get_agent():
    return LLMAgent()


@lru_cache(maxsize=1)
def get_database():
    return load_database()


def parse_form(req_form):
    selected_model = normalize_model(req_form.get("model", DEFAULT_MODEL))

    return {
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
    for item in VIBES:
        if item[0] == vibe:
            return item[2]
    return "Дружеская"


def demo_route_steps(formdata):
    points = []
    if formdata.get("start_addr"):
        points.append(formdata["start_addr"])
    if formdata.get("end_addr"):
        points.append(formdata["end_addr"])

    used = []
    steps = []
    vibe_map = {item[0]: item[2] for item in VIBES}
    vibe_verbose = vibe_map.get(formdata.get("vibe"), "")

    for index, point_name in enumerate(points):
        if PLACES:
            available = [place for place in PLACES if place.get("name") not in used] or PLACES
            place = random.choice(available)
            title = place.get("name", "Место")
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
                    f"Время в пути до следующей точки: {15 + 5 * index} минут пешком ({1.2 + 0.3 * index:.1f} км)"
                    if index < len(points) - 1
                    else ""
                ),
            }
        )
    return steps


def demo_tips(formdata):
    rest = max(formdata["budget"] - 700 * 2, 0)
    return (
        f"Остаток бюджета <span style='font-weight:bold'>{rest} ₽</span> можно потратить на десерт в кофейне у конечной точки или на покупку сувениров."
        "<br>Дополнительно: Возьмите power bank, чтобы не пропустить красивые фото!<br>"
    )


def build_summary(formdata):
    steps = demo_route_steps(formdata)
    return {
        "vibe_verbose": get_vibe_verbose(formdata["vibe"]),
        "duration_str": (
            f"{formdata['duration_hrs']} ч {formdata['duration_mins']} мин"
            if formdata["duration_mins"]
            else f"{formdata['duration_hrs']} ч"
        ),
        "budget": formdata["budget"],
        "model": formdata["model"],
        "distance": f"{5.2 + random.randint(-1, 2) * 0.3:.1f}",
        "steps": steps,
        "tips": demo_tips(formdata),
    }

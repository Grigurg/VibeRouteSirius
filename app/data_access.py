import os
import json
import re
from functools import cmp_to_key
from math import cos, radians, sqrt
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

from app.models import Object

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
EMBEDDINGS_FILE = DATA_DIR / "points_embeddings.txt"
DATABASE_FILE = DATA_DIR / "sirius_super_cool.geojson"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

FOOD_CATEGORIES = {"cafe", "restaurant", "bar", "nightlife"}

CATEGORY_ALIASES = {
    "кафе": "cafe",
    "кофейня": "cafe",
    "coffee shop": "cafe",
    "coffee_shop": "cafe",
    "cafe": "cafe",
    "ice cream": "cafe",
    "ice_cream": "cafe",
    "restaurant": "restaurant",
    "ресторан": "restaurant",
    "fast food": "restaurant",
    "fast_food": "restaurant",
    "food court": "restaurant",
    "бары": "bar",
    "бар": "bar",
    "bar": "bar",
    "pub": "bar",
    "hookah lounge": "bar",
    "hookah_lounge": "bar",
    "кальянная": "bar",
    "ночные клубы": "nightlife",
    "night club": "nightlife",
    "nightclub": "nightlife",
    "club": "nightlife",
    "развлекательные центры": "entertainment",
    "entertainment": "entertainment",
    "escape game": "entertainment",
    "escape_game": "entertainment",
    "amusement arcade": "entertainment",
    "amusement_arcade": "entertainment",
    "cinema": "entertainment",
    "theatre": "entertainment",
    "events venue": "entertainment",
    "events_venue": "entertainment",
    "park": "park",
    "парк": "park",
    "garden": "park",
    "сквер": "park",
    "beach resort": "beach",
    "beach_resort": "beach",
    "пляж": "beach",
    "beach": "beach",
    "resort": "beach",
    "viewpoint": "viewpoint",
    "смотровая": "viewpoint",
    "смотровая площадка": "viewpoint",
    "promenade": "promenade",
    "embankment": "promenade",
    "набережная": "promenade",
    "музей": "museum",
    "museum": "museum",
    "artwork": "artwork",
    "historic": "artwork",
    "memorial": "artwork",
    "памятник": "artwork",
    "attraction": "attraction",
    "theme park": "attraction",
    "theme_park": "attraction",
    "zoo": "attraction",
    "sports centre": "sports",
    "sports_centre": "sports",
    "sport": "sports",
    "stadium": "sports",
    "swimming pool": "sports",
    "swimming_pool": "sports",
    "sports hall": "sports",
    "sports_hall": "sports",
    "pitch": "sports",
    "playground": "playground",
    "детская площадка": "playground",
    "place of worship": "religious_site",
    "place_of_worship": "religious_site",
    "church": "religious_site",
    "cathedral": "religious_site",
}

KEYWORD_CATEGORY_RULES = (
    ("набереж", "promenade"),
    ("променад", "promenade"),
    ("embank", "promenade"),
    ("seafront", "promenade"),
    ("пляж", "beach"),
    ("beach", "beach"),
    ("смотров", "viewpoint"),
    ("viewpoint", "viewpoint"),
    ("кофе", "cafe"),
    ("кофейн", "cafe"),
    ("coffee", "cafe"),
    ("cappuccino", "cafe"),
    ("бар", "bar"),
    ("кальян", "bar"),
    ("pub", "bar"),
    ("музей", "museum"),
    ("museum", "museum"),
    ("парк", "park"),
    ("garden", "park"),
    ("арт", "artwork"),
    ("скульптур", "artwork"),
    ("памятник", "artwork"),
    ("art object", "artwork"),
    ("достопримеч", "attraction"),
    ("theme park", "attraction"),
)

PRICE_RANGE_RE = re.compile(r"(\d[\d ]*)\s*[–-]\s*(\d[\d ]*)\s*₽")
PRICE_FROM_RE = re.compile(r"от\s*(\d[\d ]*)\s*₽")

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("BAAI/bge-m3", local_files_only=True)
    return _embedding_model


def get_emb(sentences):
    return get_embedding_model().encode(sentences, batch_size=1)


def get_simularity(embeddings, q_embeddings):
    return get_embedding_model().similarity(embeddings, q_embeddings)


def normalize_text(text):
    if text is None:
        return ""
    return str(text).strip().lower()


def canonicalize_category_name(value):
    normalized = normalize_text(value).replace("-", " ").replace("_", " ")
    if not normalized:
        return None
    if normalized in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[normalized]
    compact = normalized.replace("  ", " ")
    if compact in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[compact]
    return None


def parse_float_value(value):
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", ".").strip())
    except ValueError:
        return None


def parse_int_value(value):
    if value in (None, ""):
        return None
    cleaned = re.sub(r"[^\d]", "", str(value))
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def parse_price_value(text):
    if not text:
        return None

    range_match = PRICE_RANGE_RE.search(text)
    if range_match:
        low = parse_int_value(range_match.group(1))
        high = parse_int_value(range_match.group(2))
        if low is not None and high is not None:
            return int(round((low + high) / 2))

    from_match = PRICE_FROM_RE.search(text.lower())
    if from_match:
        return parse_int_value(from_match.group(1))

    return None


def estimate_place_cost(props, info, categories):
    feat_text = info.get("feat_text", [])
    average_bill = None
    drink_price = None
    general_label = None

    for item in feat_text:
        text = str(item)
        lowered = text.lower()
        if "средний сч" in lowered or "average bill" in lowered:
            average_bill = parse_price_value(text)
        elif "капучино" in lowered or "coffee" in lowered or "пива" in lowered:
            drink_price = parse_price_value(text)
        elif "цены:" in lowered or "price level" in lowered:
            if "низк" in lowered or "low" in lowered:
                general_label = 450
            elif "выше среднего" in lowered or "above average" in lowered:
                general_label = 1700
            elif "высок" in lowered or "high" in lowered:
                general_label = 2500
            elif "средн" in lowered or "medium" in lowered:
                general_label = 1000

    if average_bill is not None:
        return average_bill
    if drink_price is not None and FOOD_CATEGORIES.intersection(categories):
        return drink_price
    return general_label


def extract_categories(props, info):
    categories = []

    for key in ("amenity", "leisure", "tourism", "historic", "sport", "shop", "attraction"):
        canonical = canonicalize_category_name(props.get(key))
        if canonical and canonical not in categories:
            categories.append(canonical)

    text_parts = [
        props.get("name"),
        props.get("amenity"),
        props.get("leisure"),
        props.get("tourism"),
        props.get("historic"),
        props.get("sport"),
        info.get("description"),
        " ".join(info.get("feat_text", [])),
    ]
    combined_text = " ".join(part for part in text_parts if part).lower()
    for keyword, category in KEYWORD_CATEGORY_RULES:
        if keyword in combined_text and category not in categories:
            categories.append(category)

    if not categories:
        categories.append("generic")

    return tuple(categories)


def build_text_blob(props, info, categories):
    fields = [
        props.get("name"),
        props.get("street"),
        props.get("amenity"),
        props.get("leisure"),
        props.get("tourism"),
        props.get("historic"),
        props.get("sport"),
        props.get("cuisine"),
        info.get("description"),
        " ".join(info.get("feat_text", [])),
        " ".join(categories),
    ]
    return ". ".join(str(value).strip() for value in fields if value)


def build_point_description(props, info, metadata):
    parts = [f"Название: {props.get('name', 'Без названия')}."]
    if props.get("street"):
        parts.append(f"Адрес: {props['street']}.")
    if metadata["categories"]:
        parts.append("Категории: " + ", ".join(metadata["categories"]) + ".")
    if metadata["rating_value"] is not None:
        parts.append(f"Рейтинг: {metadata['rating_value']:.1f}/5.")
    if metadata["review_count"]:
        parts.append(f"Отзывы: {metadata['review_count']}.")
    if metadata["popularity_score"]:
        parts.append(f"Popularity score: {metadata['popularity_score']}.")
    description = info.get("description")
    if description:
        parts.append(f"Описание: {description}.")
    feat_text = info.get("feat_text", [])
    if feat_text:
        parts.append("Факты: " + ", ".join(feat_text[:6]) + ".")
    return " ".join(parts)


def build_point_metadata(feature):
    props = feature.get("properties", {})
    info = feature.get("info", {})
    categories = extract_categories(props, info)
    metadata = {
        "categories": categories,
        "primary_category": categories[0] if categories else "generic",
        "rating_value": parse_float_value(info.get("rating")),
        "review_count": parse_int_value(info.get("reviews")) or 0,
        "popularity_score": parse_float_value(props.get("popularity_score")) or 0.0,
        "is_good_place": bool(info.get("is_good_place")),
        "estimated_cost_rub": estimate_place_cost(props, info, categories),
        "properties": props,
        "info": info,
        "text_blob": build_text_blob(props, info, categories),
        "normalized_name": normalize_text(props.get("name")),
        "is_food_venue": bool(FOOD_CATEGORIES.intersection(categories)),
    }
    return metadata


def build_nearest_neighbors(embeddings, k):
    if not embeddings:
        return None
    neigh = NearestNeighbors(n_neighbors=min(k, len(embeddings)))
    neigh.fit(embeddings)
    return neigh


def get_nearst_embedding(neigh, q_embeddings):
    if neigh is None:
        return np.asarray([[]], dtype=int)
    return np.asarray(neigh.kneighbors(q_embeddings, return_distance=False))


class EmbSearch:
    def __init__(self, db, k, start_embs=None):
        self.db = db
        self.emb = []

        if start_embs is None:
            with EMBEDDINGS_FILE.open("r", encoding="utf-8") as file:
                for _ in self.db:
                    line = file.readline().strip()
                    self.emb.append(list(map(float, line.split())) if line else [])
        else:
            for item in self.db:
                self.emb.append(start_embs.emb[item.id])

        self.neigh = build_nearest_neighbors(self.emb, k)

    def search(self, query):
        if not self.db:
            return []
        q_emb = get_emb([query])
        idx = get_nearst_embedding(self.neigh, q_emb)
        if idx.size == 0:
            return []
        return [self.db[i] for i in idx[0]]


def load_database():
    with DATABASE_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    objects = []
    for index, item in enumerate(data["features"]):
        x = item["geometry"]["coordinates"][0]
        y = item["geometry"]["coordinates"][1]
        props = item.get("properties", {})
        metadata = build_point_metadata(item)
        desc = build_point_description(props, item.get("info", {}), metadata)

        objects.append(
            Object(
                x,
                y,
                props.get("street", ""),
                props.get("name"),
                props.get("amenity"),
                desc,
                index,
                metadata,
            )
        )

    return objects


def dot_product(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2


def distance_to_line(a, b, p):
    x1 = a.x
    y1 = a.y
    x2 = b.x
    y2 = b.y
    x3 = p.x
    y3 = p.y
    numerator = abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
    denominator = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if denominator == 0:
        return sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    return numerator / denominator


def is_proj_in_segment(a, b, p):
    d1 = dot_product(p.x - a.x, p.y - a.y, b.x - a.x, b.y - a.y)
    d2 = dot_product(p.x - b.x, p.y - b.y, a.x - b.x, a.y - b.y)
    return d1 >= 0 and d2 >= 0


def get_points_into_route(objects_list, p_start, p_end, max_points=5, max_distance=1000):
    if not objects_list:
        return []

    work_objects = [
        Object(obj.x, obj.y, obj.street, obj.name, obj.amenity, obj.desc, obj.id, obj.other)
        for obj in objects_list
    ]
    work_p_start = Object(
        p_start.x,
        p_start.y,
        p_start.street,
        p_start.name,
        p_start.amenity,
        p_start.desc,
        p_start.id,
        p_start.other,
    )
    work_p_end = Object(
        p_end.x,
        p_end.y,
        p_end.street,
        p_end.name,
        p_end.amenity,
        p_end.desc,
        p_end.id,
        p_end.other,
    )

    mean_lat = sum(obj.y for obj in work_objects) / len(work_objects)
    lon_to_km = 111.32 * cos(radians(mean_lat))

    for obj in work_objects:
        obj.x = lon_to_km * obj.x
        obj.y = 111.32 * obj.y
    work_p_start.x = lon_to_km * work_p_start.x
    work_p_start.y = 111.32 * work_p_start.y
    work_p_end.x = lon_to_km * work_p_end.x
    work_p_end.y = 111.32 * work_p_end.y

    def comparator(a, b):
        dist_a = distance_to_line(work_p_start, work_p_end, a)
        dist_b = distance_to_line(work_p_start, work_p_end, b)
        if dist_a < dist_b:
            return -1
        if dist_a > dist_b:
            return 1
        return 0

    sorted_objects = sorted(work_objects, key=cmp_to_key(comparator))
    return sorted_objects[:max_points]


def get_points(db, qemb_s, p_start, p_end, query):
    corridor_candidates = get_points_into_route(db, p_start, p_end, 50)
    corridor_embs = EmbSearch(corridor_candidates, min(12, len(corridor_candidates)), qemb_s)
    near_path = corridor_embs.search(query)

    global_hits = qemb_s.search(query)
    merged = list(dict.fromkeys(near_path + global_hits))
    dists_to_start = [item.dist_between_points(p_start) for item in merged]
    dists_to_end = [item.dist_between_points(p_end) for item in merged]
    return merged, dists_to_start, dists_to_end

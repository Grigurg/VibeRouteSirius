import osmnx as ox
import json
import pandas as pd
from shapely.geometry import mapping

center_point = (43.405, 39.955)
radius = 3 * 10 ** 3
MIN_SCORE = 6

EXCLUDED_AMENITIES = {
    "fuel",
    "parking",
    "parking_entrance",
    "parking_space",
    "bank",
    "atm",
    "pharmacy",
    "hospital",
    "clinic",
    "dentist",
    "doctors",
    "school",
    "college",
    "university",
    "police",
    "fire_station",
    "post_office",
    "courthouse",
}

tags = {
    "amenity": True,
    "leisure": True,
    "tourism": True,
    "sport": True,
    "historic": True,
}

ENTERTAINMENT_KEYWORDS = [
    "kart", "quest", "escape", "game",
    "drive", "race", "arena", "park",
    "fun", "entertain"
]

TEXT_KEYS = [
    "name", "description", "brand",
    "amenity", "leisure", "tourism", "sport"
]


def text_contains_entertainment(row):
    parts = []
    for k in TEXT_KEYS:
        if k in row and isinstance(row[k], str):
            parts.append(row[k].lower())

    text = " ".join(parts)
    return any(k in text for k in ENTERTAINMENT_KEYWORDS)


def popularity_score(row):
    score = 0

    def has(key):
        return key in row and pd.notna(row[key])

    if has("amenity") and row["amenity"] in EXCLUDED_AMENITIES:
        return 0

    if not has("name"):
        return 0

    if has("wikidata") or has("wikipedia"):
        score += 4

    if has("description"):
        score += 1

    if has("website") or has("contact:website"):
        score += 3
    if has("phone") or has("contact:phone"):
        score += 2

    # ЕДА
    if has("amenity"):
        score += 1
    if has("cuisine"):
        score += 1
    if has("brand"):
        score += 2

    # ПАРКИ / ТУРИЗМ
    if has("leisure"):
        score += 2
    if has("tourism"):
        score += 2
    if has("historic"):
        score += 2

    # ЧАСЫ РАБОТЫ
    if has("opening_hours"):
        score += 1

    # Entertainments
    if text_contains_entertainment(row):
        score += 3

    return score


gdf = ox.features_from_point(center_point, tags=tags, dist=radius)
print(len(gdf))

gdf = gdf.dropna(axis=1, how="all")

features = []

for _, row in gdf.iterrows():
    score = popularity_score(row)
    if score < MIN_SCORE:
        continue

    props = {
        k: v
        for k, v in row.items()
        if k != "geometry" and pd.notna(v)
    }
    props["popularity_score"] = score

    point = row.geometry.centroid  # One Point

    feature = {
        "type": "Feature",
        "properties": props,
        "geometry": mapping(point),
    }

    features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features,
}

with open("sirius_poi_clean2.geojson", "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False, indent=2)

print(f"Сохранено объектов: {len(features)}")

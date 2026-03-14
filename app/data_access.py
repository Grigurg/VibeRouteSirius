import json
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

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("BAAI/bge-m3")
    return _embedding_model


def get_emb(sentences):
    return get_embedding_model().encode(sentences, batch_size=1)


def get_simularity(embeddings, q_embeddings):
    return get_embedding_model().similarity(embeddings, q_embeddings)


def build_nearest_neighbors(embeddings, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(embeddings)
    return neigh


def get_nearst_embedding(neigh, q_embeddings):
    return np.asarray(neigh.kneighbors(q_embeddings, return_distance=False))


class EmbSearch:
    def __init__(self, db, k, start_embs=None):
        self.db = db
        self.emb = []

        if start_embs is None:
            with EMBEDDINGS_FILE.open("r", encoding="utf-8") as file:
                for _ in self.db:
                    self.emb.append(list(map(float, file.readline().split())))
        else:
            for item in self.db:
                self.emb.append(start_embs.emb[item.id])

        self.neigh = build_nearest_neighbors(self.emb, k)

    def search(self, query):
        q_emb = get_emb([query])
        idx = get_nearst_embedding(self.neigh, q_emb)
        return [self.db[i] for i in idx[0]]


def load_database():
    with DATABASE_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    objects = []
    for index, item in enumerate(data["features"]):
        x = item["geometry"]["coordinates"][0]
        y = item["geometry"]["coordinates"][1]
        street = item["properties"]["street"]
        desc = f"Название: {item['properties']['name']}. "
        name = f"{item['properties']['name']}"
        amenity = item["properties"].get("amenity")

        if "amenity" in item["properties"]:
            desc += f"Тип: {item['properties']['amenity']}. "
        if "leisure" in item["properties"]:
            desc += f"Тип: {item['properties']['leisure']}. "
        if "popularity_score" in item["properties"]:
            desc += f"Уровень популярности: {item['properties']['popularity_score']}. "
        if "website" in item["properties"]:
            desc += f"Вебсайт: {item['info']['site']}. "

        if "info" in item:
            if "rating" in item["info"]:
                desc += f"Оценка по пятибалльной шкале: {item['info']['rating']}. "
            if "description" in item["info"] and item["info"]["description"] is not None:
                desc += f"Короткое описание: {item['info']['rating']}. "

            if len(item["info"]["feat_text"]) >= 1:
                desc += "Прочая информация: "
                for text in item["info"]["feat_text"]:
                    desc += text + ", "

        desc = desc.removesuffix(", ") + ". "
        objects.append(Object(x, y, street, name, amenity, desc, index, {}))

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
    work_objects = [
        Object(obj.x, obj.y, obj.street, obj.name, obj.amenity, obj.desc, obj.id, obj.other)
        for obj in objects_list
    ]
    work_p_start = Object(p_start.x, p_start.y, p_start.street, p_start.name, p_start.amenity, p_start.desc, p_start.id, p_start.other)
    work_p_end = Object(p_end.x, p_end.y, p_end.street, p_end.name, p_end.amenity, p_end.desc, p_end.id, p_end.other)

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
    ans = []
    for obj in sorted_objects:
        if len(ans) >= max_points:
            break
        ans.append(obj)

    return ans


def get_points(db, qemb_s, p_start, p_end, query):
    resd = get_points_into_route(db, p_start, p_end, 50)
    resd_embs = EmbSearch(resd, 6, qemb_s)
    resd_final = resd_embs.search(query)

    resq = qemb_s.search(query)
    for item in resq:
        resd_final.append(item)

    ans = list(dict.fromkeys(resd_final))
    dists_to_start = [item.dist_between_points(p_start) for item in ans]
    dists_to_end = [item.dist_between_points(p_end) for item in ans]

    return ans, dists_to_start, dists_to_end

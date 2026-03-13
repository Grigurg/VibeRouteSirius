import json
import random

def load_geojson(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    data = load_geojson("sirius_poi_all_info_clear_desc.geojson")
    features = data.get("features", [])
    categories = set()
    for feature in features:
        category = feature.get("properties", {}).get("amenity")
        if category and category != "Неизвестно":
            categories.add(category)
    
    categories_list = list(categories)
    
    if len(categories_list) < 2:
        print(f"Найдено только {len(categories_list)} категорий, нужно минимум 2")
        return
    category1, category2 = random.sample(categories_list, 2)
    message = f"Я хочу посетить {category1} и {category2}"
    print(message)
for i in range(30):
    main()
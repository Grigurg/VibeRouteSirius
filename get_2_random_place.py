import json
import random

def load_geojson(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def call_external_prompt_function(point1_info, point2_info):
    #вызов модели
    ""
    message = f"Отправка в модель: мне нужно пройтись по {point1_info["name"], point1_info["coordinates"]} а затем {point2_info["name"], point2_info["coordinates"]}"
    print(message)
    return "Результат от модели"

def main():
    data = load_geojson("sirius_poi_all_info_clear_desc.geojson")
    features = data.get("features", [])
    
    point1, point2 = random.sample(features, 2)
    
    point1_info = {
        "name": point1.get("properties", {}).get("name", "Без имени"),
        "type": point1.get("properties", {}).get("amenity", "Неизвестно"),
        "coordinates": point1.get("geometry", {}).get("coordinates", [0, 0])
    }
    
    point2_info = {
        "name": point2.get("properties", {}).get("name", "Без имени"),
        "type": point2.get("properties", {}).get("amenity", "Неизвестно"),
        "coordinates": point2.get("geometry", {}).get("coordinates", [0, 0])
    }
    
    result = call_external_prompt_function(point1_info, point2_info)
    print(f"Результат: {result}")
for i in range(100):
    main()
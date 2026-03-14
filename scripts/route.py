import folium
import json
import requests
import random

file = 'sirius_poi_clean2.geojson'

# Центр карты
m = folium.Map(
    location=[43.405, 39.955],
    zoom_start=12,
)


def get_route(p1, p2):
    url = (
        f"https://router.project-osrm.org/route/v1/foot/"
        f"{p1[0]},{p1[1]};{p2[0]},{p2[1]}"
        "?overview=full&geometries=geojson"
    )
    r = requests.get(url).json()
    return r["routes"][0]["geometry"]["coordinates"]


def draw_route(p1, p2):
    print("Draw: ", p1, p2)
    route_coords = get_route(p1, p2)
    route_latlngs = [[c[1], c[0]] for c in route_coords]

    folium.PolyLine(
        locations=route_latlngs,
        color='blue',
        weight=5,
        opacity=0.7,
        popup='Маршрут пешком'
    ).add_to(m)


def draw_random_route(data):
    id1 = random.randint(0, len(data))
    id2 = random.randint(0, len(data))
    p1 = data["features"][id1]["geometry"]["coordinates"]
    p2 = data["features"][id2]["geometry"]["coordinates"]
    draw_route(p1, p2)


def load_data():
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_points(data):
    for feature in data["features"]:
        lon, lat = feature["geometry"]["coordinates"]
        props = feature["properties"]

        popup = f"""
            <b>{props.get("name", "Без названия")}</b><br>
            Score: {props.get("popularity_score")}<br>
            {props.get("amenity", "")} {props.get("leisure", "")}
            """

        folium.Marker(
            location=[lat, lon],
            popup=popup,
            icon=folium.Icon(icon="info-sign"),
        ).add_to(m)


def main():
    data = load_data()
    load_points(data)
    draw_random_route(data)

    m.save("map.html")


if __name__ == '__main__':
    main()

import json
from shapely.geometry import mapping
import pandas as pd

def dump(obj, output_file="output.geojson"):
    if isinstance(obj, pd.DataFrame) and "geometry" in obj.columns:
        features = []
        for _, row in obj.iterrows():
            props = {k: v for k, v in row.items() if k != "geometry" and pd.notna(v)}
            point = row.geometry.centroid
            feature = {
                "type": "Feature",
                "properties": props,
                "geometry": mapping(point),
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        print(f"Сохранено объектов: {len(features)}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"Сохранено объектов (dict/list): {len(obj) if hasattr(obj, '__len__') else 1}")


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

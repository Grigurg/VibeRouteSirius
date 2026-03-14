import requests


def forward_geocode(query):
    query = (query or "").strip()
    if not query:
        return None

    try:
        yandex_resp = requests.get(
            "https://geocode-maps.yandex.ru/1.x/",
            params={"format": "json", "geocode": query, "results": 1},
            timeout=3,
        )
        yandex_resp.raise_for_status()
        yandex_data = yandex_resp.json()
        features = yandex_data.get("response", {}).get("GeoObjectCollection", {}).get("featureMember", [])
        if features:
            geo_object = features[0].get("GeoObject", {})
            pos = geo_object.get("Point", {}).get("pos", "")
            if pos:
                lng_str, lat_str = pos.split()[:2]
                return {"lat": float(lat_str), "lng": float(lng_str)}
    except Exception:
        pass

    try:
        nominatim_resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"format": "jsonv2", "q": query, "limit": 1, "accept-language": "ru"},
            headers={"User-Agent": "viberoute-demo/1.0"},
            timeout=5,
        )
        nominatim_resp.raise_for_status()
        data = nominatim_resp.json() or []
        if data:
            return {"lat": float(data[0]["lat"]), "lng": float(data[0]["lon"])}
    except Exception:
        pass

    return None


def reverse_geocode_coords(lat, lng):
    if not lat or not lng:
        return ""

    address = ""
    try:
        yandex_resp = requests.get(
            "https://geocode-maps.yandex.ru/1.x/",
            params={"format": "json", "geocode": f"{lng},{lat}", "kind": "house", "results": 1},
            timeout=3,
        )
        yandex_resp.raise_for_status()
        yandex_data = yandex_resp.json()
        features = yandex_data.get("response", {}).get("GeoObjectCollection", {}).get("featureMember", [])
        if features:
            geo_object = features[0].get("GeoObject", {})
            address = geo_object.get("metaDataProperty", {}).get("GeocoderMetaData", {}).get("text", "")
    except Exception:
        pass

    if not address:
        try:
            nominatim_resp = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={"format": "jsonv2", "lat": lat, "lon": lng, "accept-language": "ru"},
                headers={"User-Agent": "viberoute-demo/1.0"},
                timeout=5,
            )
            nominatim_resp.raise_for_status()
            nominatim_data = nominatim_resp.json()
            addr_parts = nominatim_data.get("address", {})
            if addr_parts:
                street = addr_parts.get("road") or addr_parts.get("pedestrian") or ""
                house = addr_parts.get("house_number") or ""
                city = addr_parts.get("city") or addr_parts.get("town") or addr_parts.get("village") or ""
                if street:
                    address = f"{street}, {house}" if house else street
                    if city:
                        address = f"{address}, {city}"
                else:
                    address = nominatim_data.get("display_name", "")
            else:
                address = nominatim_data.get("display_name", "")
        except Exception:
            pass

    return address or f"{lat}, {lng}"

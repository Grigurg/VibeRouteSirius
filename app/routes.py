from flask import jsonify, render_template, request, send_from_directory

from app.geocoding import forward_geocode, reverse_geocode_coords
from app.planner import (
    MODEL_OPTIONS,
    PLACES,
    build_summary,
    get_ui_text,
    get_vibes,
    get_end_location_coords,
    get_places,
    get_start_location_coords,
    normalize_language,
    parse_form,
    LANGUAGES,
)
from app.llm import DEFAULT_MODEL


def init_app(app):
    @app.route("/", methods=["GET", "POST"])
    def index():
        loading = False
        formdata = {}
        result_data = None
        generated = False

        if request.method == "POST":
            loading = True
            formdata = parse_form(request.form)
            lang = normalize_language(formdata.get("lang", "ru"))
            start_coords = get_start_location_coords(request.form)
            end_coords = get_end_location_coords(request.form)

            if not start_coords and formdata.get("start_addr"):
                start_coords = forward_geocode(formdata.get("start_addr"), lang)
            if not end_coords and formdata.get("end_addr"):
                end_coords = forward_geocode(formdata.get("end_addr"), lang)

            if start_coords:
                formdata["start_lat"] = str(start_coords["lat"])
                formdata["start_lng"] = str(start_coords["lng"])
            if end_coords:
                formdata["end_lat"] = str(end_coords["lat"])
                formdata["end_lng"] = str(end_coords["lng"])

            result_data = build_summary(formdata)
            generated = True
            loading = False
        else:
            for key in [
                "lang",
                "start_addr",
                "end_addr",
                "start_lat",
                "start_lng",
                "end_lat",
                "end_lng",
                "duration_hrs",
                "duration_mins",
                "budget",
                "vibe",
                "extra_notes",
                "model",
                "map_lat",
                "map_lng",
                "map_zoom",
            ]:
                if key in request.args:
                    formdata[key] = request.args[key]

            formdata.setdefault("lang", "ru")
            formdata.setdefault("start_addr", "Сириус Арена, Сириус")
            formdata.setdefault("end_addr", "Сочи Парк, Сириус")
            formdata.setdefault("start_lat", "43.40881998152516")
            formdata.setdefault("start_lng", "39.952640447932154")
            formdata.setdefault("end_lat", "43.4047")
            formdata.setdefault("end_lng", "39.9670")
            formdata.setdefault("model", DEFAULT_MODEL)
            formdata.setdefault("map_lat", "43.4085")
            formdata.setdefault("map_lng", "39.9625")
            formdata.setdefault("map_zoom", "14")

        lang = normalize_language(formdata.get("lang", "ru"))
        formdata["lang"] = lang

        if result_data is not None:
            try:
                places, description = get_places(formdata)
                result_data["route_description"] = description if description != "no-description" else None
            except Exception as exc:
                places = PLACES
                result_data["route_description"] = None
                result_data["error"] = str(exc)
        else:
            places = PLACES

        return render_template(
            "index.html",
            formdata=formdata,
            ui=get_ui_text(lang),
            languages=LANGUAGES,
            vibes=get_vibes(lang),
            model_options=MODEL_OPTIONS,
            result_data=result_data,
            generated=generated,
            loading=loading,
            places=places,
        )

    @app.route("/reverse_geocode")
    def reverse_geocode():
        lat = request.args.get("lat", "")
        lng = request.args.get("lng", "")
        lang = normalize_language(request.args.get("lang", "ru"))
        return jsonify({"address": reverse_geocode_coords(lat, lng, lang)})

    @app.route("/geocode")
    def geocode():
        query = (request.args.get("query", "") or "").strip()
        lang = normalize_language(request.args.get("lang", "ru"))
        if not query:
            return jsonify({"lat": None, "lng": None})
        coords = forward_geocode(query, lang)
        if coords:
            return jsonify(coords)
        return jsonify({"lat": None, "lng": None})

    @app.route("/style.css")
    def style_css():
        return send_from_directory(app.static_folder, "style.css", mimetype="text/css")

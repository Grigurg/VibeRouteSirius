import json
import classes


def get_database():
    with open('sirius_super_cool.geojson', 'r', encoding='utf-8') as f:
        data = json.load(f)

    objects = []

    cnt = 0

    for i in data['features']:
        x = i['geometry']['coordinates'][0]
        y = i['geometry']['coordinates'][1]
        id = cnt
        street = i['properties']['street']
        desc = f"Название: {i['properties']['name']}. "
        name = f"{i['properties']['name']}"
        amenity = None
        if 'amenity' in i['properties']:
            amenity = i['properties']['amenity']
        if 'amenity' in i['properties']:
            desc += f"Тип: {i['properties']['amenity']}. "
        if 'leisure' in i['properties']:
            desc += f"Тип: {i['properties']['leisure']}. "
        if 'popularity_score' in i['properties']:
            desc += f"Уровень популярности: {i['properties']['popularity_score']}. "
        if 'website' in i['properties']:
            desc += f"Вебсайт: {i['info']['site']}. "

        if 'info' in i:
            if 'rating' in i['info']:
                desc += f"Оценка по пятибалльной шкале: {i['info']['rating']}. "
            if 'description' in i['info'] and i['info']['description'] != None:
                desc += f"Короткое описание: {i['info']['rating']}. "

            if len(i['info']['feat_text']) >= 1:
                desc += 'Прочая информация: '
                for j in i['info']['feat_text']:
                    desc += j + ', '

        desc = desc.removesuffix(', ') + '. '
        objects.append(classes.Object(x, y, street, name, amenity, desc, id, {}))
        cnt += 1
    return objects

import math


class Object:
    def __init__(
        self,
        x,
        y,
        street,
        name=None,
        amenity=None,
        desc=None,
        id=None,
        other_params=None,
    ):
        self.x = x
        self.y = y
        self.id = id
        self.desc = desc
        self.other = other_params or {}
        self.street = street
        self.name = name
        self.amenity = amenity

    @property
    def categories(self):
        return tuple(self.other.get("categories", ()))

    @property
    def primary_category(self):
        return self.other.get("primary_category")

    @property
    def rating_value(self):
        return self.other.get("rating_value")

    @property
    def review_count(self):
        return self.other.get("review_count", 0)

    @property
    def popularity_score(self):
        return self.other.get("popularity_score", 0.0)

    @property
    def is_good_place(self):
        return bool(self.other.get("is_good_place"))

    @property
    def estimated_cost_rub(self):
        return self.other.get("estimated_cost_rub")

    @property
    def text_blob(self):
        return self.other.get("text_blob", self.desc or "")

    def display_name(self):
        return self.name or self.street or f"Point {self.id}"

    def matches_category(self, category):
        normalized = (category or "").strip().lower()
        if not normalized:
            return False
        primary = (self.primary_category or "").strip().lower()
        categories = {item.strip().lower() for item in self.categories}
        return normalized == primary or normalized in categories

    def dist_between_points(self, other):
        radius = 6371000
        lon1, lat1, lon2, lat2 = map(math.radians, [self.x, self.y, other.x, other.y])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        a = max(0, min(1, a))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return round(radius * c)

    def __eq__(self, other):
        if not isinstance(other, Object):
            return False
        if self.id is not None and other.id is not None:
            return self.id == other.id
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        if self.id is not None:
            return hash(self.id)
        return hash((self.x, self.y, self.street))

    def __repr__(self):
        return f"{self.x=}; {self.y=}; {self.id=}, {self.desc=}; {self.other=}; {self.street=}"

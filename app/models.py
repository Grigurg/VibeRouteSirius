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
        self.other = other_params
        self.street = street
        self.name = name
        self.amenity = amenity

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
        return isinstance(other, Object) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(self.desc)

    def __repr__(self):
        return f"{self.x=}; {self.y=}; {self.id=}, {self.desc=}; {self.other=}; {self.street=}"

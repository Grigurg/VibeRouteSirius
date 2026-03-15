from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from app.data_access import (
    FOOD_CATEGORIES,
    EmbSearch,
    canonicalize_category_name,
    get_emb,
    get_points_into_route,
    normalize_text,
)
from app.models import Object

PACE_PROFILES = {
    "relaxed": {
        "label": "relaxed",
        "speed_m_per_min": 40,
        "stop_share": 0.30,
    },
    "normal": {
        "label": "normal",
        "speed_m_per_min": 48,
        "stop_share": 0.22,
    },
    "active": {
        "label": "active",
        "speed_m_per_min": 58,
        "stop_share": 0.15,
    },
}

VIBE_QUERY_HINTS = {
    "friendly": "promenade, park, scenic place, cafe, public space",
    "romantic": "quiet promenade, scenic viewpoint, beach, park, cafe",
    "family": "family-friendly park, playground, promenade, cafe",
    "active": "promenade, sports area, park, attraction",
    "cozy": "quiet cafe, calm park, promenade, scenic place",
    "cultural": "museum, artwork, landmark, promenade, attraction",
}

VIBE_PREFERRED_CATEGORIES = {
    "friendly": ["promenade", "park", "cafe", "attraction"],
    "romantic": ["promenade", "beach", "viewpoint", "park", "cafe"],
    "family": ["park", "playground", "promenade", "cafe", "attraction"],
    "active": ["sports", "promenade", "park", "attraction"],
    "cozy": ["cafe", "park", "promenade", "viewpoint"],
    "cultural": ["museum", "artwork", "attraction", "promenade"],
}

VIBE_KEYWORDS = {
    "friendly": ["friendly", "social", "easygoing", "друж", "общ", "компан"],
    "romantic": ["romantic", "романт", "уют", "закат", "вид", "quiet"],
    "family": ["family", "children", "дет", "сем"],
    "active": ["active", "sport", "энерг", "динамич"],
    "cozy": ["cozy", "quiet", "calm", "спокой", "уют"],
    "cultural": ["cultural", "museum", "history", "культур", "музей"],
}

STOP_MINUTES_BY_CATEGORY = {
    "cafe": 30,
    "restaurant": 38,
    "bar": 32,
    "nightlife": 40,
    "park": 18,
    "promenade": 14,
    "beach": 20,
    "viewpoint": 12,
    "museum": 28,
    "artwork": 12,
    "attraction": 22,
    "sports": 24,
    "playground": 18,
    "entertainment": 24,
    "religious_site": 15,
    "generic": 12,
}

POINT_SCORE_WEIGHTS = {
    "relevance_score": 4.6,
    "path_score": 3.0,
    "diversity_score": 0.8,
    "quality_score": 2.5,
    "order_fit_score": 1.6,
    "constraint_fit_score": 1.6,
    "detour_penalty": 3.4,
    "budget_penalty": 2.7,
}

ROUTE_SCORE_WEIGHTS = {
    "diversity_bonus": 3.2,
    "coherence_bonus": 2.8,
    "constraint_completion_bonus": 3.4,
    "total_detour_penalty": 4.2,
    "overbudget_penalty": 5.2,
    "category_repeat_penalty": 2.5,
    "underfilled_route_penalty": 3.4,
    "overfilled_route_penalty": 4.4,
}

DEFAULT_RETRIEVAL_LIMIT = 48
DEFAULT_NARROWED_LIMIT = 12
DEFAULT_BEAM_WIDTH = 28
DEFAULT_COMPLETED_LIMIT = 80


@dataclass
class PlanningRequest:
    start: Object
    end: Object
    lang: str
    model_name: str
    total_minutes: int
    pace: str
    vibe: str
    budget_rub: int
    extra_notes: str
    hard_constraints: dict
    soft_preferences: dict
    notes_summary: str
    speed_m_per_min: int
    walking_minutes: int
    distance_budget_m: int
    direct_distance_m: int


@dataclass
class ScoredCandidate:
    point: Object
    total_score: float
    components: dict
    progress_ratio: float
    distance_to_path_m: float
    insertion_detour_m: int
    estimated_stop_minutes: int
    reasons: list[str] = field(default_factory=list)


@dataclass
class RouteState:
    point_ids: tuple[int, ...]
    last_point: Object
    walked_distance_m: int
    stop_minutes: int
    cost_rub: int
    food_count: int
    category_counts: dict[str, int]
    seq_progress: int
    required_satisfied: frozenset[str]
    point_score_sum: float
    transition_penalty_sum: float
    beam_score: float
    last_progress_ratio: float
    total_path_deviation_m: float


@dataclass
class RoutePlan:
    stop_points: list[Object]
    total_distance_m: int
    estimated_duration_min: int
    estimated_cost_rub: int
    route_score: float
    route_reasons: list[str]
    stop_reasons: dict[int, list[str]]
    candidate_scores: dict[int, ScoredCandidate]
    debug: dict

    @property
    def stop_ids(self):
        return [point.id for point in self.stop_points]


def clamp(value, lower=0.0, upper=1.0):
    return max(lower, min(upper, value))


def dedupe_preserve_order(values):
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def cosine_similarity(vector_a, vector_b):
    if vector_a is None or vector_b is None:
        return 0.0
    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def project_to_local(point: Object, reference_lat: float):
    scale_x = 111_320 * math.cos(math.radians(reference_lat))
    scale_y = 111_320
    return point.x * scale_x, point.y * scale_y


def segment_metrics(start: Object, end: Object, point: Object):
    reference_lat = (start.y + end.y + point.y) / 3
    ax, ay = project_to_local(start, reference_lat)
    bx, by = project_to_local(end, reference_lat)
    px, py = project_to_local(point, reference_lat)
    abx = bx - ax
    aby = by - ay
    ab_len_sq = abx ** 2 + aby ** 2
    if ab_len_sq == 0:
        return math.dist((ax, ay), (px, py)), 0.0, 0.0
    t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
    progress = clamp(t, 0.0, 1.0)
    proj_x = ax + progress * abx
    proj_y = ay + progress * aby
    distance = math.dist((px, py), (proj_x, proj_y))
    segment_length = math.sqrt(ab_len_sq)
    return distance, progress, segment_length


def normalize_constraint_term(value):
    normalized = normalize_text(value)
    if not normalized:
        return ""
    canonical = canonicalize_category_name(normalized)
    return canonical or normalized


def term_matches_point(term, point: Object):
    normalized_term = normalize_constraint_term(term)
    if not normalized_term:
        return False
    if point.matches_category(normalized_term):
        return True
    text_blob = normalize_text(point.text_blob)
    name = normalize_text(point.name)
    street = normalize_text(point.street)
    return (
        normalized_term in text_blob
        or normalized_term in name
        or normalized_term in street
    )


def estimate_stop_minutes(point: Object):
    for category in point.categories:
        if category in STOP_MINUTES_BY_CATEGORY:
            return STOP_MINUTES_BY_CATEGORY[category]
    return STOP_MINUTES_BY_CATEGORY["generic"]


class DeterministicRoutePlanner:
    def __init__(self, db):
        self.db = db
        self.embedding_index = EmbSearch(db, min(DEFAULT_RETRIEVAL_LIMIT, len(db)))
        self.embedding_matrix = np.asarray(self.embedding_index.emb, dtype=float)
        self.embedding_search_available = True
        self.embedding_warning_logged = False

    def get_pace_profile(self, pace):
        return PACE_PROFILES.get(pace, PACE_PROFILES["relaxed"])

    def build_distance_budget(self, start_point, end_point, total_minutes, pace):
        profile = self.get_pace_profile(pace)
        walking_minutes = max(total_minutes * (1 - profile["stop_share"]), 25)
        computed_budget = int(round(walking_minutes * profile["speed_m_per_min"]))
        direct_distance = start_point.dist_between_points(end_point)
        min_required_budget = int(direct_distance * 1.12)
        return {
            "pace_label": profile["label"],
            "speed_m_per_min": profile["speed_m_per_min"],
            "walking_minutes": int(round(walking_minutes)),
            "distance_budget_m": max(computed_budget, min_required_budget),
            "direct_distance_m": direct_distance,
        }

    def build_request(
        self,
        start_point,
        end_point,
        vibe,
        total_minutes,
        pace,
        budget_rub,
        extra_notes,
        lang,
        model_name,
        parsed_request,
    ):
        budget = self.build_distance_budget(start_point, end_point, total_minutes, pace)
        hard_constraints = dict(parsed_request.get("hard_constraints", {}))
        soft_preferences = dict(parsed_request.get("soft_preferences", {}))

        soft_vibes = dedupe_preserve_order(
            [
                normalize_constraint_term(item)
                for item in [vibe] + list(soft_preferences.get("vibe", []))
                if normalize_constraint_term(item)
            ]
        )
        soft_preferences["vibe"] = soft_vibes
        soft_preferences["preferred_categories"] = dedupe_preserve_order(
            [
                normalize_constraint_term(item)
                for item in list(VIBE_PREFERRED_CATEGORIES.get(vibe, []))
                + list(soft_preferences.get("preferred_categories", []))
                if normalize_constraint_term(item)
            ]
        )
        soft_preferences["optional_preferences"] = dedupe_preserve_order(
            [
                normalize_constraint_term(item)
                for item in soft_preferences.get("optional_preferences", [])
                if normalize_constraint_term(item)
            ]
        )

        hard_constraints["must_visit_in_order"] = [
            normalize_constraint_term(item)
            for item in hard_constraints.get("must_visit_in_order", [])
            if normalize_constraint_term(item)
        ]
        hard_constraints["avoid_categories"] = dedupe_preserve_order(
            normalize_constraint_term(item)
            for item in hard_constraints.get("avoid_categories", [])
            if normalize_constraint_term(item)
        )
        hard_constraints["required_categories"] = dedupe_preserve_order(
            normalize_constraint_term(item)
            for item in hard_constraints.get("required_categories", [])
            if normalize_constraint_term(item)
        )

        inferred_max_stops = max(1, min(5, round(total_minutes / 35))) if total_minutes > 0 else 1
        hard_constraints["max_intermediate_stops"] = max(
            0,
            min(
                5,
                int(
                    inferred_max_stops
                    if hard_constraints.get("max_intermediate_stops") is None
                    else hard_constraints.get("max_intermediate_stops")
                ),
            ),
        )
        hard_constraints["food_limit"] = int(
            1 if hard_constraints.get("food_limit") is None else hard_constraints.get("food_limit")
        )
        explicit_duration_limit = hard_constraints.get("max_total_duration_min")
        hard_constraints["max_total_duration_min"] = int(
            min(total_minutes, explicit_duration_limit)
            if explicit_duration_limit is not None
            else total_minutes
        )
        hard_constraints["max_total_budget_rub"] = (
            None
            if budget_rub >= 10_000
            else int(
                min(budget_rub, hard_constraints.get("max_total_budget_rub"))
                if hard_constraints.get("max_total_budget_rub") is not None
                else budget_rub
            )
        )
        explicit_distance_limit = hard_constraints.get("max_total_distance_m")
        hard_constraints["max_total_distance_m"] = int(
            min(budget["distance_budget_m"], explicit_distance_limit)
            if explicit_distance_limit is not None
            else budget["distance_budget_m"]
        )

        return PlanningRequest(
            start=start_point,
            end=end_point,
            lang=lang,
            model_name=model_name,
            total_minutes=total_minutes,
            pace=pace,
            vibe=vibe,
            budget_rub=budget_rub,
            extra_notes=extra_notes,
            hard_constraints=hard_constraints,
            soft_preferences=soft_preferences,
            notes_summary=parsed_request.get("notes_summary", ""),
            speed_m_per_min=budget["speed_m_per_min"],
            walking_minutes=budget["walking_minutes"],
            distance_budget_m=budget["distance_budget_m"],
            direct_distance_m=budget["direct_distance_m"],
        )

    def build_retrieval_queries(self, request: PlanningRequest):
        preference_terms = dedupe_preserve_order(
            list(request.soft_preferences.get("preferred_categories", []))
            + list(request.soft_preferences.get("optional_preferences", []))
            + list(request.hard_constraints.get("required_categories", []))
            + list(request.hard_constraints.get("must_visit_in_order", []))
            + list(request.soft_preferences.get("vibe", []))
        )

        queries = []
        if request.extra_notes.strip():
            queries.append(request.extra_notes.strip())
        if request.notes_summary.strip():
            queries.append(request.notes_summary.strip())
        if preference_terms:
            queries.append(", ".join(preference_terms))
        queries.append(VIBE_QUERY_HINTS.get(request.vibe, VIBE_QUERY_HINTS["friendly"]))
        return dedupe_preserve_order(query for query in queries if query.strip())

    def log_embedding_fallback(self, exc):
        if not self.embedding_warning_logged:
            logger.warning("Embedding search is unavailable, falling back to lexical retrieval: {}", exc)
            self.embedding_warning_logged = True

    def lexical_search(self, pool, query, limit=24):
        tokens = [token for token in re.split(r"[^a-zA-Zа-яА-Я0-9_]+", normalize_text(query)) if len(token) >= 3]
        unique_tokens = dedupe_preserve_order(tokens)
        scored = []
        for point in pool:
            haystack = normalize_text(point.text_blob)
            score = 0.0
            for token in unique_tokens:
                if token in haystack:
                    score += 1.0
                if point.matches_category(token):
                    score += 1.5
            if point.is_good_place:
                score += 0.2
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    point.is_good_place,
                    point.popularity_score,
                    point.rating_value or 0.0,
                    point.review_count,
                    point,
                )
            )

        scored.sort(key=lambda item: item[:-1], reverse=True)
        return [item[-1] for item in scored[:limit]]

    def safe_search(self, search_index, query, fallback_pool, limit=24):
        if not self.embedding_search_available:
            return self.lexical_search(fallback_pool, query, limit)
        try:
            return search_index.search(query)[:limit]
        except Exception as exc:
            self.embedding_search_available = False
            self.log_embedding_fallback(exc)
            return self.lexical_search(fallback_pool, query, limit)

    def retrieve_candidates(self, request: PlanningRequest):
        queries = self.build_retrieval_queries(request)
        corridor = get_points_into_route(self.db, request.start, request.end, max_points=140)
        corridor_index = EmbSearch(corridor, min(24, len(corridor)), self.embedding_index)

        collected = []
        for query in queries:
            collected.extend(self.safe_search(corridor_index, query, corridor, 24))
            collected.extend(self.safe_search(self.embedding_index, query, self.db, 24))

        quality_fallback = sorted(
            corridor,
            key=lambda point: (
                point.is_good_place,
                point.popularity_score,
                point.rating_value or 0,
                point.review_count,
            ),
            reverse=True,
        )
        collected.extend(quality_fallback[:24])

        unique = []
        seen_ids = set()
        for point in collected:
            if point.id in seen_ids:
                continue
            if point.id is None:
                continue
            if point.id in (request.start.id, request.end.id):
                continue
            if set(point.categories).intersection(request.hard_constraints["avoid_categories"]):
                continue
            seen_ids.add(point.id)
            unique.append(self.db[point.id])
            if len(unique) >= DEFAULT_RETRIEVAL_LIMIT:
                break

        logger.info(
            "Retrieved {} candidates using {} queries",
            len(unique),
            len(queries),
        )
        return unique, queries

    def semantic_relevance(self, point, query_embeddings):
        if point.id is None or not len(query_embeddings):
            return 0.0
        point_embedding = self.embedding_matrix[point.id]
        return max(cosine_similarity(point_embedding, query) for query in query_embeddings)

    def category_match_score(self, point, request):
        score = 0.0
        preferred = request.soft_preferences.get("preferred_categories", [])
        required = request.hard_constraints.get("required_categories", [])
        sequence = request.hard_constraints.get("must_visit_in_order", [])

        if any(term_matches_point(term, point) for term in preferred):
            score += 0.55
        if any(term_matches_point(term, point) for term in required):
            score += 0.80
        if any(term_matches_point(term, point) for term in sequence):
            score += 0.55
        return clamp(score, 0.0, 1.0)

    def preference_keyword_score(self, point, request):
        keywords = dedupe_preserve_order(
            list(request.soft_preferences.get("optional_preferences", []))
            + list(VIBE_KEYWORDS.get(request.vibe, []))
            + list(request.soft_preferences.get("vibe", []))
        )
        if not keywords:
            return 0.0
        text_blob = normalize_text(point.text_blob)
        matched = sum(1 for keyword in keywords if normalize_text(keyword) and normalize_text(keyword) in text_blob)
        return clamp(matched / max(len(keywords), 1), 0.0, 1.0)

    def build_query_embeddings(self, queries):
        if not queries:
            return np.asarray([])
        if not self.embedding_search_available:
            return np.asarray([])
        try:
            return np.asarray(get_emb(queries), dtype=float)
        except Exception as exc:
            self.embedding_search_available = False
            self.log_embedding_fallback(exc)
            return np.asarray([])

    def quality_signal(self, point):
        rating_norm = clamp(((point.rating_value or 0.0) - 3.5) / 1.5)
        review_norm = clamp(math.log1p(point.review_count) / math.log1p(10_000))
        popularity_norm = clamp(point.popularity_score / 11.0)
        good_place_norm = 1.0 if point.is_good_place else 0.0
        return (
            0.45 * rating_norm
            + 0.25 * review_norm
            + 0.20 * popularity_norm
            + 0.10 * good_place_norm
        )

    def order_fit_score(self, point, request, progress_ratio):
        sequence = request.hard_constraints.get("must_visit_in_order", [])
        if not sequence:
            return 0.0

        best_score = 0.0
        slot_count = len(sequence)
        for index, term in enumerate(sequence):
            if not term_matches_point(term, point):
                continue
            expected_progress = (index + 1) / (slot_count + 1)
            fit = 1.0 - min(abs(progress_ratio - expected_progress) / 0.55, 1.0)
            best_score = max(best_score, fit)
        return clamp(best_score, 0.0, 1.0)

    def constraint_fit_score(self, point, request):
        score = 0.0
        if any(term_matches_point(term, point) for term in request.hard_constraints.get("required_categories", [])):
            score += 0.7
        if request.hard_constraints.get("food_limit", 1) > 0 and point.other.get("is_food_venue"):
            if any(term_matches_point(term, point) for term in request.hard_constraints.get("must_visit_in_order", [])):
                score += 0.25
        return clamp(score, 0.0, 1.0)

    def budget_penalty(self, point, request):
        max_budget = request.hard_constraints.get("max_total_budget_rub")
        if max_budget is None or not point.estimated_cost_rub:
            return 0.0
        threshold = max_budget * 0.55
        if point.estimated_cost_rub <= threshold:
            return 0.0
        return clamp((point.estimated_cost_rub - threshold) / max(max_budget, 1), 0.0, 1.2)

    def build_candidate_reasons(self, point, components, request):
        reasons = []
        if components["category_match"] >= 0.45:
            reasons.append("matches requested categories")
        if components["order_fit_score"] >= 0.45:
            reasons.append("fits the requested stop order")
        if components["quality_score"] >= 0.55:
            reasons.append("looks high-quality from ratings and popularity")
        if components["path_score"] >= 0.55:
            reasons.append("stays close to the natural path")
        if components["keyword_score"] >= 0.35:
            reasons.append("matches the requested atmosphere")
        if point.estimated_cost_rub and request.hard_constraints.get("max_total_budget_rub"):
            if self.budget_penalty(point, request) == 0:
                reasons.append("fits the budget")
        if not reasons:
            reasons.append("adds a relevant stop without a major detour")
        return reasons

    def score_candidates(self, candidates, request: PlanningRequest):
        queries = self.build_retrieval_queries(request)
        query_embeddings = self.build_query_embeddings(queries)
        corridor_width = max(250.0, request.direct_distance_m * 0.18)
        detour_allowance = max(350.0, request.distance_budget_m - request.direct_distance_m, request.direct_distance_m * 0.45)

        interim = []
        for point in candidates:
            distance_to_path_m, progress_ratio, _ = segment_metrics(request.start, request.end, point)
            insertion_detour_m = (
                request.start.dist_between_points(point)
                + point.dist_between_points(request.end)
                - request.direct_distance_m
            )

            relevance_similarity = clamp((self.semantic_relevance(point, query_embeddings) + 1.0) / 2.0)
            category_match = self.category_match_score(point, request)
            keyword_score = self.preference_keyword_score(point, request)
            relevance_score = clamp(
                0.60 * relevance_similarity
                + 0.25 * category_match
                + 0.15 * keyword_score
            )

            path_proximity = clamp(1.0 - (distance_to_path_m / corridor_width))
            forward_progress = clamp(progress_ratio, 0.0, 1.0)
            path_score = clamp(0.7 * path_proximity + 0.3 * forward_progress)
            quality_score = self.quality_signal(point)
            detour_penalty = clamp(insertion_detour_m / detour_allowance, 0.0, 1.3)
            budget_penalty = self.budget_penalty(point, request)
            order_fit_score = self.order_fit_score(point, request, progress_ratio)
            constraint_fit_score = self.constraint_fit_score(point, request)

            components = {
                "semantic_similarity": relevance_similarity,
                "category_match": category_match,
                "keyword_score": keyword_score,
                "relevance_score": relevance_score,
                "path_score": path_score,
                "quality_score": quality_score,
                "order_fit_score": order_fit_score,
                "constraint_fit_score": constraint_fit_score,
                "detour_penalty": detour_penalty,
                "budget_penalty": budget_penalty,
                "diversity_score": 0.0,
            }
            interim.append((point, components, progress_ratio, distance_to_path_m, insertion_detour_m))

        category_freq = Counter(item[0].primary_category or "generic" for item in interim)
        scored_candidates = []
        for point, components, progress_ratio, distance_to_path_m, insertion_detour_m in interim:
            category = point.primary_category or "generic"
            rarity = 1.0 - ((category_freq[category] - 1) / max(len(interim) - 1, 1))
            components["diversity_score"] = clamp(rarity, 0.0, 1.0)

            total_score = (
                POINT_SCORE_WEIGHTS["relevance_score"] * components["relevance_score"]
                + POINT_SCORE_WEIGHTS["path_score"] * components["path_score"]
                + POINT_SCORE_WEIGHTS["diversity_score"] * components["diversity_score"]
                + POINT_SCORE_WEIGHTS["quality_score"] * components["quality_score"]
                + POINT_SCORE_WEIGHTS["order_fit_score"] * components["order_fit_score"]
                + POINT_SCORE_WEIGHTS["constraint_fit_score"] * components["constraint_fit_score"]
                - POINT_SCORE_WEIGHTS["detour_penalty"] * components["detour_penalty"]
                - POINT_SCORE_WEIGHTS["budget_penalty"] * components["budget_penalty"]
            )

            scored_candidates.append(
                ScoredCandidate(
                    point=point,
                    total_score=total_score,
                    components=components,
                    progress_ratio=progress_ratio,
                    distance_to_path_m=distance_to_path_m,
                    insertion_detour_m=insertion_detour_m,
                    estimated_stop_minutes=estimate_stop_minutes(point),
                    reasons=self.build_candidate_reasons(point, components, request),
                )
            )

        scored_candidates.sort(key=lambda candidate: candidate.total_score, reverse=True)
        return scored_candidates

    def are_near_duplicates(self, left: ScoredCandidate, right: ScoredCandidate):
        if left.point.id == right.point.id:
            return True
        if left.point.other.get("normalized_name") and left.point.other.get("normalized_name") == right.point.other.get("normalized_name"):
            return True
        same_category = (left.point.primary_category or "generic") == (right.point.primary_category or "generic")
        return same_category and left.point.dist_between_points(right.point) <= 90

    def narrow_candidates(self, scored_candidates, request: PlanningRequest):
        selected = []
        category_counts = Counter()

        def try_add(candidate):
            if any(self.are_near_duplicates(candidate, existing) for existing in selected):
                return False
            category = candidate.point.primary_category or "generic"
            soft_cap = 3 if category in FOOD_CATEGORIES else 2
            if category_counts[category] >= soft_cap and len(selected) < DEFAULT_NARROWED_LIMIT - 2:
                return False
            selected.append(candidate)
            category_counts[category] += 1
            return True

        for term in request.hard_constraints.get("must_visit_in_order", []) + request.hard_constraints.get("required_categories", []):
            for candidate in scored_candidates:
                if term_matches_point(term, candidate.point):
                    if try_add(candidate):
                        break

        for candidate in scored_candidates:
            if len(selected) >= DEFAULT_NARROWED_LIMIT:
                break
            try_add(candidate)

        if len(selected) < DEFAULT_NARROWED_LIMIT:
            for candidate in scored_candidates:
                if candidate in selected:
                    continue
                selected.append(candidate)
                if len(selected) >= DEFAULT_NARROWED_LIMIT:
                    break

        selected.sort(key=lambda candidate: candidate.total_score, reverse=True)
        logger.info("Narrowed to {} candidates", len(selected))
        return selected

    def point_matches_future_sequence_slot(self, point, sequence, seq_progress):
        for index in range(seq_progress + 1, len(sequence)):
            if term_matches_point(sequence[index], point):
                return True
        return False

    def transition_penalty(self, state, candidate, request):
        incremental_detour = (
            state.last_point.dist_between_points(candidate.point)
            + candidate.point.dist_between_points(request.end)
            - state.last_point.dist_between_points(request.end)
        )
        backward_penalty = 0.0
        if candidate.progress_ratio + 0.035 < state.last_progress_ratio:
            backward_penalty = min(state.last_progress_ratio - candidate.progress_ratio, 0.6)

        path_penalty = clamp(candidate.distance_to_path_m / max(250.0, request.direct_distance_m * 0.18))
        return incremental_detour, (incremental_detour / max(request.distance_budget_m, 1)) + backward_penalty + 0.35 * path_penalty

    def route_duration_minutes(self, walked_distance_m, stop_minutes, request):
        walking_minutes = walked_distance_m / max(request.speed_m_per_min, 1)
        return int(round(walking_minutes + stop_minutes))

    def extend_state(self, state, candidate, request, candidate_map):
        if candidate.point.id in state.point_ids:
            return None

        avoid_categories = set(request.hard_constraints.get("avoid_categories", []))
        if set(candidate.point.categories).intersection(avoid_categories):
            return None

        sequence = request.hard_constraints.get("must_visit_in_order", [])
        seq_progress = state.seq_progress
        if sequence:
            if seq_progress < len(sequence) and term_matches_point(sequence[seq_progress], candidate.point):
                seq_progress += 1
            elif self.point_matches_future_sequence_slot(candidate.point, sequence, state.seq_progress):
                return None

        additional_food = 1 if candidate.point.other.get("is_food_venue") else 0
        new_food_count = state.food_count + additional_food
        if new_food_count > request.hard_constraints.get("food_limit", 1):
            return None

        leg_distance = state.last_point.dist_between_points(candidate.point)
        projected_total_distance = (
            state.walked_distance_m
            + leg_distance
            + candidate.point.dist_between_points(request.end)
        )
        if projected_total_distance > request.hard_constraints["max_total_distance_m"]:
            return None

        stop_minutes = state.stop_minutes + candidate.estimated_stop_minutes
        projected_duration = self.route_duration_minutes(
            state.walked_distance_m + leg_distance + candidate.point.dist_between_points(request.end),
            stop_minutes,
            request,
        )
        if projected_duration > request.hard_constraints["max_total_duration_min"]:
            return None

        added_cost = candidate.point.estimated_cost_rub or 0
        projected_cost = state.cost_rub + added_cost
        max_budget = request.hard_constraints.get("max_total_budget_rub")
        if max_budget is not None and projected_cost > max_budget:
            return None

        remaining_slots = request.hard_constraints["max_intermediate_stops"] - len(state.point_ids) - 1
        required_terms = request.hard_constraints.get("required_categories", [])
        required_satisfied = set(state.required_satisfied)
        for term in required_terms:
            if term_matches_point(term, candidate.point):
                required_satisfied.add(term)
        missing_required = [term for term in required_terms if term not in required_satisfied]
        missing_sequence = max(len(sequence) - seq_progress, 0)
        if remaining_slots < max(len(missing_required), missing_sequence):
            return None

        incremental_detour_m, transition_penalty = self.transition_penalty(state, candidate, request)
        if incremental_detour_m > max(900, int(request.distance_budget_m * 0.42)):
            return None

        new_category_counts = dict(state.category_counts)
        category = candidate.point.primary_category or "generic"
        new_category_counts[category] = new_category_counts.get(category, 0) + 1

        point_score_sum = state.point_score_sum + candidate.total_score
        beam_score = point_score_sum - (state.transition_penalty_sum + transition_penalty) * 5.0

        return RouteState(
            point_ids=state.point_ids + (candidate.point.id,),
            last_point=candidate.point,
            walked_distance_m=state.walked_distance_m + leg_distance,
            stop_minutes=stop_minutes,
            cost_rub=projected_cost,
            food_count=new_food_count,
            category_counts=new_category_counts,
            seq_progress=seq_progress,
            required_satisfied=frozenset(required_satisfied),
            point_score_sum=point_score_sum,
            transition_penalty_sum=state.transition_penalty_sum + transition_penalty,
            beam_score=beam_score,
            last_progress_ratio=candidate.progress_ratio,
            total_path_deviation_m=state.total_path_deviation_m + candidate.distance_to_path_m,
        )

    def finalize_state(self, state, request, candidate_map):
        total_distance_m = state.walked_distance_m + state.last_point.dist_between_points(request.end)
        total_duration_min = self.route_duration_minutes(total_distance_m, state.stop_minutes, request)
        max_budget = request.hard_constraints.get("max_total_budget_rub")

        if total_distance_m > request.hard_constraints["max_total_distance_m"]:
            return None
        if total_duration_min > request.hard_constraints["max_total_duration_min"]:
            return None
        if max_budget is not None and state.cost_rub > max_budget:
            return None
        if state.seq_progress < len(request.hard_constraints.get("must_visit_in_order", [])):
            return None
        missing_required = [
            term
            for term in request.hard_constraints.get("required_categories", [])
            if term not in state.required_satisfied
        ]
        if missing_required:
            return None

        stop_points = [self.db[point_id] for point_id in state.point_ids]
        route_score, route_reasons = self.score_route(
            state,
            stop_points,
            total_distance_m,
            total_duration_min,
            request,
            candidate_map,
        )
        stop_reasons = {
            point_id: candidate_map[point_id].reasons
            for point_id in state.point_ids
            if point_id in candidate_map
        }
        return RoutePlan(
            stop_points=stop_points,
            total_distance_m=total_distance_m,
            estimated_duration_min=total_duration_min,
            estimated_cost_rub=state.cost_rub,
            route_score=route_score,
            route_reasons=route_reasons,
            stop_reasons=stop_reasons,
            candidate_scores=candidate_map,
            debug={},
        )

    def score_route(self, state, stop_points, total_distance_m, total_duration_min, request, candidate_map):
        distinct_categories = len({point.primary_category or "generic" for point in stop_points})
        diversity_bonus = distinct_categories / max(len(stop_points), 1) if stop_points else 0.0
        avg_path_deviation = state.total_path_deviation_m / max(len(stop_points), 1) if stop_points else 0.0
        coherence_bonus = clamp(1.0 - (avg_path_deviation / max(250.0, request.direct_distance_m * 0.18)))
        coherence_bonus = 0.5 * coherence_bonus + 0.5 * clamp(state.last_progress_ratio)

        constraint_completion_bonus = 1.0
        total_detour_penalty = clamp((total_distance_m - request.direct_distance_m) / max(request.distance_budget_m, 1), 0.0, 1.2)
        overbudget_penalty = 0.0
        if request.hard_constraints.get("max_total_budget_rub") is not None:
            max_budget = request.hard_constraints["max_total_budget_rub"]
            overbudget_penalty = clamp((state.cost_rub - max_budget) / max(max_budget, 1), 0.0, 1.0)

        category_repeat_penalty = (
            sum(max(count - 1, 0) for count in state.category_counts.values()) / max(len(stop_points), 1)
            if stop_points
            else 0.0
        )

        duration_ratio = total_duration_min / max(request.total_minutes, 1)
        underfilled_route_penalty = clamp((0.75 - duration_ratio) / 0.75, 0.0, 1.0)
        overfilled_route_penalty = clamp((duration_ratio - 1.0) / 0.25, 0.0, 1.0)

        route_score = (
            state.point_score_sum
            + ROUTE_SCORE_WEIGHTS["diversity_bonus"] * diversity_bonus
            + ROUTE_SCORE_WEIGHTS["coherence_bonus"] * coherence_bonus
            + ROUTE_SCORE_WEIGHTS["constraint_completion_bonus"] * constraint_completion_bonus
            - ROUTE_SCORE_WEIGHTS["total_detour_penalty"] * total_detour_penalty
            - ROUTE_SCORE_WEIGHTS["overbudget_penalty"] * overbudget_penalty
            - ROUTE_SCORE_WEIGHTS["category_repeat_penalty"] * category_repeat_penalty
            - ROUTE_SCORE_WEIGHTS["underfilled_route_penalty"] * underfilled_route_penalty
            - ROUTE_SCORE_WEIGHTS["overfilled_route_penalty"] * overfilled_route_penalty
        )

        route_reasons = []
        if request.hard_constraints.get("must_visit_in_order"):
            route_reasons.append("respects the requested stop order")
        if diversity_bonus >= 0.75 and stop_points:
            route_reasons.append("balances different types of places")
        if coherence_bonus >= 0.65:
            route_reasons.append("keeps the route coherent and close to the direct path")
        if total_detour_penalty <= 0.35:
            route_reasons.append("avoids major detours")
        if request.hard_constraints.get("max_total_budget_rub") is not None and state.cost_rub <= request.hard_constraints["max_total_budget_rub"]:
            route_reasons.append("stays within the budget")
        if duration_ratio >= 0.8:
            route_reasons.append("fills the requested walk duration reasonably well")
        if not route_reasons:
            route_reasons.append("provides the best trade-off between relevance and efficiency")

        return route_score, route_reasons

    def beam_search(self, narrowed_candidates, request: PlanningRequest):
        candidate_map = {candidate.point.id: candidate for candidate in narrowed_candidates}
        initial_state = RouteState(
            point_ids=(),
            last_point=request.start,
            walked_distance_m=0,
            stop_minutes=0,
            cost_rub=0,
            food_count=0,
            category_counts={},
            seq_progress=0,
            required_satisfied=frozenset(),
            point_score_sum=0.0,
            transition_penalty_sum=0.0,
            beam_score=0.0,
            last_progress_ratio=0.0,
            total_path_deviation_m=0.0,
        )

        beam = [initial_state]
        completed = []
        seen_completed = set()

        for _depth in range(request.hard_constraints["max_intermediate_stops"] + 1):
            next_beam = []
            for state in beam:
                finalized = self.finalize_state(state, request, candidate_map)
                if finalized is not None:
                    signature = tuple(finalized.stop_ids)
                    if signature not in seen_completed:
                        completed.append(finalized)
                        seen_completed.add(signature)

                if len(state.point_ids) >= request.hard_constraints["max_intermediate_stops"]:
                    continue

                for candidate in narrowed_candidates:
                    new_state = self.extend_state(state, candidate, request, candidate_map)
                    if new_state is None:
                        continue
                    next_beam.append(new_state)

            if not next_beam:
                break

            next_beam.sort(key=lambda state: state.beam_score, reverse=True)
            beam = next_beam[:DEFAULT_BEAM_WIDTH]

            if len(completed) >= DEFAULT_COMPLETED_LIMIT:
                break

        completed.sort(key=lambda route: route.route_score, reverse=True)
        logger.info("Beam search produced {} completed routes", len(completed))
        return completed[:DEFAULT_COMPLETED_LIMIT]

    def build_direct_route(self, request: PlanningRequest):
        reasons = ["keeps a simple direct walk between the requested endpoints"]
        if request.hard_constraints.get("max_total_budget_rub") is not None:
            reasons.append("avoids unnecessary spending")
        return RoutePlan(
            stop_points=[],
            total_distance_m=request.direct_distance_m,
            estimated_duration_min=self.route_duration_minutes(request.direct_distance_m, 0, request),
            estimated_cost_rub=0,
            route_score=0.0,
            route_reasons=reasons,
            stop_reasons={},
            candidate_scores={},
            debug={"fallback": "direct_route"},
        )

    def plan(self, request: PlanningRequest):
        retrieved, queries = self.retrieve_candidates(request)
        scored = self.score_candidates(retrieved, request)
        narrowed = self.narrow_candidates(scored, request)
        routes = self.beam_search(narrowed, request)

        if routes:
            best_route = routes[0]
            best_route.debug.update(
                {
                    "retrieval_queries": queries,
                    "retrieved_count": len(retrieved),
                    "narrowed_count": len(narrowed),
                    "completed_route_count": len(routes),
                }
            )
            return best_route

        logger.warning("No strict route found, falling back to direct route")
        direct_route = self.build_direct_route(request)
        direct_route.debug.update(
            {
                "retrieval_queries": queries,
                "retrieved_count": len(retrieved),
                "narrowed_count": len(narrowed),
                "completed_route_count": 0,
            }
        )
        return direct_route

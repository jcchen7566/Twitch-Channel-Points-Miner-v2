"""Unsupervised title clustering via character bigram TF-IDF + dimension reduction + DBSCAN.

Discovers natural game-type groupings from Chinese-language betting titles
without manual labels or NLP. Uses character 2-grams which capture shared
substrings across titles (e.g. 誰會贏 → 誰會, 會贏; baseball props → 安打/全壘打/失分/ER).
"""

import math
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


class TitleClusterer:
    """Character bigram-based title clusterer.

    Fit on a list of titles → each title assigned to a cluster label (-1 for noise).
    Predict for new titles via nearest-cluster cosine similarity.
    """

    def __init__(self, svd_dims: int = 10, eps: float = 0.3, min_samples: int = 1):
        self.vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(2, 2),
            max_features=2000,
            binary=True,
        )
        self.svd = TruncatedSVD(n_components=svd_dims, random_state=42)
        self.clustering = DBSCAN(eps=eps, min_samples=min_samples)
        self._fitted = False

    def fit(self, titles: list[str]) -> "TitleClusterer":
        titles = list(titles)
        n = len(titles)
        if n < 3:
            # Too few titles — every title is its own singleton cluster
            self.labels_ = list(range(n))
            self.title_to_label_ = dict(zip(titles, self.labels_))
            self.cluster_centers_ = {}
            self._fitted = True
            return self

        X = self.vectorizer.fit_transform(titles)
        Xr = self.svd.fit_transform(X)
        self.labels_ = self.clustering.fit_predict(Xr)
        self.title_to_label_ = dict(zip(titles, self.labels_))

        # Compute cluster centers in SVD space
        centers = {}
        for idx, label in enumerate(self.labels_):
            centers.setdefault(label, []).append(Xr[idx])
        self.cluster_centers_ = {
            label: np.mean(vecs, axis=0) for label, vecs in centers.items()
        }

        self._fitted = True
        return self

    def predict(self, title: str) -> int:
        if not self._fitted:
            return -1
        if title in self.title_to_label_:
            return self.title_to_label_[title]
        # New title: nearest-cluster cosine similarity
        X = self.vectorizer.transform([title])
        Xr = self.svd.transform(X)
        vec = Xr[0]
        best_label = -1
        best_sim = -1.0
        for label, center in self.cluster_centers_.items():
            sim = float(np.dot(vec, center) / (np.linalg.norm(vec) * np.linalg.norm(center) + 1e-10))
            if sim > best_sim:
                best_sim = sim
                best_label = label
        return best_label

    @property
    def n_clusters(self) -> int:
        return len(set(self.labels_)) if hasattr(self, "labels_") else 0

    def cluster_sizes(self) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for v in self.title_to_label_.values():
            counts[v] += 1
        return dict(counts)


# ── Legacy backward-compatible interface ───────────────────────────────

def cluster_title(title: str) -> str:
    """
    Legacy title regime classifier. Used by game_regime_learner,
    multi_alpha_regime_expert, slrabbit_specialist_v2.
    """
    t = title.lower()
    if any(k in t for k in ["lol", "英雄聯盟", "勝敗", "輸贏", "潘森", "這把"]):
        return "REGIME_LOL"
    if any(k in t for k in ["tft", "戰棋", "棋", "第幾"]):
        return "REGIME_TFT"
    if any(k in t for k in ["大富翁", "彈珠", "賭", "過關", "幾場"]):
        return "REGIME_LUCK"
    if any(k in t for k in ["聊", "唱歌", "日常"]):
        return "REGIME_CHAT"
    return "REGIME_VARIETY"


def get_event_regime(event) -> str:
    """Legacy composite key: STREAMER_GAME."""
    return f"{event.streamer}_{cluster_title(event.main_title)}"


# ── Online Dynamic Cluster Manager (sklearn-free) ─────────────────────

class _Cluster:
    """An online cluster of titles — maintains cumulative n-gram centroid."""
    __slots__ = ("name", "ngram_sum", "count")

    def __init__(self, name: str = ""):
        self.name = name
        self.ngram_sum: dict[str, float] = defaultdict(float)
        self.count = 0

    def add(self, vec: dict[str, float]) -> None:
        for k, v in vec.items():
            self.ngram_sum[k] += v
        self.count += 1

    @property
    def centroid(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {k: v / self.count for k, v in self.ngram_sum.items()}

    def cosine(self, vec: dict[str, float]) -> float:
        cent = self.centroid
        if not cent or not vec:
            return 0.0
        dot = sum(v * cent.get(k, 0.0) for k, v in vec.items())
        n1 = math.sqrt(sum(v * v for v in vec.values()))
        n2 = math.sqrt(sum(v * v for v in cent.values()))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)


# Default keyword fallback for slrabbit99 seed clusters (0-6).
# Excludes CATCH_ALL — anything unmatched by keywords becomes a dynamic cluster.
_SLRABBIT_KEYWORDS = [
    (["誰會贏", "哪邊會贏", " vs ", "誰贏"], 0),
    (["失分", "勝投", "全壘打", "安打", "觸身", "er ", "er)"], 1),
    (["命一關不刷", "一關不刷"], 2),
    (["第幾名", "開到幾星", "抽"], 3),
    (["會不會過關", "能不能過關", "一把決勝負"], 4),
    (["最穩的那台"], 5),
    (["丟骨頭"], 6),
]


def _default_keyword_fallback(title: str) -> int | None:
    """Match title against v1 keyword patterns. Returns seed idx or None."""
    t = (title or "").strip().lower()
    for keywords, idx in _SLRABBIT_KEYWORDS:
        if any(k in t for k in keywords):
            return idx
    return None


class DynamicClusterManager:
    """Online dynamic clustering for short CJK titles.

    Hybrid approach:
    1. Try n-gram cosine similarity to all existing cluster centroids.
    2. If best match >= threshold → assign there.
    3. Else try keyword fallback (for seed clusters) → assign there.
    4. Else create a new dynamic cluster.

    Seed clusters are pre-populated from keyword examples. New clusters
    form organically when titles don't resemble any existing cluster.
    """

    def __init__(
        self,
        seed_examples: list[list[str]],
        threshold: float = 0.25,
        keyword_fallback=None,
    ):
        self.threshold = threshold
        self._keyword_fallback = keyword_fallback
        self.clusters: list[_Cluster] = []
        self.title_map: dict[str, int] = {}

        for examples in seed_examples:
            c = _Cluster()
            for ex in examples:
                v = self._ngrams(ex)
                if any(x > 0 for x in v.values()):
                    c.add(v)
            self.clusters.append(c)

    @staticmethod
    def _ngrams(text: str, n_range=(2, 3)) -> dict[str, float]:
        """Bigram + trigram frequency vector (no unigrams — too noisy)."""
        text = text.lower().strip()
        g: dict[str, float] = defaultdict(float)
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(text) - n + 1):
                g[text[i:i + n]] += 1.0
        return dict(g)

    def assign(self, title: str) -> int:
        """Return cluster index — may create a new dynamic cluster."""
        cached = self.title_map.get(title)
        if cached is not None:
            return cached

        vec = self._ngrams(title)

        # Step 1 — n-gram cosine to all existing centroids
        best_idx = -1
        best_sim = -1.0
        for i, c in enumerate(self.clusters):
            sim = c.cosine(vec)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        # Step 2 — keyword fallback for seed clusters
        if best_sim < self.threshold and self._keyword_fallback is not None:
            kw = self._keyword_fallback(title)
            if kw is not None:
                self.clusters[kw].add(vec)
                self.title_map[title] = kw
                return kw

        # Step 3 — threshold sufficient → assign to best match
        if best_sim >= self.threshold:
            self.clusters[best_idx].add(vec)
            self.title_map[title] = best_idx
            return best_idx

        # Step 4 — truly novel → spawn new dynamic cluster
        new_idx = len(self.clusters)
        c = _Cluster(f"dyn_{new_idx}")
        c.add(vec)
        self.clusters.append(c)
        self.title_map[title] = new_idx
        return new_idx

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

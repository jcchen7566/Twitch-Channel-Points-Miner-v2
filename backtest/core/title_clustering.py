"""Unsupervised title clustering via character bigram TF-IDF + dimension reduction + DBSCAN.

Discovers natural game-type groupings from Chinese-language betting titles
without manual labels or NLP. Uses character 2-grams which capture shared
substrings across titles (e.g. 誰會贏 → 誰會, 會贏; baseball props → 安打/全壘打/失分/ER).
"""

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

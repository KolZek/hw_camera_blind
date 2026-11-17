from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import Detection
from .kalman import KalmanCV


@dataclass(frozen=True)
class TrackView:
    """Состояние трека для счётчика, записи CSV и визуализации.

    bbox — (x, y, width, height), center — (cx, cy), координаты в пикселях.
    source — "measurement" или "prediction"; confirmed — подтверждён ли трек.
    missed — число последовательных пропусков; recovered — вернулось ли
    измерение после пропуска именно в текущем кадре.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    source: str
    confirmed: bool
    missed: int
    recovered: bool = False


class _Track:
    __slots__ = (
        "track_id",
        "kf",
        "size",
        "hits",
        "missed",
        "confirmed",
        "recovered",
        "source",
    )

    def __init__(
        self, track_id: int, kf: KalmanCV, size: tuple[float, float], min_hits: int
    ) -> None:
        self.track_id = track_id
        self.kf = kf
        self.size = (float(size[0]), float(size[1]))
        self.hits = 1
        self.missed = 0
        self.confirmed = 1 >= min_hits
        self.recovered = False
        self.source = "measurement"

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        cx, cy = self.kf.center
        w, h = self.size
        return (float(cx - w / 2.0), float(cy - h / 2.0), w, h)


class MultiObjectTracker:
    """Сопровождать объекты по центрам обнаружений.

    Параметры
    ----------
    min_hits : int
        Число успешных наблюдений для подтверждения трека, включая первое.
    max_missed : int
        Допустимое число последовательных кадров без измерения.
        Удаляйте трек, когда missed становится больше max_missed.
    max_distance : float
        Максимальное расстояние между прогнозом и измерением в пикселях.
    process_var, measurement_var : float
        Параметры фильтра Калмана.
    """

    def __init__(
        self,
        min_hits: int = 3,
        max_missed: int = 15,
        max_distance: float = 80.0,
        process_var: float = 25.0,
        measurement_var: float = 36.0,
    ) -> None:
        if int(min_hits) < 1:
            raise ValueError("min_hits must be >= 1")

        if int(max_missed) < 0:
            raise ValueError("max_missed must be >= 0")

        self.min_hits = int(min_hits)
        self.max_missed = int(max_missed)
        self.max_distance = float(max_distance)
        self.process_var = float(process_var)
        self.measurement_var = float(measurement_var)
        self._tracks: list[_Track] = []
        self._next_id = 1

    def step(self, detections: Iterable[Detection], dt: float = 1.0) -> list[TrackView]:
        """Обработать один кадр и вернуть все активные треки, включая прогнозы.

        detections — обнаружения текущего кадра; пустой список означает пропуск.
        dt — число кадров с прошлого шага. Каждое измерение сопоставляется
        не более чем одному треку. ID удалённых треков не используйте повторно.
        """
        detections = list(detections)

        prev_missed: dict[int, int] = {}
        for trk in self._tracks:
            prev_missed[trk.track_id] = trk.missed
            trk.kf.predict(dt)
            trk.missed += 1
            trk.recovered = False
            trk.source = "prediction"

        matched_t: set[int] = set()
        matched_d: set[int] = set()
        n_t, n_d = len(self._tracks), len(detections)
        if n_t and n_d:
            INF = np.inf
            cost = np.full((n_t, n_d), INF, dtype=float)
            for i, trk in enumerate(self._tracks):
                tc = trk.kf.center
                for j, det in enumerate(detections):
                    dc = (det.x + det.width / 2.0, det.y + det.height / 2.0)
                    dist = float(np.hypot(tc[0] - dc[0], tc[1] - dc[1]))
                    if dist <= self.max_distance:
                        cost[i, j] = dist

            BIG = 1e6
            cost_safe = np.where(np.isfinite(cost), cost, BIG)
            rows, cols = linear_sum_assignment(cost_safe)
            for r, c in zip(rows, cols):
                if not np.isfinite(cost[r, c]):
                    continue
                trk = self._tracks[r]
                det = detections[c]
                dc = (det.x + det.width / 2.0, det.y + det.height / 2.0)
                trk.kf.update(dc)
                trk.size = (det.width, det.height)
                trk.missed = 0
                trk.hits += 1
                if trk.hits >= self.min_hits:
                    trk.confirmed = True
                trk.recovered = prev_missed.get(trk.track_id, 0) > 0
                trk.source = "measurement"
                matched_t.add(r)
                matched_d.add(c)

        for j, det in enumerate(detections):
            if j in matched_d:
                continue

            dc = (det.x + det.width / 2.0, det.y + det.height / 2.0)
            kf = KalmanCV(dc, self.process_var, self.measurement_var)

            self._tracks.append(
                _Track(self._next_id, kf, (det.width, det.height), self.min_hits)
            )
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]

        views: list[TrackView] = []
        for trk in self._tracks:
            cx, cy = trk.kf.center
            views.append(
                TrackView(
                    track_id=trk.track_id,
                    bbox=trk.bbox,
                    center=(float(cx), float(cy)),
                    source=trk.source,
                    confirmed=trk.confirmed,
                    missed=trk.missed,
                    recovered=trk.recovered,
                )
            )

        return views

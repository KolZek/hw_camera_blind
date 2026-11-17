from __future__ import annotations

from collections.abc import Iterable


class LineCounter:
    """Считать входы и выходы через конечный отрезок start → end.

    start, end — координаты концов в пикселях. entry_direction задаёт вход:
    "negative_to_positive" или "positive_to_negative" по знаку векторного
    произведения (end - start) и (center - start). Для горизонтальной линии
    слева направо первое значение соответствует движению сверху вниз.
    initial_occupancy — начальная заполненность. Создайте открытые счётчики
    entered и exited; начальное значение обоих равно нулю.
    """

    def __init__(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        entry_direction: str = "negative_to_positive",
        initial_occupancy: int = 0,
    ) -> None:
        if entry_direction not in ("negative_to_positive", "positive_to_negative"):
            raise ValueError(f"unknown entry_direction: {entry_direction!r}")
        if (float(start[0]), float(start[1])) == (float(end[0]), float(end[1])):
            raise ValueError("line endpoints must differ")

        self.start = (float(start[0]), float(start[1]))
        self.end = (float(end[0]), float(end[1]))
        self.entry_direction = entry_direction
        self.initial_occupancy = int(initial_occupancy)
        self._entered = 0
        self._exited = 0

        self._last_side: dict[int, int] = {}
        self._last_point: dict[int, tuple[float, float]] = {}

        self._entered_ids: set[int] = set()
        self._exited_ids: set[int] = set()

    @property
    def entered(self) -> int:
        return self._entered

    @property
    def exited(self) -> int:
        return self._exited

    @property
    def current(self) -> int:
        """Вернуть баланс initial_occupancy + entered - exited без обрезания."""
        return self.initial_occupancy + self._entered - self._exited

    def _side(self, point: tuple[float, float]) -> int:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        vx = point[0] - self.start[0]
        vy = point[1] - self.start[1]

        cross = dx * vy - dy * vx
        if cross > 1e-9:
            return 1
        if cross < -1e-9:
            return -1

        return 0

    def _crossing_param(
        self, p1: tuple[float, float], p2: tuple[float, float]
    ) -> float | None:
        """Параметр t точки пересечения прямых на отрезке start→end, или None."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        wx = p2[0] - p1[0]
        wy = p2[1] - p1[1]
        vx = self.start[0] - p1[0]
        vy = self.start[1] - p1[1]
        denom = wx * dy - wy * dx

        if abs(denom) < 1e-12:
            return None

        t = (vx * wy - vy * wx) / denom
        return t

    def observe(
        self, track_id: int, center: Iterable[float], measured: bool = True
    ) -> str | None:
        """Принять центр трека и вернуть "entered", "exited" либо None.

        measured=False означает прогноз: он не подтверждает пересечение.
        Считайте каждое направление не более одного раза для одного ID.
        Точка точно на линии не должна стирать последнее положение вне неё.
        """
        point = (float(center[0]), float(center[1]))
        side = self._side(point)

        if side == 0:
            return None

        prev_side = self._last_side.get(track_id)
        prev_point = self._last_point.get(track_id)

        if prev_side is None:
            self._last_side[track_id] = side
            self._last_point[track_id] = point
            return None

        if side == prev_side:
            self._last_point[track_id] = point
            return None

        if not measured:
            return None

        t = self._crossing_param(prev_point, point)
        if t is None or t < 0.0 or t > 1.0:
            self._last_side[track_id] = side
            self._last_point[track_id] = point
            return None

        self._last_side[track_id] = side
        self._last_point[track_id] = point

        if self.entry_direction == "negative_to_positive":
            entering = prev_side < 0 < side
        else:
            entering = prev_side > 0 > side

        if entering:
            if track_id in self._entered_ids:
                return None
            self._entered_ids.add(track_id)
            self._entered += 1
            return "entered"
        else:
            if track_id in self._exited_ids:
                return None
            self._exited_ids.add(track_id)
            self._exited += 1
            return "exited"

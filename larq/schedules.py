from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from cached_property import cached_property


class AbstractSchedule(ABC):
    # TODO: Make this a component once Zookeeper has a post-init hook.

    update_freq: str = "step"
    subtract_start_time: bool = True

    @abstractmethod
    def __call__(self, t: int, start_time: int = 0) -> float:
        raise NotImplementedError


@dataclass
class StepDecaySchedule(AbstractSchedule):
    # TODO: needs documentation

    initial_value: float
    decay_steps: List[int]
    decay_factor: float
    update_freq: str = "epoch"
    subtract_start_time: bool = False

    def __call__(self, t: int, start_time: int = 0) -> float:
        # Subtract `start_time` from `t` to ensure the schedule starts at its beginning
        if self.subtract_start_time:
            t -= start_time

        value = self.initial_value
        for step in self.decay_steps:
            if step <= t:
                value *= self.decay_factor
            else:
                break
        return value


@dataclass
class Schedule(AbstractSchedule):
    schedule: Callable[[int], Any]

    def __call__(self, t: int, start_time: int = 0) -> float:
        # Subtract `start_time` from `t` to ensure the schedule starts at its beginning
        if self.subtract_start_time:
            t -= start_time

        return self.schedule(t)


@dataclass
class CombinedSchedule(AbstractSchedule):
    # TODO: needs documentation

    schedules: List[AbstractSchedule]
    starting_times: List[int]

    _current_index: int = field(default=-1, init=False, repr=False)
    _next_start: Optional[int] = field(default=None, init=False, repr=False)
    _previous_start: Optional[int] = field(default=None, init=False, repr=False)

    @cached_property
    def update_freq(self):
        return self.schedules[0].update_freq

    def __post_init__(self):
        if len(set(s.update_freq for s in self.schedules)) != 1:
            raise ValueError("All schedules must have the same `update_freq`.")

        if len(self.starting_times) != len(self.schedules) - 1:
            raise ValueError(
                "'starting_times' must have length 'len(schedules) - 1'. "
                f"Received list of length {len(self.starting_times)}."
            )

        if not all(
            x < y for x, y in zip([0] + self.starting_times, self.starting_times)
        ):
            raise ValueError(
                "`starting_times` should be positive and strictly monotonically "
                f"increasing! Received {self.starting_times}."
            )

    def _find_current_index(self, t: int) -> int:
        index = 0
        for start in self.starting_times:
            if t >= start:
                index += 1
            else:
                break

        return index

    def __call__(self, t: int) -> float:
        # If this is the first time this is called, select the
        # appropriate schedule given t
        if self._current_index < 0:
            self._current_index = self._find_current_index(t)
            self._next_start = (
                self.starting_times[self._current_index]
                if self._current_index < len(self.starting_times)
                else None
            )
            self._previous_start = (
                self.starting_times[self._current_index - 1]
                if self._current_index > 0
                else 0
            )

        # Check if we need to switch to the next schedule
        if self._next_start and t >= self._next_start:
            self._current_index += 1
            self._previous_start = self._next_start
            self._next_start = (
                self.starting_times[self._current_index]
                if self._current_index < len(self.starting_times)
                else None
            )

        return self.schedules[self._current_index](t=t, start_time=self._previous_start)

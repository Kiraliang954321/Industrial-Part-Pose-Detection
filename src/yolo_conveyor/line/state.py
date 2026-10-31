"""Explicit indexed-cell sequencing and one-time event/action latches."""

from enum import Enum
import math


class CellState(str, Enum):
    IDLE = "IDLE"
    INDEXING = "INDEXING"
    SETTLING = "SETTLING"
    OBSERVING = "OBSERVING"
    RELEASING = "RELEASING"
    REJECTING = "REJECTING"
    HALTED = "HALTED"
    COMPLETE = "COMPLETE"
    EXECUTION_FAILED = "EXECUTION_FAILED"


class CellSequencer:
    def __init__(self, job_id):
        self.job_id = job_id
        self.state = CellState.IDLE
        self.transitions = []
        self.event = None

    def _change(self, expected, destination, time_s, reason):
        if not math.isfinite(time_s) or (self.transitions and time_s < self.transitions[-1]["sim_time_s"]):
            raise ValueError("state transition times must be finite and monotonic")
        if self.state != expected:
            raise ValueError(f"invalid transition from {self.state} (expected {expected})")
        self.transitions.append({"from": self.state.value, "to": destination.value, "sim_time_s": time_s, "reason": reason})
        self.state = destination

    def start(self, time_s):
        self._change(CellState.IDLE, CellState.INDEXING, time_s, "PLC_job_loaded")

    def arrived(self, time_s):
        self._change(CellState.INDEXING, CellState.SETTLING, time_s, "index_encoder_at_station")

    def inspect(self, time_s):
        self._change(CellState.SETTLING, CellState.OBSERVING, time_s, "settling_complete")

    def decide(self, event, time_s):
        if self.event is not None or event["event_id"] != self.job_id:
            raise ValueError("duplicate or mismatched final event")
        destinations = {"PASS": CellState.RELEASING, "FAIL_ORIENTATION": CellState.REJECTING, "UNCERTAIN": CellState.HALTED}
        if event["outcome"] not in destinations:
            raise ValueError("unsupported event outcome")
        self._change(CellState.OBSERVING, destinations[event["outcome"]], time_s, event["reason"])
        self.event = event

    def complete(self, time_s, success):
        if self.state not in (CellState.RELEASING, CellState.REJECTING):
            raise ValueError("only an automatic action can complete; a halted cell needs explicit recovery")
        self._change(self.state, CellState.COMPLETE if success else CellState.EXECUTION_FAILED,
                     time_s, "action_completed" if success else "action_failed")

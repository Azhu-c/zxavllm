# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-local fairness state, measured in scheduler steps."""

from dataclasses import dataclass, field


@dataclass
class DyntraLBPausePolicy:
    max_pause_steps: int
    min_run_steps: int
    paused_at: dict[str, int] = field(default_factory=dict)
    protected_until: dict[str, int] = field(default_factory=dict)

    def can_pause(self, request_id: str, step: int) -> bool:
        return step >= self.protected_until.get(request_id, 0)

    def pause(self, request_id: str, step: int) -> None:
        self.paused_at.setdefault(request_id, step)

    def resume(self, request_id: str, step: int) -> None:
        if self.paused_at.pop(request_id, None) is not None:
            # The resumed step is the first protected decoding step.
            self.protected_until[request_id] = step + self.min_run_steps

    def overdue(self, step: int) -> list[str]:
        return [req_id for req_id, paused in self.paused_at.items() if step - paused >= self.max_pause_steps]

    def discard(self, request_id: str) -> None:
        self.paused_at.pop(request_id, None)
        self.protected_until.pop(request_id, None)

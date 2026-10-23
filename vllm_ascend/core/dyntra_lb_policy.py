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


@dataclass
class DyntraLBFirstTokenPolicy:
    """Bound LB-only admission deferrals before the first generated token.

    A bypass only opens the LB gate: KV readiness, batch capacity, priorities,
    and token budgets still belong to the normal scheduler. The per-step cap
    limits extra admission attempts; this is not an end-to-end TTFT deadline.
    """

    max_wait_steps: int
    max_admissions_per_step: int
    deferred: dict[str, tuple[int, int]] = field(default_factory=dict)
    admission_step: int = -1
    admissions_this_step: int = 0
    admitted: set[str] = field(default_factory=set)

    def can_admit(self, request_id: str, step: int, selected: bool, has_output: bool) -> bool:
        if has_output:
            self.discard(request_id)
            return selected
        if selected:
            # A selection need not result in computation (KV/budgets can block
            # it). Preserve the deferral history until output or request cleanup.
            return True
        if self.admission_step != step:
            self.admission_step = step
            self.admissions_this_step = 0
            self.admitted.clear()
        if request_id in self.admitted:
            return True
        count, last_step = self.deferred.get(request_id, (0, -1))
        if count >= self.max_wait_steps:
            if self.admissions_this_step < self.max_admissions_per_step:
                self.admitted.add(request_id)
                self.admissions_this_step += 1
                return True
            return False
        if last_step != step:
            self.deferred[request_id] = count + 1, step
        return False

    def discard(self, request_id: str) -> None:
        self.deferred.pop(request_id, None)
        # Cleanup must not refund an admission already attempted this step.
        self.admitted.discard(request_id)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

from vllm_ascend.core.dyntra_cp_policy import DynamicCPConfig


class DyntraLBConfig:
    """Configuration object for ``additional_config["scheduler_config"]["dyntra_lb_config"]``."""

    _defaults = {
        "enabled": False,
        "enable_diagnostics": False,
        "mode": "dynamic",
        "policy_profile": "tpot",
        "first_token_max_lb_wait_steps": 4,
        "first_token_max_admissions_per_step": 1,
        "start_step": 250,
        "end_step": -1,
        "bubble_threshold": 5.0,
        "long_req_block_threshold": 700,
        "dynamic_max_step": 256,
        "balance_interval": 1,
        "max_pause_steps": 4,
        "min_run_steps": 4,
        "max_pauses_per_rank": 1,
        "min_throughput_gain": 0.02,
        "step_latency_fixed_us": 20000.0,
        "step_latency_per_block_us": 19.2,
        "dynamic_cp_config": None,
    }
    _valid_modes = {"static", "dynamic"}

    def __init__(self, user_config: dict | None = None):
        if user_config is None:
            user_config = {}
        elif not isinstance(user_config, dict):
            raise ValueError(f"dyntra_lb_config must be a dict, got {type(user_config).__name__}.")

        unknown = set(user_config) - set(self._defaults)
        if unknown:
            raise ValueError(f"Unknown dyntra_lb_config keys: {sorted(unknown)}")

        self.enabled = user_config.get("enabled", self._defaults["enabled"])
        self.enable_diagnostics = user_config.get(
            "enable_diagnostics",
            self._defaults["enable_diagnostics"],
        )
        self.mode = user_config.get("mode", self._defaults["mode"])
        self.policy_profile = user_config.get("policy_profile", self._defaults["policy_profile"])
        self.start_step = user_config.get("start_step", self._defaults["start_step"])
        self.end_step = user_config.get("end_step", self._defaults["end_step"])
        self.bubble_threshold = user_config.get("bubble_threshold", self._defaults["bubble_threshold"])
        self.long_req_block_threshold = user_config.get(
            "long_req_block_threshold",
            self._defaults["long_req_block_threshold"],
        )
        self.dynamic_max_step = user_config.get("dynamic_max_step", self._defaults["dynamic_max_step"])
        self.dynamic_cp_config = DynamicCPConfig(user_config.get("dynamic_cp_config"))
        for key in (
            "first_token_max_lb_wait_steps",
            "first_token_max_admissions_per_step",
            "balance_interval",
            "max_pause_steps",
            "min_run_steps",
            "max_pauses_per_rank",
            "min_throughput_gain",
            "step_latency_fixed_us",
            "step_latency_per_block_us",
        ):
            setattr(self, key, user_config.get(key, self._defaults[key]))
        self._validate_config()

    def _validate_config(self):
        if not isinstance(self.policy_profile, str) or self.policy_profile not in {"original", "tpot", "ttft"}:
            raise ValueError("dyntra_lb_config.policy_profile must be original, tpot, or ttft")
        if self.dynamic_cp_config.enabled and self.policy_profile != "tpot":
            raise ValueError("The original/TTFT comparison profiles require dynamic_cp_config.enabled=false")
        if self.dynamic_cp_config.enabled and not self.enabled:
            raise ValueError("dynamic_cp_config requires dyntra_lb_config.enabled=true")
        if not isinstance(self.enabled, bool):
            raise ValueError(f"dyntra_lb_config.enabled must be a bool, got {type(self.enabled).__name__}.")
        if not isinstance(self.enable_diagnostics, bool):
            raise ValueError(
                f"dyntra_lb_config.enable_diagnostics must be a bool, got {type(self.enable_diagnostics).__name__}."
            )
        if not isinstance(self.mode, str) or self.mode not in self._valid_modes:
            raise ValueError(f"dyntra_lb_config.mode must be one of {sorted(self._valid_modes)}, got {self.mode!r}.")

        for key in (
            "first_token_max_lb_wait_steps",
            "first_token_max_admissions_per_step",
            "start_step",
            "end_step",
            "long_req_block_threshold",
            "dynamic_max_step",
            "balance_interval",
            "max_pause_steps",
            "min_run_steps",
            "max_pauses_per_rank",
        ):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"dyntra_lb_config.{key} must be an int, got {type(value).__name__}.")

        for key in (
            "bubble_threshold",
            "min_throughput_gain",
            "step_latency_fixed_us",
            "step_latency_per_block_us",
        ):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"dyntra_lb_config.{key} must be a finite number, got {value!r}.")
            setattr(self, key, float(value))

        if self.start_step < 0:
            raise ValueError(f"dyntra_lb_config.start_step must be >= 0, got {self.start_step}.")
        if self.end_step < -1:
            raise ValueError(f"dyntra_lb_config.end_step must be -1 or >= 0, got {self.end_step}.")
        if self.end_step != -1 and self.end_step <= self.start_step:
            raise ValueError(
                "dyntra_lb_config.end_step must be greater than start_step when it is set, "
                f"got start_step={self.start_step}, end_step={self.end_step}."
            )
        if self.bubble_threshold <= 0:
            raise ValueError(f"dyntra_lb_config.bubble_threshold must be > 0, got {self.bubble_threshold}.")
        if self.long_req_block_threshold <= 0:
            raise ValueError(
                f"dyntra_lb_config.long_req_block_threshold must be > 0, got {self.long_req_block_threshold}."
            )
        if self.dynamic_max_step <= 0:
            raise ValueError(f"dyntra_lb_config.dynamic_max_step must be > 0, got {self.dynamic_max_step}.")
        for key in (
            "balance_interval",
            "max_pause_steps",
            "min_run_steps",
            "first_token_max_lb_wait_steps",
            "first_token_max_admissions_per_step",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"dyntra_lb_config.{key} must be > 0, got {getattr(self, key)}.")
        if self.max_pauses_per_rank < 0:
            raise ValueError("dyntra_lb_config.max_pauses_per_rank must be >= 0.")
        if not 0 <= self.min_throughput_gain < 1:
            raise ValueError("dyntra_lb_config.min_throughput_gain must be >= 0 and < 1.")
        for key in ("step_latency_fixed_us", "step_latency_per_block_us"):
            if getattr(self, key) <= 0:
                raise ValueError(f"dyntra_lb_config.{key} must be > 0, got {getattr(self, key)}.")

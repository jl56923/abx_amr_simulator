"""
Lightweight environment wrappers for debugging and visibility.

Currently includes:
- StepPrinter: prints each transition (obs/action/reward/termination/info keys)
  for short debug runs. Keep disabled for long trainings to avoid slowdowns.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional
import io
from pathlib import Path

import numpy as np
import gymnasium as gym


class StepPrinter(gym.Wrapper):
    """
    A simple wrapper that prints each env transition.

    Parameters
    - print_obs: whether to print observations (may be large)
    - obs_formatter: optional callable to format obs for printing
    - step_formatter: optional callable to format entire step (obs, action, reward, terminated, truncated, info). If provided, overrides default step formatting for custom interpretation.
    - print_info_keys: subset of info keys to print; if None prints key list only
    - max_print_steps: stop printing after this many steps (wrapper stays active)
    - prefix: string prefix for each line to make logs searchable
    - float_precision: number of decimals for floats
    - log_path: optional file path to mirror StepPrinter output
    - log_all_steps: when logging to file, capture all steps regardless of max_print_steps
    - mirror_stdout: whether to also print to stdout (default True)
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        print_obs: bool = False,
        formatter: Optional[Callable[[Any, Any, Any, Any, Any, Any], str]] = None,
        print_info_keys: Optional[Iterable[str]] = None,
        max_print_steps: Optional[int] = None,
        prefix: str = "",
        float_precision: int = 3,
        log_path: Optional[str] = None,
        log_all_steps: bool = False,
        mirror_stdout: bool = True,
    ) -> None:
        super().__init__(env)
        self._step_idx = 0
        self._print_obs = print_obs
        self._formatter = formatter
        self._print_info_keys = list(print_info_keys) if print_info_keys is not None else None
        self._max_print_steps = max_print_steps
        self._prefix = prefix
        self._float_fmt = f"{{:.{float_precision}f}}"
        self._log_all_steps = log_all_steps
        self._mirror_stdout = mirror_stdout

        self._log_file: Optional[io.TextIOBase] = None
        if log_path:
            path_obj = Path(log_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            # line buffered text file to keep logs ordered
            self._log_file = path_obj.open("a", buffering=1)

        # Track previous observation for step formatting
        self._prev_obs: Any = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_idx = 0
        self._prev_obs = obs
        self._emit(
            f"{self._prefix}reset: info_keys={list(info.keys()) if isinstance(info, dict) else []}"
        )
        if self._print_obs:
            self._emit(f"{self._prefix}obs0: {self._format_obs(obs)}")
        return obs, info

    def step(self, action):  # type: ignore[override]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_idx += 1
        if self._should_print() or self._should_log():
            # Use custom formatter if provided, otherwise use default formatting
            if self._formatter is not None:
                try:
                    step_str = self._formatter(self._prev_obs, action, reward, terminated, truncated, info)
                    self._emit(f"{self._prefix}{step_str}")
                except Exception:
                    self._emit_default_step(self._prev_obs, action, reward, terminated, truncated, info)
            else:
                self._emit_default_step(self._prev_obs, action, reward, terminated, truncated, info)
        # Update previous observation to the next observation
        self._prev_obs = obs
        return obs, reward, terminated, truncated, info

    def log_header(self, message: str) -> None:
        """
        Emit a custom header/divider line to the log.
        Useful for marking evaluation cycles or other phases.
        """
        self._emit(f"{self._prefix}{message}")

    def _emit_default_step(self, prev_obs, action, reward, terminated, truncated, info) -> None:
        """Default step formatting when no formatter is provided."""
        obs_str = self._format_obs(prev_obs)
        action_str = self._format_action(action)
        r_str = self._format_float(reward)
        term_str = f"T={bool(terminated)}"
        trunc_str = f"Tr={bool(truncated)}"
        info_desc = self._format_info(info)
        self._emit(
            f"{self._prefix}step {self._step_idx}: saw={obs_str} a={action_str} r={r_str} {term_str} {trunc_str} {info_desc}"
        )


    # ----- helpers -----
    def _should_print(self) -> bool:
        if self._max_print_steps is None:
            return True
        return self._step_idx <= self._max_print_steps

    def _should_log(self) -> bool:
        if self._log_file is None:
            return False
        if self._log_all_steps:
            return True
        return self._should_print()

    def _emit(self, line: str) -> None:
        if self._mirror_stdout and self._should_print():
            print(line)
        if self._should_log():
            try:
                self._log_file.write(line + "\n")
            except Exception:
                # fail silently; logging should not break env stepping
                pass

    def _format_float(self, x: Any) -> str:
        try:
            return self._float_fmt.format(float(x))
        except Exception:
            return str(x)

    def _format_obs(self, obs: Any) -> str:
        if isinstance(obs, np.ndarray):
            shape = obs.shape
            if obs.size <= 16:
                return f"ndarray{shape} {np.array2string(obs, precision=3, floatmode='fixed')}"
            head = np.array2string(obs.ravel()[:8], precision=3, floatmode='fixed')
            tail = np.array2string(obs.ravel()[-4:], precision=3, floatmode='fixed')
            return f"ndarray{shape} head={head} ... tail={tail}"
        return str(obs)

    def _format_action(self, action: Any) -> str:
        # Try to convert to np array for consistent display
        try:
            arr = np.asarray(action)
        except Exception:
            return str(action)

        # If env exposes mapping, print compact counts per antibiotic
        mapping = getattr(self.env, "get_action_to_antibiotic_mapping", None)
        if callable(mapping):
            try:
                idx_to_abx = mapping()
                # flatten choices
                flat = arr.flatten()
                # fallback when flat is scalar
                if flat.ndim == 0:
                    flat = np.array([int(flat)])
                names = [idx_to_abx.get(int(i), str(int(i))) for i in flat]
                # compress as counts per name for readability
                counts = {}
                for n in names:
                    counts[n] = counts.get(n, 0) + 1
                return f"{flat.tolist()} {counts}"
            except Exception:
                pass

        return arr.tolist().__str__()

    def _format_info(self, info: Any) -> str:
        if not isinstance(info, dict):
            return "info=<non-dict>"
        if self._print_info_keys is None:
            return f"info_keys={list(info.keys())}"
        subset = {k: info.get(k, None) for k in self._print_info_keys}
        # Format floats nicely
        for k, v in subset.items():
            if isinstance(v, float):
                subset[k] = float(self._format_float(v))
        return f"info={subset}"

    # Forward unknown attribute access to the wrapped env so tools expecting
    # original env attributes (e.g., antibiotic parameters) still work.
    def __getattr__(self, name: str):  # noqa: B004
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.env, name)

    def close(self):  # type: ignore[override]
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
        return super().close()

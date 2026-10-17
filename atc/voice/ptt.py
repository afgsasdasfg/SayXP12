"""Push-to-talk from a global keyboard hotkey (pynput) and/or an X-Plane command.

For a joystick button: in X-Plane bind the button to "Contact ATC"
(sim/operation/contact_atc); sim_link reports its press/release here.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

log = logging.getLogger("ptt")


def _parse_key(name: str):
    from pynput import keyboard
    name = name.strip().lower()
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    try:
        return getattr(keyboard.Key, name)
    except AttributeError as e:
        raise ValueError(f"unknown PTT key {name!r} (use e.g. 'f12', 'ctrl_r', 'scroll_lock', 'caps_lock')") from e


class PTT:
    """Calls on_change(True) on press and on_change(False) on release (from any thread)."""

    def __init__(self, on_change: Callable[[bool], None], key: str | None = "ctrl_r"):
        self.on_change = on_change
        self.key = _parse_key(key) if key else None
        self.pressed = False
        self._sources: set[str] = set()
        self._lock = threading.Lock()
        self._listener = None

    def _set(self, source: str, down: bool) -> None:
        with self._lock:
            was = bool(self._sources)
            if down:
                self._sources.add(source)
            else:
                self._sources.discard(source)
            now = bool(self._sources)
        if now != was:
            self.pressed = now
            log.debug("PTT %s (%s)", "down" if now else "up", source)
            self.on_change(now)

    def start(self) -> None:
        if self.key is None:
            return
        from pynput import keyboard

        def on_press(k):
            if k == self.key:
                self._set("key", True)

        def on_release(k):
            if k == self.key:
                self._set("key", False)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def sim_command(self, name: str, active: bool) -> None:
        """Hook for SimLink(on_command=...)."""
        self._set("sim", active)

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()

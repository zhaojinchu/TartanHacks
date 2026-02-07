#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True, slots=True)
class KeyAction:
    key: str
    label: str
    command: str | None = None


KEYMAP: list[KeyAction] = [
    KeyAction("1", "Open bin 0", "O0"),
    KeyAction("q", "Close bin 0", "C0"),
    KeyAction("2", "Open bin 1", "O1"),
    KeyAction("w", "Close bin 1", "C1"),
    KeyAction("3", "Open bin 2", "O2"),
    KeyAction("e", "Close bin 2", "C2"),
    KeyAction("a", "Open all bins"),
    KeyAction("z", "Close all bins"),
    KeyAction("s", "Show latest bin status"),
    KeyAction("h", "Show help"),
    KeyAction("x", "Exit"),
]


class RawKeyboard:
    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old_settings: list[Any] | None = None

    def __enter__(self) -> "RawKeyboard":
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


def _http_json(method: str, url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, method=method, data=data, headers=headers)
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else None


def send_command(api_base: str, command: str, timeout: float) -> tuple[bool, str]:
    url = f"{api_base}/api/arduino/command"
    try:
        payload = _http_json("POST", url, {"command": command}, timeout)
        if not isinstance(payload, dict) or not payload.get("ok"):
            return False, f"Command `{command}` failed: {payload}"
        return True, f"Sent `{command}`"
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return False, f"HTTP {exc.code}: {detail or exc.reason}"
    except Exception as exc:
        return False, f"Request error: {exc}"


def get_bins(api_base: str, timeout: float) -> tuple[bool, list[dict[str, Any]] | str]:
    url = f"{api_base}/api/bins"
    try:
        payload = _http_json("GET", url, None, timeout)
        if not isinstance(payload, list):
            return False, f"Unexpected /api/bins payload: {payload}"
        return True, payload
    except Exception as exc:
        return False, f"Failed to fetch bins: {exc}"


def print_help() -> None:
    print("\nManual Bin Control Keys")
    for action in KEYMAP:
        cmd = f" -> {action.command}" if action.command else ""
        print(f"  {action.key}: {action.label}{cmd}")
    print("")


def print_status(api_base: str, timeout: float) -> None:
    ok, result = get_bins(api_base, timeout)
    if not ok:
        print(f"[status] {result}")
        return
    bins = result  # type: ignore[assignment]
    print("\nLatest Bin Status")
    for idx, row in enumerate(bins):
        fullness = row.get("fullness_percent")
        fullness_text = "N/A" if fullness is None else f"{float(fullness):.1f}%"
        print(
            f"  [{idx}] {row.get('bin_id')} | {row.get('status')} | "
            f"fullness={fullness_text} | distance={row.get('distance_cm')}"
        )
    print("")


def key_to_action(ch: str) -> KeyAction | None:
    for action in KEYMAP:
        if action.key == ch:
            return action
    return None


def execute_special(api_base: str, key: str, timeout: float) -> tuple[bool, str]:
    if key == "a":
        outcomes = [send_command(api_base, cmd, timeout) for cmd in ("O0", "O1", "O2")]
        if all(ok for ok, _ in outcomes):
            return True, "Sent open-all: O0 O1 O2"
        return False, "; ".join(msg for _, msg in outcomes)
    if key == "z":
        outcomes = [send_command(api_base, cmd, timeout) for cmd in ("C0", "C1", "C2")]
        if all(ok for ok, _ in outcomes):
            return True, "Sent close-all: C0 C1 C2"
        return False, "; ".join(msg for _, msg in outcomes)
    return False, f"Unhandled special key `{key}`"


def run(args: argparse.Namespace) -> int:
    if not sys.stdin.isatty():
        print("This CLI needs an interactive TTY.", file=sys.stderr)
        return 1

    print(f"Manual bin control connected to {args.api_base}")
    print("Press `h` for help, `x` to exit.")
    print_help()
    print_status(args.api_base, args.timeout)

    next_status = time.monotonic() + args.status_interval
    with RawKeyboard():
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable:
                ch = sys.stdin.read(1).lower()
                action = key_to_action(ch)
                if action is None:
                    continue

                if action.key == "x":
                    print("\nExiting manual control.")
                    return 0
                if action.key == "h":
                    print_help()
                    continue
                if action.key == "s":
                    print_status(args.api_base, args.timeout)
                    next_status = time.monotonic() + args.status_interval
                    continue

                if action.command is not None:
                    ok, msg = send_command(args.api_base, action.command, args.timeout)
                else:
                    ok, msg = execute_special(args.api_base, action.key, args.timeout)
                prefix = "[ok]" if ok else "[error]"
                print(f"\n{prefix} {msg}")

            now = time.monotonic()
            if args.status_interval > 0 and now >= next_status:
                print_status(args.api_base, args.timeout)
                next_status = now + args.status_interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive keyboard CLI for manual bin open/close.")
    parser.add_argument("--api-base", default="http://localhost:8000", help="Backend API base URL")
    parser.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout seconds")
    parser.add_argument(
        "--status-interval",
        type=float,
        default=5.0,
        help="Seconds between automatic status snapshots (0 disables auto snapshots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()

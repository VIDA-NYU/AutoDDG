from __future__ import annotations

from datetime import datetime

from beartype import beartype


@beartype
def log_print(
    text: str | None,
    symbol_begin: str = "=",
    symbol_end: str | None = None,
    num: int = 30,
) -> None:
    print(symbol_begin * num)
    if text is not None:
        print(text)
    if symbol_end:
        print(symbol_end * num)


@beartype
def get_log_time() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")

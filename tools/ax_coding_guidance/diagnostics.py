"""Diagnostics shared by the AX guidance checker."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Diagnostic:
    rule: str
    path: str
    line: int
    column: int
    symbol: str
    message: str

    def format(self) -> str:
        subject = f" [{self.symbol}]" if self.symbol else ""
        return f"{self.path}:{self.line}:{self.column}: {self.rule}{subject} {self.message}"

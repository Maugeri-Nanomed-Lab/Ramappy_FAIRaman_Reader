"""Command-line entry point."""

from __future__ import annotations

import argparse

import flet as ft

from .app import FairamanViewerApp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Viewer Flet per file FAIRaman .h5")
    parser.add_argument("file", nargs="?", help="file .h5 da aprire all'avvio")
    args = parser.parse_args(argv)

    def start(page: ft.Page):
        FairamanViewerApp(page, initial_file=args.file)

    ft.run(start)
    return 0

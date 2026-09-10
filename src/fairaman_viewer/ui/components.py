"""Small reusable Flet control factories."""

import flet as ft

def _dropdown(value: str, options: list[str], *, label: str = "", width: int | None = None,
              on_select=None) -> ft.Dropdown:
    return ft.Dropdown(
        value=value,
        label=label or None,
        width=width,
        dense=True,
        options=[ft.DropdownOption(key=item, text=item) for item in options],
        on_select=on_select,
    )


def _num_field(value: str = "", *, label: str = "", width: int = 88, on_change=None) -> ft.TextField:
    return ft.TextField(
        value=value,
        label=label or None,
        width=width,
        dense=True,
        on_change=on_change,
    )


def _card(title: str, content: ft.Control, *, expand: bool = False) -> ft.Container:
    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(title, weight=ft.FontWeight.BOLD, size=14),
                ft.Divider(height=1),
                content,
            ],
            expand=expand,
            spacing=8,
        ),
        padding=10,
        expand=expand,
    )


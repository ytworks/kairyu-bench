"""Explicitly installed experiment plugin; importing the package does not register it."""


def register() -> None:
    from .loader import register as install
    install()

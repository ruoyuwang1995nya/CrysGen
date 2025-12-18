"""Lightweight registry for reusable tool functions."""

from __future__ import annotations

from typing import Callable, Dict


class Tools:
	_TOOLS: Dict[str, Callable] = {}

	@classmethod
	def register(cls, name: str):
		"""Decorator to register a callable under ``name``."""

		def decorator(func: Callable) -> Callable:
			cls._TOOLS[name] = func
			return func

		return decorator

	@classmethod
	def get(cls, name: str) -> Callable:
		try:
			return cls._TOOLS[name]
		except KeyError as exc:
			raise KeyError(f"Tool '{name}' is not registered") from exc

	@classmethod
	def all(cls) -> Dict[str, Callable]:
		return dict(cls._TOOLS)

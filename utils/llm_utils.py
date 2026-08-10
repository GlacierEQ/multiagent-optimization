"""Compatibility aliases for legacy integration imports."""

from adapters.llm import call_model


def call_llm(prompt: str, *args, **kwargs):
    return call_model(prompt, *args, **kwargs)

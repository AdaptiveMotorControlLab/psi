"""Programmable Semantic Interventions with lazy public imports."""
from importlib import import_module

__version__ = "0.1.0"
__all__ = [
    "PSI",
    "ProgrammableSemanticIntervention",
    "Prompts",
    "Program",
    "Values",
    "Slots",
    "At",
    "Gated",
    "Table",
    "scan_layers",
    "presets",
]


def __getattr__(
    name,
):
    """Resolve public exports on first access without importing training eagerly."""
    if name not in __all__:
        raise AttributeError(name)

    module_name = {
        "PSI": "api",
        "ProgrammableSemanticIntervention": "psi",
        "scan_layers": "api",
        "Prompts": "prompts",
        "presets": "presets",
    }.get(name, "program")
    module = import_module(f"{__name__}.{module_name}")
    value = module if name == "presets" else getattr(module, name)
    globals()[name] = value

    return value

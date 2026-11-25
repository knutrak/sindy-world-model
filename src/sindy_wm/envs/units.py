"""Our own unit types (configs/units/<NAME>.json, SMAClite's unit format) next to the built-in ones.

`register()` loads them into SMAClite's cache of custom types, so scenario files can use them by name from any
working directory. Names are upper case, like the built-in ones.
"""

from functools import cache

from smaclite.env.units.unit_type import CustomUnitType, StandardUnit, UnitStats

from sindy_wm.paths import UNIT_DIR


@cache
def register() -> tuple[str, ...]:
    """Load every unit in configs/units/ (once); returns their names."""
    names = tuple(sorted(p.stem for p in UNIT_DIR.glob("*.json")))
    for name in names:
        if name in StandardUnit.__members__:
            raise ValueError(f"{UNIT_DIR / name}.json has the name of a built-in unit")
        CustomUnitType.from_file(name, str(UNIT_DIR))
    return names


def unit_name(unit_type) -> str:
    """A unit type's name, e.g. "MARINE". Custom types have no `name` of their own."""
    if isinstance(unit_type, CustomUnitType):
        return unit_type.stats.name
    return getattr(unit_type, "name", str(unit_type))


def stats(name: str) -> UnitStats:
    """SMAClite's statistics of a built-in or custom unit type."""
    if name in StandardUnit.__members__:
        return StandardUnit[name].stats
    if name not in register():
        raise KeyError(name)
    return CustomUnitType.from_file(name, str(UNIT_DIR)).stats

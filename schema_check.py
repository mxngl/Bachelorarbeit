#!/usr/bin/env python3
"""
schema_check.py — Messskript: zählt Schemafehler (ifcopenshell.validate) in einem Original und einem
Ergebnis und zeigt, welche Fehler neu hinzugekommen sind. Messinstrument für Kap. 4.6 und 7,
kein Teil der Pipeline (Gültigkeit im Betrieb prüfen der Structural und der Semantic–Geometric Validator).

KI-Hinweis: mit Unterstützung von Claude (Anthropic) erstellt, Dokumentation in Anhang F.

Nutzung:  python schema_check.py test-models\\AC20-FZK-Haus.ifc ergebnis_loeschen.ifc
"""

from __future__ import annotations

import sys

import ifcopenshell
import ifcopenshell.validate


def messages(path: str) -> list[str]:
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(ifcopenshell.open(path), logger)
    return [f"{st.get('instance', '')} | {st.get('attribute', '')} | {st.get('message', '')}"
            for st in logger.statements]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    before, after = messages(sys.argv[1]), messages(sys.argv[2])
    known = set(before)
    new = [m for m in after if m not in known]
    print(f"Schemafehler Original:  {len(before)}")
    print(f"Schemafehler Ergebnis:  {len(after)}")
    print(f"davon neu im Ergebnis:  {len(new)}")
    for m in new[:20]:
        print("  +", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())

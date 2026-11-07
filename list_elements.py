#!/usr/bin/env python3
"""
list_elements.py — listet Geschosse und Bauteile eines IFC-Modells (Klasse, Name, Geschoss, GlobalId, Wirt).
Zweck: Namen und GlobalIds für Benchmark-Aufgaben und Ground Truth nachschlagen.

Nutzung:  python list_elements.py test-models/AC20-FZK-Haus.ifc > fzk_elemente.txt
"""

import sys
from collections import Counter

import ifcopenshell

import llm_stations as L
import neo4j2ifc as r
import structural_validator as sv


def main(path: str) -> None:
    model = ifcopenshell.open(path)
    schema = sv.schema_for(model)
    view = L.GraphView(r.snapshot_from_ifc(model, path), schema)
    print("# Geschosse")
    for s in view.list_storeys():
        print(f"{s['guid']}  {s['name']}  (Höhe {s['elevation_m']} m)")
    els = sorted((sid for sid in view.nodes if view.is_element(sid)),
                 key=lambda s: (view.cls(s), str(view.nodes[s].get("Name") or ""), s))
    names = Counter((view.cls(s), view.nodes[s].get("Name")) for s in els)
    print(f"\n# Bauteile ({len(els)}); '!' = Name+Klasse nicht eindeutig")
    for sid in els:
        b = view.brief(sid)
        h = view.host_of(sid)
        dup = "!" if names[(b["class"], b["name"])] > 1 else " "
        host = f"  Wirt: {view.nodes[h].get('Name')}" if h is not None else ""
        print(f"{dup} {b['class']:<22} {str(b['name']):<34} {str(b['storey']):<14} {b['guid']}{host}")


if __name__ == "__main__":
    main(sys.argv[1])

#!/usr/bin/env python3
"""
test_roundtrip.py — prüft neo4j2ifc.py ohne Neo4j-Verbindung.

Der Zielgraph wird im Speicher aus der IFC-Datei gebildet und gezielt verändert (so wie es später
der Executor in Neo4j tut). Danach läuft derselbe Weg wie im echten Betrieb: Differenz, Patch,
Schreiben, Wiedereinlesen, Round-Trip-Prüfung.

KI-Hinweis: mit Unterstützung von Claude (Anthropic) erstellt, Dokumentation in Anhang F.

Nutzung:  python test_roundtrip.py AC20-FZK-Haus.ifc
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile

import ifcopenshell
import ifcopenshell.validate

import neo4j2ifc as r


def run_case(path, mutate):
    """Führt einen Fall aus und liefert (diff, report, patched, rest, target)."""
    source = os.path.basename(path)
    model = ifcopenshell.open(path)
    base = r.snapshot_from_ifc(model, source)
    target = copy.deepcopy(base)
    mutate(model, base, target)
    diff = r.diff_snapshots(base, target)
    rep = r.apply_diff(model, base, target, diff)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.ifc")
        model.write(out)
        patched = ifcopenshell.open(out)           # inkl. Schreiben und Wiedereinlesen
        rest = r.verify_roundtrip(patched, target, rep, source)
    return diff, rep, patched, rest, target


def drop_node(target, sid):
    """Wie DETACH DELETE: Knoten und alle Kanten daran entfernen."""
    del target.nodes[sid]
    for key in [k for k, d in target.edges.items() if k[0] == sid or d == sid]:
        del target.edges[key]


def unreachable(model, target):
    """Knoten, die von keinem IfcRoot-Knoten aus erreichbar sind (verwaist)."""
    roots = [sid for sid in target.nodes if _is_root(model, sid)]
    out = {}
    for (s, _, _, _), d in target.edges.items():
        out.setdefault(s, set()).add(d)
    seen, stack = set(roots), list(roots)
    while stack:
        for d in out.get(stack.pop(), ()):
            if d not in seen:
                seen.add(d)
                stack.append(d)
    return set(target.nodes) - seen


def _is_root(model, sid):
    try:
        return model.by_id(sid).is_a("IfcRoot")
    except RuntimeError:
        return False


def edge(snap, src, name):
    return snap.edges[(src, name, None, None)]


def case_identity(path):
    def mutate(model, base, target):
        pass
    diff, rep, patched, rest, _ = run_case(path, mutate)
    assert diff.is_empty(), diff.summary()
    assert rest.is_empty(), rest.summary()
    assert len(list(patched)) == len(list(ifcopenshell.open(path)))
    return "unverändert: Differenz leer, Datei gleich groß wie das Original"


def case_move(path):
    state = {}

    def mutate(model, base, target):
        wall = model.by_type("IfcWallStandardCase")[0]
        lp = edge(target, wall.id(), "ObjectPlacement")
        ax = edge(target, lp, "RelativePlacement")
        pt = edge(target, ax, "Location")
        x, y, z = target.nodes[pt]["Coordinates"]
        target.nodes[pt]["Coordinates"] = [x + 1.0, y, z]
        state.update(pt=pt, new=(x + 1.0, y, z))

    diff, rep, patched, rest, _ = run_case(path, mutate)
    assert len(diff.changed_nodes) == 1 and not diff.added_nodes and not diff.removed_nodes
    assert rest.is_empty(), rest.summary()
    assert tuple(patched.by_id(state["pt"]).Coordinates) == state["new"]
    return "Wand verschoben: nur ein CartesianPoint geändert"


def case_property_types(path):
    state = {}

    def mutate(model, base, target):
        sv = next(n for n, p in target.nodes.items()
                  if p["ifc_class"] == "IfcPropertySingleValue" and p.get("NominalValue_type") == "IfcLabel")
        q = next(n for n, p in target.nodes.items() if p["ifc_class"] == "IfcQuantityLength")
        target.nodes[sv]["NominalValue"] = "GEAENDERT"
        old = target.nodes[q]["LengthValue"]
        target.nodes[q]["LengthValue"] = old + 0.5
        state.update(sv=sv, q=q, q_new=old + 0.5)

    diff, rep, patched, rest, _ = run_case(path, mutate)
    assert rest.is_empty(), rest.summary()
    nv = patched.by_id(state["sv"]).NominalValue
    assert nv.is_a() == "IfcLabel" and nv.wrappedValue == "GEAENDERT", nv
    assert patched.by_id(state["q"]).LengthValue == state["q_new"]
    return "Properties geändert, getypter Wert bleibt IfcLabel"


def case_add_and_retarget(path):
    state = {}

    def mutate(model, base, target):
        new_id = max(target.nodes) + 1000
        target.nodes[new_id] = {"step_id": new_id, "ifc_class": "IfcCartesianPoint",
                                "Coordinates": [1.0, 2.0, 3.0]}
        wall = model.by_type("IfcWallStandardCase")[1]
        lp = edge(target, wall.id(), "ObjectPlacement")
        ax = edge(target, lp, "RelativePlacement")
        old_pt = edge(target, ax, "Location")
        target.edges[(ax, "Location", None, None)] = new_id
        drop_node(target, old_pt)
        state.update(new_id=new_id, ax=ax, old_pt=old_pt)

    diff, rep, patched, rest, _ = run_case(path, mutate)
    assert diff.added_nodes == [state["new_id"]] and diff.removed_nodes == [state["old_pt"]]
    assert len(diff.changed_attrs) == 1
    assert rest.is_empty(), rest.summary()
    pt = patched.by_id(state["ax"]).Location
    assert tuple(pt.Coordinates) == (1.0, 2.0, 3.0)
    assert pt.id() == rep.added_ids[state["new_id"]]
    return "neuer Punkt angelegt, Location umgehängt, alter Punkt gelöscht (Datei vergibt eigene ID)"


def case_delete_clean(path):
    """Löscht eine IfcAnnotation samt Placement-Kette (wie ein vollständiger Executor-Lauf)."""
    state = {}

    def mutate(model, base, target):
        ann = next(a for a in model.by_type("IfcAnnotation") if a.Representation is not None)
        state["ann"] = ann.id()
        state["shapes_before"] = len(model.by_type("IfcProductDefinitionShape"))
        drop_node(target, ann.id())
        for sid in sorted(unreachable(model, target)):       # verwaiste Placements usw. mitlöschen
            drop_node(target, sid)

    diff, rep, patched, rest, _ = run_case(path, mutate)
    assert state["ann"] in diff.removed_nodes and len(diff.removed_nodes) > 1
    assert rest.is_empty(), rest.summary()
    assert rep.removed_geometry >= 1, "Geometrie wurde nicht aufgeräumt"
    assert len(patched.by_type("IfcProductDefinitionShape")) == state["shapes_before"] - 1
    return (f"Annotation gelöscht: {len(diff.removed_nodes)} Knoten, "
            f"{rep.removed_geometry} verwaiste Geometrie-Instanzen entfernt, "
            f"{len(diff.changed_attrs)} Referenzattribute angepasst")


def schema_messages(model):
    """Schemafehler laut IfcOpenShell. Liegt bewusst im Test: Gültigkeit prüfen die Validatoren."""
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(model, logger)
    return [f"{st.get('instance', '')} | {st.get('attribute', '')} | {st.get('message', '')}"
            for st in logger.statements]


def case_cascade_evidence(path):
    """Beleg für Kapitel 4.6: Fenster nur per DETACH DELETE, ohne Kaskadenregeln."""
    state = {}

    def mutate(model, base, target):
        win = model.by_type("IfcWindow")[0]
        state["before"] = schema_messages(model)
        drop_node(target, win.id())

    diff, rep, patched, rest, _ = run_case(path, mutate)
    new = [m for m in schema_messages(patched) if m not in set(state["before"])]
    assert new, "Ohne Kaskade müssten neue Schemafehler entstehen"
    empty_lists = sum("RelatedObjects" in m for m in new)
    return (f"{len(new)} neue Schemafehler gegenüber dem Original "
            f"({empty_lists}x leeres RelatedObjects, {len(new) - empty_lists}x Pflichtattribut leer)")


def case_delete_incomplete(path):
    """Fenster nur per DETACH DELETE, ohne Kaskade. Abweichungen zwischen Datei und Graph
    sind erwartet und betreffen genau die verwaisten Knoten (bewertet werden sie im Validator)."""
    state = {}

    def mutate(model, base, target):
        win = model.by_type("IfcWindow")[0]
        state["win"] = win.id()
        drop_node(target, win.id())
        state["orphans"] = unreachable(model, target)

    diff, rep, patched, rest, _ = run_case(path, mutate)
    assert diff.removed_nodes == [state["win"]]
    assert state["orphans"], "Test sinnlos: keine Waisen entstanden"
    assert set(rest.added_nodes) == state["orphans"], "Abweichung nicht genau die Waisen"
    assert not rest.changed_nodes and not rest.removed_nodes
    assert {src for src, _ in rest.changed_attrs} <= state["orphans"], "Abweichung außerhalb der Waisen"
    return (f"Fenster ohne Kaskade gelöscht: Datei und Graph weichen genau in den "
            f"{len(state['orphans'])} verwaisten Knoten ab")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "AC20-FZK-Haus.ifc"
    cases = [case_identity, case_move, case_property_types, case_add_and_retarget,
             case_delete_clean, case_delete_incomplete, case_cascade_evidence]
    failed = 0
    for fn in cases:
        try:
            print(f"OK   {fn.__name__}: {fn(path)}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(cases) - failed}/{len(cases)} Fälle bestanden")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

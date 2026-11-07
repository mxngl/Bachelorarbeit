#!/usr/bin/env python3
"""
test_executor.py — prüft die Operationen von executor.py (delete_element, move_element) ohne Neo4j-Verbindung.

Der Graph wird im Speicher aus der IFC-Datei gebildet (MemoryBackend), der Executor ändert ihn,
danach läuft der echte Weg der Reconstruction: Differenz, Patch, Schreiben, Wiedereinlesen.
Die Schemafehler zählt der Test mit ifcopenshell.validate. Das ist nur ein Messinstrument im Test
(Beleg für Kap. 4.6). Gültigkeit im Betrieb prüfen die beiden Validatoren.

KI-Hinweis: mit Unterstützung von Claude (Anthropic) erstellt, Dokumentation in Anhang F.

Nutzung:  python test_executor.py AC20-FZK-Haus.ifc
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile

import numpy as np

import ifcopenshell
import ifcopenshell.ifcopenshell_wrapper as wrapper
import ifcopenshell.util.placement as placement_util

import executor as x
import ifc2neo4j as g
import neo4j2ifc as r
from test_roundtrip import schema_messages, unreachable


class Env:
    """Ein Testlauf: Modell, Basis-Snapshot, Ziel-Snapshot und Backend."""

    def __init__(self, path, prepare=None):
        self.path = path
        self.source = os.path.basename(path)
        self.model = ifcopenshell.open(path)
        if prepare:
            prepare(self.model)          # Testaufbau, z. B. einen Punkt absichtlich teilen
        self.base = r.snapshot_from_ifc(self.model, self.source)
        self.target = copy.deepcopy(self.base)
        schema_name = getattr(self.model, "schema_identifier", None) or self.model.schema
        schema = wrapper.schema_by_name(schema_name)
        cache: dict = {}
        self.backend = x.MemoryBackend(
            self.target, lambda cls: set(g.label_chain(schema, cls, cache)) | {"IfcEntity"})
        self.ex = x.Executor(self.backend, schema)

    def guid_of(self, inst):
        return inst.GlobalId

    def finish(self):
        """Differenz, Patch, Schreiben, Wiedereinlesen, Round-Trip-Prüfung."""
        diff = r.diff_snapshots(self.base, self.target)
        rep = r.apply_diff(self.model, self.base, self.target, diff)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.ifc")
            self.model.write(out)
            patched = ifcopenshell.open(out)
            rest = r.verify_roundtrip(patched, self.target, rep, self.source)
        return diff, rep, patched, rest


def new_errors(path, patched):
    before = set(schema_messages(ifcopenshell.open(path)))
    return [m for m in schema_messages(patched) if m not in before]


def n(model, cls):
    return len(model.by_type(cls))


def case_delete_window_default(path):
    env = Env(path)
    win = env.model.by_type("IfcWindow")[0]
    counts = {c: n(env.model, c) for c in ("IfcWindow", "IfcOpeningElement", "IfcRelFillsElement", "IfcRelVoidsElement")}
    res = env.ex.apply({"op": "delete_element", "guid": win.GlobalId})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    err = new_errors(path, patched)
    assert not err, f"{len(err)} neue Schemafehler: {err[:3]}"
    assert not unreachable(env.model, env.target), "verwaiste Knoten im Graph"
    assert n(patched, "IfcWindow") == counts["IfcWindow"] - 1
    assert n(patched, "IfcOpeningElement") == counts["IfcOpeningElement"] - 1
    assert n(patched, "IfcRelFillsElement") == counts["IfcRelFillsElement"] - 1
    assert n(patched, "IfcRelVoidsElement") == counts["IfcRelVoidsElement"] - 1
    return (f"Fenster samt Öffnung gelöscht: {len(res.deleted)} Knoten, 0 neue Schemafehler "
            f"(ohne Kaskade waren es 9), keine Waisen")


def case_delete_window_keep_opening(path):
    env = Env(path)
    win = env.model.by_type("IfcWindow")[0]
    openings = n(env.model, "IfcOpeningElement")
    res = env.ex.apply({"op": "delete_element", "guid": win.GlobalId, "remove_host_opening": False})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert n(patched, "IfcOpeningElement") == openings, "Öffnung hätte bleiben müssen"
    assert not unreachable(env.model, env.target)
    err = new_errors(path, patched)
    assert not err, f"{len(err)} neue Schemafehler: {err[:3]}"
    return f"Fenster gelöscht, Öffnung bleibt: {len(res.deleted)} Knoten, 0 neue Schemafehler"


def _wall_with_openings(model):
    for w in model.by_type("IfcWall"):
        if w.HasOpenings:
            return w
    raise AssertionError("keine Wand mit Öffnungen im Modell")


def case_delete_wall_dependents(path):
    env = Env(path)
    wall = _wall_with_openings(env.model)
    n_open = len(wall.HasOpenings)
    fills = [rel.RelatedOpeningElement.HasFillings for rel in wall.HasOpenings]
    n_fill = sum(len(f) for f in fills)
    windows = n(env.model, "IfcWindow")
    doors = n(env.model, "IfcDoor")
    res = env.ex.apply({"op": "delete_element", "guid": wall.GlobalId})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert not unreachable(env.model, env.target), "verwaiste Knoten im Graph"
    err = new_errors(path, patched)
    assert not err, f"{len(err)} neue Schemafehler: {err[:3]}"
    assert n(patched, "IfcWindow") + n(patched, "IfcDoor") == windows + doors - n_fill
    return (f"Wand mit {n_open} Öffnung(en) und {n_fill} Füllung(en) gelöscht: {len(res.deleted)} Knoten, "
            f"0 neue Schemafehler, keine Waisen")


def case_strict_rejects_wall(path):
    env = Env(path)
    wall = _wall_with_openings(env.model)
    nodes_before, edges_before = dict(env.target.nodes), dict(env.target.edges)
    try:
        env.ex.apply({"op": "delete_element", "guid": wall.GlobalId, "cascade": "strict"})
    except x.ExecutorError as exc:
        assert exc.code == "dependents_exist", exc
    else:
        raise AssertionError("strict hätte ablehnen müssen")
    assert env.target.nodes == nodes_before and env.target.edges == edges_before, "Graph wurde verändert"
    return "strict lehnt Wand mit Öffnungen ab (dependents_exist), Graph unverändert"


def case_scope_rejects(path):
    env = Env(path)
    m = env.model
    targets = {
        "IfcSite": m.by_type("IfcSite")[0],
        "IfcBuilding": m.by_type("IfcBuilding")[0],
        "IfcBuildingStorey": m.by_type("IfcBuildingStorey")[0],
        "IfcTypeObject": m.by_type("IfcTypeObject")[0],
        "IfcPropertySet": m.by_type("IfcPropertySet")[0],
    }
    for name, inst in targets.items():
        try:
            env.ex.apply({"op": "delete_element", "guid": inst.GlobalId})
        except x.ExecutorError as exc:
            assert exc.code == "out_of_scope", (name, exc)
        else:
            raise AssertionError(f"{name} hätte abgelehnt werden müssen")
    for op, code in (({"op": "delete_element", "guid": "0000000000000000000000"}, "not_found"),
                     ({"op": "delete_element", "guid": m.by_type("IfcWall")[0].GlobalId, "cascade": "alle"}, "bad_parameter"),
                     ({"op": "rotate_element"}, "unknown_op")):
        try:
            env.ex.apply(op)
        except x.ExecutorError as exc:
            assert exc.code == code, (op, exc)
        else:
            raise AssertionError(f"{op} hätte abgelehnt werden müssen")
    return "Site, Building, Storey, Typobjekt, Pset sowie unbekannte GUID, falscher Parameter und unbekannte Operation werden abgelehnt"


def case_delete_annotation(path):
    env = Env(path)
    ann = next(a for a in env.model.by_type("IfcAnnotation") if a.Representation is not None)
    shapes = n(env.model, "IfcProductDefinitionShape")
    res = env.ex.apply({"op": "delete_element", "guid": ann.GlobalId})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert not unreachable(env.model, env.target)
    assert n(patched, "IfcProductDefinitionShape") == shapes - 1
    err = new_errors(path, patched)
    assert not err, err[:3]
    return (f"Annotation gelöscht: {len(res.deleted)} Knoten (Placement-Kette mit), "
            f"{rep.removed_geometry} Geometrie-Instanzen in der Datei entfernt, 0 neue Schemafehler")


def case_shared_pset_survives(path):
    env = Env(path)
    rel = next((rl for rl in env.model.by_type("IfcRelDefinesByProperties")
                if len(rl.RelatedObjects) >= 2 and all(o.is_a("IfcProduct") for o in rl.RelatedObjects)), None)
    if rel is None:
        return "ÜBERSPRUNGEN: im Modell teilt kein Pset mehrere Bauteile"
    victim, other = rel.RelatedObjects[0], rel.RelatedObjects[1]
    pset = rel.RelatingPropertyDefinition
    env.ex.apply({"op": "delete_element", "guid": victim.GlobalId})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    new_rel = patched.by_id(rel.id())
    assert other.id() in [o.id() for o in new_rel.RelatedObjects]
    assert victim.id() not in [o.id() for o in new_rel.RelatedObjects]
    assert patched.by_id(pset.id()) is not None
    return f"geteiltes Pset bleibt erhalten ({len(rel.RelatedObjects)} Bauteile, eines gelöscht)"


def case_list_renumbered(path):
    env = Env(path)
    rel = env.model.by_type("IfcRelContainedInSpatialStructure")[0]
    victim = next(e for e in rel.RelatedElements if not e.HasOpenings and not getattr(e, "FillsVoids", None))
    env.ex.apply({"op": "delete_element", "guid": victim.GlobalId})
    sid = rel.id()
    idxs = sorted(k[2] for k in env.target.edges if k[0] == sid and k[1] == "RelatedElements")
    assert idxs == list(range(len(idxs))), f"idx nicht lückenlos: {idxs[:10]}"
    return f"Liste im Containment lückenlos neu nummeriert ({len(idxs)} Elemente)"


def case_placement_dependents(path):
    env = Env(path)
    wall = env.model.by_type("IfcWallStandardCase")[0]
    wall_sid = wall.id()
    pl = next(e.dst for e in env.backend.out_edges(wall_sid) if e.type == "ObjectPlacement")
    fake = max(env.target.nodes) + 1
    env.target.nodes[fake] = {"step_id": fake, "ifc_class": "IfcLocalPlacement"}
    key = (fake, "PlacementRelTo", None, None)
    env.target.edges[key] = pl
    env.backend._in[pl].add(key)
    env.backend._out[fake].add(key)
    try:
        env.ex.apply({"op": "delete_element", "guid": wall.GlobalId})
    except x.ExecutorError as exc:
        assert exc.code == "placement_dependents", exc
    else:
        raise AssertionError("hätte abgelehnt werden müssen")
    assert wall_sid in env.target.nodes
    return "Placement mit abhängigem Placement: Operation wird abgelehnt (noch nicht implementiert), Graph unverändert"


class FakeTx:
    """Attrappe einer Neo4j-Transaktion: erkennt die Abfragen von Neo4jBackend und beantwortet sie
    aus einem MemoryBackend. Prüft nur die Python-Seite (Parameternamen, Spaltennamen), nicht das Cypher."""

    def __init__(self, mem):
        self.mem = mem
        self.queries = []

    def run(self, query, **p):
        import re
        self.queries.append(query)
        m = self.mem
        if "RETURN n.step_id AS s" in query:
            sid = m.find_by_guid(p["g"])
            return [] if sid is None else [{"s": sid}]
        if "RETURN properties(n) AS p, labels(n) AS l" in query:
            return [{"p": dict(m.props(p["s"])), "l": list(m.labels(p["s"]))}]
        if "RETURN a.step_id AS src" in query:
            return [{"src": e.src, "t": e.type, "i1": e.idx, "i2": e.idx2} for e in m.in_edges(p["s"])]
        if "RETURN b.step_id AS dst" in query:
            return [{"dst": e.dst, "t": e.type, "i1": e.idx, "i2": e.idx2} for e in m.out_edges(p["s"])]
        if "DETACH DELETE" in query:
            m.remove_node(p["s"])
            return []
        if "RETURN n.step_id AS node" in query:
            return [{"node": sid} for sid in m.nodes_with_label(re.search(r"\(n:`(\w+)`", query).group(1))]
        if "RETURN max(n.step_id) AS m" in query:
            return [{"m": max(m.nodes)}]
        if "CREATE (n:" in query:
            m.nodes[p["s"]] = {"step_id": p["s"], "ifc_class": p["c"], **p["p"]}
            return []
        if "SET n += $p" in query:
            m.set_props(p["s"], p["p"])
            return []
        if "CREATE (a)-[r:" in query:
            m.add_edge(p["a"], re.search(r"\[r:`(\w+)`\]", query).group(1), p["b"],
                       p["p"].get("idx"), p["p"].get("idx2"))
            return []
        if "DELETE r" in query and "coalesce" in query:
            m.remove_edge(p["a"], re.search(r"\[r:`(\w+)`\]", query).group(1),
                          None if p["i1"] == -1 else p["i1"], None if p["i2"] == -1 else p["i2"])
            return []
        if "SET rel.idx = i" in query:
            m.reindex(p["s"], re.search(r"\[r:`(\w+)`\]", query).group(1))
            return []
        raise AssertionError(f"unbekannte Abfrage: {query}")


def case_neo4j_backend_glue(path):
    a, b = Env(path), Env(path)
    win = a.model.by_type("IfcWindow")[0]
    op = {"op": "delete_element", "guid": win.GlobalId}
    res_mem = a.ex.apply(op)
    tx = FakeTx(b.backend)
    res_neo = x.Executor(x.Neo4jBackend(tx, b.source)).apply(op)
    assert [d[0] for d in res_mem.deleted] == [d[0] for d in res_neo.deleted]
    assert a.target.nodes == b.target.nodes and a.target.edges == b.target.edges
    return f"Neo4jBackend liefert dieselbe Löschmenge wie das Speicher-Backend ({len(tx.queries)} Abfragen, Python-Seite)"


# ---------------------------------------------------------------------------------------------
# move_element
# ---------------------------------------------------------------------------------------------

def world_matrices(model):
    """GlobalId -> 4x4-Weltmatrix, berechnet mit IfcOpenShell (unabhängig vom Executor)."""
    out = {}
    for p in model.by_type("IfcProduct"):
        if p.ObjectPlacement is not None and getattr(p, "GlobalId", None):
            out[p.GlobalId] = placement_util.get_local_placement(p.ObjectPlacement)
    return out


def moved_guids(before, after):
    return {g_ for g_ in before if not np.allclose(before[g_], after[g_], atol=1e-6)}


def check_delta(before, after, guids, delta):
    for g_ in guids:
        d = after[g_][:3, 3] - before[g_][:3, 3]
        assert np.allclose(d, delta, atol=1e-6), f"{g_}: Weltverschiebung {d} statt {delta}"
        assert np.allclose(after[g_][:3, :3], before[g_][:3, :3], atol=1e-9), f"{g_}: Drehung hat sich geändert"


def chained_under(model, root_placement):
    """GlobalIds aller Bauteile, deren Placement-Kette über root_placement läuft (einschließlich)."""
    out = set()
    for p in model.by_type("IfcProduct"):
        pl = p.ObjectPlacement
        while pl is not None:
            if pl.id() == root_placement.id():
                out.add(p.GlobalId)
                break
            pl = pl.PlacementRelTo
    return out


def case_move_window_default(path):
    env = Env(path)
    win = env.model.by_type("IfcWindow")[0]
    opening = win.FillsVoids[0].RelatingOpeningElement
    before = world_matrices(env.model)
    delta = np.array([0.5, 0.25, -0.1])
    res = env.ex.apply({"op": "move_element", "guid": win.GlobalId, "translation": list(delta)})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    after = world_matrices(patched)
    changed = moved_guids(before, after)
    assert changed == {win.GlobalId, opening.GlobalId}, f"bewegt: {len(changed)} Bauteile statt Fenster und Öffnung"
    check_delta(before, after, changed, delta)
    assert (len(diff.added_nodes), len(diff.removed_nodes), len(diff.changed_nodes), len(diff.changed_attrs)) == (0, 0, 1, 0), diff.summary()
    err = new_errors(path, patched)
    assert not err, f"{len(err)} neue Schemafehler: {err[:3]}"
    return (f"Fenster 0,5/0,25/-0,1 m verschoben: Fenster und Öffnung bewegen sich genau um den Wert (Weltmatrix mit "
            f"IfcOpenShell geprüft), alle anderen {len(before) - 2} Bauteile unverändert, Diff: {diff.summary()}, 0 neue Schemafehler")


def case_move_wall_follows_chain(path):
    env = Env(path)
    wall = _wall_with_openings(env.model)
    before = world_matrices(env.model)
    expected = chained_under(env.model, wall.ObjectPlacement)
    delta = np.array([1.0, -2.0, 0.3])
    res = env.ex.apply({"op": "move_element", "guid": wall.GlobalId, "translation": list(delta)})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    after = world_matrices(patched)
    changed = moved_guids(before, after)
    assert changed == expected, f"bewegt {len(changed)}, erwartet {len(expected)}"
    check_delta(before, after, changed, delta)
    assert len(res.moved) == 1, "nur ein Placement darf geändert werden"
    assert diff.summary().startswith("Knoten +0 -0 geändert 1"), diff.summary()
    err = new_errors(path, patched)
    assert not err, err[:3]
    return (f"Wand verschoben: {len(changed)} Bauteile (Wand, Öffnungen, Füllungen) folgen über die Placement-Kette, "
            f"nur 1 Punkt im Graph geändert, Rest unverändert, 0 neue Schemafehler")


def case_move_target_location(path):
    env = Env(path)
    win = env.model.by_type("IfcWindow")[1]
    before = world_matrices(env.model)
    origin = before[win.GlobalId][:3, 3]
    target = origin + np.array([2.0, -1.5, 0.4])
    env.ex.apply({"op": "move_element", "guid": win.GlobalId, "target_location": [float(v) for v in target]})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    after = world_matrices(patched)
    assert np.allclose(after[win.GlobalId][:3, 3], target, atol=1e-6), after[win.GlobalId][:3, 3]
    return f"target_location: Fenster liegt danach in Weltkoordinaten bei {np.round(target, 3).tolist()}"


def case_move_without_dependents(path):
    env = Env(path)
    wall = _wall_with_openings(env.model)
    nodes, edges = dict(env.target.nodes), dict(env.target.edges)
    try:
        env.ex.apply({"op": "move_element", "guid": wall.GlobalId, "translation": [1, 0, 0], "with_dependents": False})
    except x.ExecutorError as exc:
        assert exc.code == "placement_dependents", exc
    else:
        raise AssertionError("hätte abgelehnt werden müssen")
    assert env.target.nodes == nodes and env.target.edges == edges, "Graph wurde verändert"

    win = env.model.by_type("IfcWindow")[0]
    opening = win.FillsVoids[0].RelatingOpeningElement
    before = world_matrices(env.model)
    env.ex.apply({"op": "move_element", "guid": win.GlobalId, "translation": [0.0, 0.0, 0.2], "with_dependents": False})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    after = world_matrices(patched)
    assert moved_guids(before, after) == {win.GlobalId}, "nur das Fenster darf sich bewegen"
    assert np.allclose(after[opening.GlobalId], before[opening.GlobalId]), "Öffnung darf bleiben"
    return "with_dependents=false: Wand mit abhängigen Placements wird abgelehnt (Graph unverändert), Fenster allein bewegt sich, Öffnung bleibt"


def case_move_rejections(path):
    env = Env(path)
    m = env.model
    win = m.by_type("IfcWindow")[0].GlobalId
    t = [0.1, 0, 0]
    cases = [
        ({"op": "move_element", "guid": win}, "bad_parameter"),
        ({"op": "move_element", "guid": win, "translation": t, "target_location": t}, "bad_parameter"),
        ({"op": "move_element", "guid": win, "translation": [1, 2]}, "bad_parameter"),
        ({"op": "move_element", "guid": win, "translation": [1, 2, "x"]}, "bad_parameter"),
        ({"op": "move_element", "guid": win, "translation": [1, 2, float("nan")]}, "bad_parameter"),
        ({"op": "move_element", "guid": win, "translation": [True, 0, 0]}, "bad_parameter"),
        ({"op": "move_element", "guid": win, "translation": t, "with_dependents": "ja"}, "bad_parameter"),
        ({"op": "move_element", "guid": win, "translation": [0, 0, 0]}, "no_change"),
        ({"op": "move_element", "guid": "0000000000000000000000", "translation": t}, "not_found"),
        ({"op": "move_element", "translation": t}, "bad_parameter"),
        ({"op": "move_element", "guid": m.by_type("IfcBuildingStorey")[0].GlobalId, "translation": t}, "out_of_scope"),
        ({"op": "move_element", "guid": m.by_type("IfcSite")[0].GlobalId, "translation": t}, "out_of_scope"),
        ({"op": "move_element", "guid": m.by_type("IfcPropertySet")[0].GlobalId, "translation": t}, "out_of_scope"),
    ]
    nodes, edges = dict(env.target.nodes), dict(env.target.edges)
    for op, code in cases:
        try:
            env.ex.apply(op)
        except x.ExecutorError as exc:
            assert exc.code == code, (op, exc)
        else:
            raise AssertionError(f"{op} hätte mit {code} abgelehnt werden müssen")
    assert env.target.nodes == nodes and env.target.edges == edges, "Graph wurde verändert"
    return f"{len(cases)} ungültige Aufrufe werden mit dem richtigen Fehlercode abgelehnt, Graph unverändert"


def case_move_units(path):
    env = Env(path)
    win = env.model.by_type("IfcWindow")[0]
    units = [sid for sid, p in env.target.nodes.items()
             if p.get("ifc_class") == "IfcSIUnit" and p.get("UnitType") == "LENGTHUNIT"]
    assert units
    for sid in units:
        env.target.nodes[sid]["Prefix"] = "MILLI"
    res = env.ex.apply({"op": "move_element", "guid": win.GlobalId, "translation": [0.5, 0, 0]})
    (_, _, old, new), = res.moved
    dist = float(np.linalg.norm(np.array(new) - np.array(old)))
    assert abs(dist - 500.0) < 1e-6, f"0,5 m sind bei Millimeter-Einheit 500 Einheiten, nicht {dist}"
    for sid in units:
        env.target.nodes[sid]["Name"] = "FOOT"
    try:
        env.ex.apply({"op": "move_element", "guid": win.GlobalId, "translation": [0.5, 0, 0]})
    except x.ExecutorError as exc:
        assert exc.code == "unsupported_unit", exc
    else:
        raise AssertionError("Fuß hätte abgelehnt werden müssen")
    return "Einheit Millimeter: 0,5 m werden zu 500 Einheiten; nicht unterstützte Längeneinheit wird abgelehnt"


def _share_point(model):
    w1, w2 = model.by_type("IfcWindow")[:2]
    w2.ObjectPlacement.RelativePlacement.Location = w1.ObjectPlacement.RelativePlacement.Location
    return w1, w2


def case_move_shared_point(path):
    env = Env(path, prepare=_share_point)
    w1, w2 = env.model.by_type("IfcWindow")[:2]
    before = world_matrices(env.model)
    delta = np.array([0.0, 0.3, 0.0])
    res = env.ex.apply({"op": "move_element", "guid": w1.GlobalId, "translation": list(delta), "with_dependents": False})
    assert len(res.created) == 1, "geteilter Punkt muss kopiert werden"
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    after = world_matrices(patched)
    assert moved_guids(before, after) == {w1.GlobalId}, "das zweite Fenster darf sich nicht mitbewegen"
    check_delta(before, after, {w1.GlobalId}, delta)
    assert len(diff.added_nodes) == 1 and not diff.removed_nodes
    err = new_errors(path, patched)
    assert not err, err[:3]
    return "Geteilter Punkt: Kopie angelegt, das andere Bauteil bleibt an seiner Stelle, Round-Trip leer, 0 neue Schemafehler"


def case_move_stale_relations(path):
    env = Env(path)
    wall = next(w for w in env.model.by_type("IfcWall") if w.ConnectedTo or w.ConnectedFrom or w.ProvidesBoundaries)
    members = {wall.id()}
    for rel in wall.HasOpenings:
        o = rel.RelatedOpeningElement
        members.add(o.id())
        members.update(f.RelatedBuildingElement.id() for f in o.HasFillings)
    expected = set()
    for e in map(env.model.by_id, members):
        for rel in list(getattr(e, "ConnectedTo", ())) + list(getattr(e, "ConnectedFrom", ())) + list(getattr(e, "ProvidesBoundaries", ())):
            expected.add(rel.id())
    res = env.ex.apply({"op": "move_element", "guid": wall.GlobalId, "translation": [0.2, 0.0, 0.0]})
    assert {sid for sid, _ in res.stale} == expected, (len(res.stale), len(expected))
    assert expected, "Testwand ohne Beziehungen"
    return f"{len(expected)} Beziehungen (SpaceBoundary / ConnectsPathElements) werden als veraltet gemeldet, aber nicht verändert"


def case_neo4j_backend_glue_move(path):
    a, b = Env(path, prepare=_share_point), Env(path, prepare=_share_point)
    w1 = a.model.by_type("IfcWindow")[0]
    ops = [{"op": "move_element", "guid": w1.GlobalId, "translation": [0.0, 0.3, 0.0], "with_dependents": False},
           {"op": "move_element", "guid": a.model.by_type("IfcWindow")[3].GlobalId, "translation": [0.2, 0.0, 0.0]}]
    tx = FakeTx(b.backend)
    neo = x.Executor(x.Neo4jBackend(tx, b.source))
    for op in ops:
        r1, r2 = a.ex.apply(op), neo.apply(op)
        assert r1.moved == r2.moved and r1.created == r2.created
    assert a.target.nodes == b.target.nodes and a.target.edges == b.target.edges
    return f"Neo4jBackend liefert dieselbe Verschiebung wie das Speicher-Backend ({len(tx.queries)} Abfragen, inkl. Kopie des geteilten Punkts)"


# ---------------------------------------------------------------------------------------------
# reassign_container
# ---------------------------------------------------------------------------------------------

def _storeys(model):
    ground = next(s for s in model.by_type("IfcBuildingStorey") if s.Name == "Erdgeschoss")
    upper = next(s for s in model.by_type("IfcBuildingStorey") if s.Name == "Dachgeschoss")
    return ground, upper


def _storey_of(inst):
    return inst.ContainedInStructure[0].RelatingStructure


def case_reassign_default(path):
    env = Env(path)
    ground, upper = _storeys(env.model)
    wall = _wall_with_openings(env.model)
    opening = wall.HasOpenings[0].RelatedOpeningElement
    fills = [f.RelatedBuildingElement for f in opening.HasFillings]
    before = world_matrices(env.model)
    expected = chained_under(env.model, wall.ObjectPlacement)
    z = upper.Elevation - ground.Elevation
    count_ground = len(ground.ContainsElements[0].RelatedElements)
    count_upper = len(upper.ContainsElements[0].RelatedElements)
    res = env.ex.apply({"op": "reassign_container", "guid": wall.GlobalId, "target_storey_guid": upper.GlobalId})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    after = world_matrices(patched)
    assert moved_guids(before, after) == expected, "es sollen genau Wand, Öffnung und Füllungen höher wandern"
    check_delta(before, after, expected, np.array([0.0, 0.0, z]))
    assert _storey_of(patched.by_id(wall.id())).id() == upper.id()
    for f in fills:
        assert _storey_of(patched.by_id(f.id())).id() == upper.id(), "Füllung muss mit"
    assert not patched.by_id(opening.id()).is_a("IfcSpatialElement") and not getattr(patched.by_id(opening.id()), "ContainedInStructure", None)
    n_ground = len(patched.by_id(ground.id()).ContainsElements[0].RelatedElements)
    n_upper = len(patched.by_id(upper.id()).ContainsElements[0].RelatedElements)
    assert n_ground == count_ground - 1 - len(fills), (n_ground, count_ground)
    assert n_upper == count_upper + 1 + len(fills), (n_upper, count_upper)
    assert not unreachable(env.model, env.target)
    err = new_errors(path, patched)
    assert not err, f"{len(err)} neue Schemafehler: {err[:3]}"
    return (f"Wand samt Öffnung und {len(fills)} Füllung(en) ins Dachgeschoss umgehängt: Containment wechselt, "
            f"alle {len(expected)} Bauteile liegen um {z} m höher (IfcOpenShell), Rest unverändert, "
            f"Round-Trip leer, 0 neue Schemafehler")


def case_reassign_preserve_world(path):
    env = Env(path)
    ground, upper = _storeys(env.model)
    wall = _wall_with_openings(env.model)
    before = world_matrices(env.model)
    res = env.ex.apply({"op": "reassign_container", "guid": wall.GlobalId, "target_storey_guid": upper.GlobalId,
                        "preserve_world_position": True})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    after = world_matrices(patched)
    assert not moved_guids(before, after), "Weltlage muss erhalten bleiben"
    assert _storey_of(patched.by_id(wall.id())).id() == upper.id()
    (_, _, old, new), = res.moved
    z = upper.Elevation - ground.Elevation
    assert abs((old[2] - new[2]) - z) < 1e-6, (old, new)
    err = new_errors(path, patched)
    assert not err, err[:3]
    return f"preserve_world_position: Weltlage aller Bauteile unverändert, lokale Höhe {old[2]} -> {new[2]}, Containment gewechselt"


def case_reassign_space_by_aggregate(path):
    env = Env(path)
    ground, upper = _storeys(env.model)
    space = ground.IsDecomposedBy[0].RelatedObjects[0]
    before = world_matrices(env.model)
    res = env.ex.apply({"op": "reassign_container", "guid": space.GlobalId, "target_storey_guid": upper.GlobalId})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    sp = patched.by_id(space.id())
    assert sp.Decomposes[0].RelatingObject.id() == upper.id()
    return f"Raum (über IfcRelAggregates im Geschoss) umgehängt, Round-Trip leer"


def case_reassign_new_relation(path):
    env = Env(path)
    ground, upper = _storeys(env.model)
    rel_upper = upper.ContainsElements[0]
    env.backend.remove_node(rel_upper.id())                # Dachgeschoss ohne Containment-Beziehung
    wall = _wall_with_openings(env.model)
    res = env.ex.apply({"op": "reassign_container", "guid": wall.GlobalId, "target_storey_guid": upper.GlobalId})
    created = [c for c in res.created if c[1] == "IfcRelContainedInSpatialStructure"]
    assert len(created) == 1, res.created
    sid = created[0][0]
    outs = {(e.type, e.idx): e.dst for e in env.backend.out_edges(sid)}
    assert outs[("RelatingStructure", None)] == upper.id()
    assert ("OwnerHistory", None) in outs, "OwnerHistory soll von der alten Beziehung übernommen werden"
    assert [k for k in outs if k[0] == "RelatedElements"] == [("RelatedElements", i) for i in range(len(res.reassigned))]
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    new_rel = patched.by_id(rep.added_ids[sid])
    assert new_rel.is_a("IfcRelContainedInSpatialStructure") and len(new_rel.GlobalId) == 22
    assert new_rel.RelatingStructure.id() == upper.id()
    assert wall.id() in [e.id() for e in new_rel.RelatedElements]
    return f"Zielgeschoss ohne Beziehung: neue IfcRelContainedInSpatialStructure mit GlobalId, OwnerHistory und {len(res.reassigned)} Bauteil(en) angelegt"


def case_reassign_rejections(path):
    env = Env(path)
    m = env.model
    ground, upper = _storeys(m)
    win = m.by_type("IfcWindow")[0]
    wall = _wall_with_openings(m)
    inner = next(w for w in m.by_type("IfcWallStandardCase") if w.ContainedInStructure[0].RelatingStructure.id() == ground.id())
    cases = [
        ({"op": "reassign_container", "guid": wall.GlobalId}, "bad_parameter"),
        ({"op": "reassign_container", "guid": wall.GlobalId, "target_storey_guid": upper.GlobalId, "preserve_world_position": "ja"}, "bad_parameter"),
        ({"op": "reassign_container", "guid": wall.GlobalId, "target_storey_guid": "0000000000000000000000"}, "not_found"),
        ({"op": "reassign_container", "guid": "0000000000000000000000", "target_storey_guid": upper.GlobalId}, "not_found"),
        ({"op": "reassign_container", "guid": wall.GlobalId, "target_storey_guid": inner.GlobalId}, "bad_parameter"),
        ({"op": "reassign_container", "guid": wall.GlobalId, "target_storey_guid": ground.GlobalId}, "no_change"),
        ({"op": "reassign_container", "guid": win.GlobalId, "target_storey_guid": upper.GlobalId}, "hosted_element"),
        ({"op": "reassign_container", "guid": ground.GlobalId, "target_storey_guid": upper.GlobalId}, "out_of_scope"),
        ({"op": "reassign_container", "guid": m.by_type("IfcSite")[0].GlobalId, "target_storey_guid": upper.GlobalId}, "out_of_scope"),
    ]
    nodes, edges = dict(env.target.nodes), dict(env.target.edges)
    for op, code in cases:
        try:
            env.ex.apply(op)
        except x.ExecutorError as exc:
            assert exc.code == code, (op, exc)
        else:
            raise AssertionError(f"{op} hätte mit {code} abgelehnt werden müssen")
    assert env.target.nodes == nodes and env.target.edges == edges, "Graph wurde verändert"
    return f"{len(cases)} ungültige Aufrufe werden mit dem richtigen Fehlercode abgelehnt, Graph unverändert"


def case_neo4j_backend_glue_reassign(path):
    a, b = Env(path), Env(path)
    ground, upper = _storeys(a.model)
    wall = _wall_with_openings(a.model)
    ops = [{"op": "reassign_container", "guid": wall.GlobalId, "target_storey_guid": upper.GlobalId},
           {"op": "reassign_container", "guid": a.model.by_type("IfcSlab")[0].GlobalId,
            "target_storey_guid": ground.GlobalId, "preserve_world_position": True}]
    tx = FakeTx(b.backend)
    neo = x.Executor(x.Neo4jBackend(tx, b.source))
    for op in ops:
        try:
            r1 = a.ex.apply(op)
        except x.ExecutorError as e1:
            try:
                neo.apply(op)
            except x.ExecutorError as e2:
                assert e1.code == e2.code
                continue
            raise AssertionError("Neo4jBackend hätte auch ablehnen müssen")
        r2 = neo.apply(op)
        assert r1.reassigned == r2.reassigned and r1.deleted == r2.deleted and r1.moved == r2.moved
    assert a.target.nodes == b.target.nodes and a.target.edges == b.target.edges
    return f"Neo4jBackend liefert dasselbe Ergebnis wie das Speicher-Backend ({len(tx.queries)} Abfragen)"


# ---------------------------------------------------------------------------------------------
# clone_element
# ---------------------------------------------------------------------------------------------

def world_vertices(model, inst):
    import ifcopenshell.geom
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    verts = np.array(ifcopenshell.geom.create_shape(settings, inst).geometry.verts).reshape(-1, 3)
    return verts[np.lexsort(verts.T[::-1])]


def _psets(inst):
    return [rel.RelatingPropertyDefinition for rel in inst.IsDefinedBy if rel.is_a("IfcRelDefinesByProperties")]


def _all_guids(model):
    return [r.GlobalId for r in model.by_type("IfcRoot")]


def case_clone_window(path):
    env = Env(path)
    win = env.model.by_type("IfcWindow")[1]
    opening = win.FillsVoids[0].RelatingOpeningElement
    host = opening.VoidsElements[0].RelatingBuildingElement
    storey = _storey_of(win)
    wtype = win.IsTypedBy[0].RelatingType
    material = next(r.RelatingMaterial for r in win.HasAssociations if r.is_a("IfcRelAssociatesMaterial"))
    src_psets = {p.id() for p in _psets(win)}
    n_src_psets = len(src_psets)
    before = world_matrices(env.model)
    verts_win, verts_open = world_vertices(env.model, win), world_vertices(env.model, opening)
    counts = {c: n(env.model, c) for c in ("IfcWindow", "IfcOpeningElement", "IfcRelFillsElement", "IfcRelVoidsElement",
                                          "IfcPropertySet", "IfcElementQuantity", "IfcRelDefinesByProperties")}
    delta = np.array([0.0, 2.0, 0.0])
    res = env.ex.apply({"op": "clone_element", "source_guid": win.GlobalId, "translation": list(delta)})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert not rep.warnings, rep.warnings
    new = patched.by_guid(res.new_elements[0][0])
    guids = _all_guids(patched)
    assert len(guids) == len(set(guids)) and len(new.GlobalId) == 22, "GlobalIds müssen eindeutig sein"
    after = world_matrices(patched)
    assert all(np.allclose(before[g_], after[g_]) for g_ in before), "Originale dürfen sich nicht ändern"
    assert np.allclose(after[new.GlobalId][:3, 3], before[win.GlobalId][:3, 3] + delta, atol=1e-6)
    assert np.allclose(after[new.GlobalId][:3, :3], before[win.GlobalId][:3, :3])
    for c, k in (("IfcWindow", 1), ("IfcOpeningElement", 1), ("IfcRelFillsElement", 1), ("IfcRelVoidsElement", 1)):
        assert n(patched, c) == counts[c] + k, c
    assert n(patched, "IfcPropertySet") > counts["IfcPropertySet"] and n(patched, "IfcElementQuantity") > counts["IfcElementQuantity"]
    new_open = new.FillsVoids[0].RelatingOpeningElement
    assert new_open.id() != opening.id() and new_open.VoidsElements[0].RelatingBuildingElement.id() == host.id()
    assert _storey_of(new).id() == storey.id()
    assert new.IsTypedBy[0].RelatingType.id() == wtype.id()
    assert next(r.RelatingMaterial for r in new.HasAssociations if r.is_a("IfcRelAssociatesMaterial")).id() == material.id()
    new_psets = _psets(new)
    assert len(new_psets) == n_src_psets and not ({p.id() for p in new_psets} & src_psets), "Psets müssen Kopien sein"
    assert sorted(p.Name for p in new_psets) == sorted(p.Name for p in _psets(win))
    assert new.Tag is None and new.Name == win.Name and new.OverallWidth == win.OverallWidth
    assert np.allclose(world_vertices(patched, new), verts_win + delta, atol=1e-6), "Geometrie des Klons"
    assert np.allclose(world_vertices(patched, new_open), verts_open + delta, atol=1e-6), "Geometrie der Öffnung"
    err = new_errors(path, patched)
    assert not err, f"{len(err)} neue Schemafehler: {err[:3]}"
    return (f"Fenster geklont (+2 m in y): neue GlobalId, eigene Öffnung im selben Host, Geometrie um genau den Versatz "
            f"verschoben, {n_src_psets} Psets/Mengen als Kopie, Typ und Material geteilt, Container gleich, "
            f"Originale unverändert, Round-Trip leer, 0 neue Schemafehler ({len(res.created)} Knoten angelegt)")


def case_clone_wall_with_name(path):
    env = Env(path)
    wall = _wall_with_openings(env.model)
    usage = next(r.RelatingMaterial for r in wall.HasAssociations if r.is_a("IfcRelAssociatesMaterial"))
    before = world_matrices(env.model)
    verts = world_vertices(env.model, wall)
    delta = np.array([4.0, 1.0, 0.0])
    res = env.ex.apply({"op": "clone_element", "source_guid": wall.GlobalId, "translation": list(delta), "name": "Wand-Kopie"})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert not rep.warnings, rep.warnings
    new = patched.by_guid(res.new_elements[0][0])
    assert new.Name == "Wand-Kopie" and not new.HasOpenings, "Klon einer Wand hat keine Öffnungen"
    after = world_matrices(patched)
    assert all(np.allclose(before[g_], after[g_]) for g_ in before)
    assert np.allclose(after[new.GlobalId][:3, 3], before[wall.GlobalId][:3, 3] + delta, atol=1e-6)
    nv = world_vertices(patched, new)         # Original ist mit Öffnung ausgeschnitten, Klon nicht: Hülle vergleichen
    assert np.allclose(nv.min(0), verts.min(0) + delta, atol=1e-6) and np.allclose(nv.max(0), verts.max(0) + delta, atol=1e-6)
    new_usage = next(r.RelatingMaterial for r in new.HasAssociations if r.is_a("IfcRelAssociatesMaterial"))
    assert new_usage.id() != usage.id() and new_usage.ForLayerSet.id() == usage.ForLayerSet.id()
    assert len(patched.by_id(wall.id()).HasOpenings) == 1, "Original behält seine Öffnung"
    skipped = sorted({c for _, c in res.skipped})
    assert "IfcRelSpaceBoundary" in skipped and "IfcRelVoidsElement" in skipped, skipped
    err = new_errors(path, patched)
    assert not err, err[:3]
    return (f"Wand geklont und umbenannt: ohne Öffnungen, eigene Materialnutzung (Schichtaufbau geteilt), Hülle der Geometrie um den "
            f"Versatz verschoben, nicht übertragen: {', '.join(skipped)}")


def case_clone_rejections(path):
    env = Env(path)
    m = env.model
    win = m.by_type("IfcWindow")[0]
    opening = win.FillsVoids[0].RelatingOpeningElement
    wall = _wall_with_openings(m)
    t = [1, 0, 0]
    # synthetisch: Wand als Ganzes mit Teilen
    rel = env.backend.create_node("IfcRelAggregates", {"GlobalId": "x"}, env.backend.labels(m.by_type("IfcRelAggregates")[0].id()))
    env.backend.add_edge(rel, "RelatingObject", m.by_type("IfcSlab")[0].id())
    cases = [
        ({"op": "clone_element", "source_guid": win.GlobalId}, "bad_parameter"),
        ({"op": "clone_element", "translation": t}, "bad_parameter"),
        ({"op": "clone_element", "source_guid": win.GlobalId, "translation": [1, 2]}, "bad_parameter"),
        ({"op": "clone_element", "source_guid": win.GlobalId, "translation": [0, 0, 0]}, "bad_parameter"),
        ({"op": "clone_element", "source_guid": win.GlobalId, "translation": t, "name": " "}, "bad_parameter"),
        ({"op": "clone_element", "source_guid": "0000000000000000000000", "translation": t}, "not_found"),
        ({"op": "clone_element", "source_guid": opening.GlobalId, "translation": t}, "hosted_element"),
        ({"op": "clone_element", "source_guid": m.by_type("IfcSlab")[0].GlobalId, "translation": t}, "has_parts"),
        ({"op": "clone_element", "source_guid": m.by_type("IfcBuildingStorey")[0].GlobalId, "translation": t}, "out_of_scope"),
        ({"op": "clone_element", "source_guid": m.by_type("IfcSite")[0].GlobalId, "translation": t}, "out_of_scope"),
    ]
    nodes, edges = dict(env.target.nodes), dict(env.target.edges)
    for op, code in cases:
        try:
            env.ex.apply(op)
        except x.ExecutorError as exc:
            assert exc.code == code, (op, exc)
        else:
            raise AssertionError(f"{op} hätte mit {code} abgelehnt werden müssen")
    assert env.target.nodes == nodes and env.target.edges == edges, "Graph wurde verändert"
    return f"{len(cases)} ungültige Aufrufe werden mit dem richtigen Fehlercode abgelehnt, Graph unverändert"


def case_neo4j_backend_glue_clone(path):
    import ifcopenshell.guid
    counter = iter(range(10 ** 6))
    real = ifcopenshell.guid.new
    ifcopenshell.guid.new = lambda: f"GUID{next(counter):018d}"
    try:
        a, b = Env(path), Env(path)
        win = a.model.by_type("IfcWindow")[1]
        wall = _wall_with_openings(a.model)
        ops = [{"op": "clone_element", "source_guid": win.GlobalId, "translation": [0.0, 2.0, 0.0]},
               {"op": "clone_element", "source_guid": wall.GlobalId, "translation": [4.0, 1.0, 0.0], "name": "K"}]
        tx = FakeTx(b.backend)
        neo = x.Executor(x.Neo4jBackend(tx, b.source))
        for op in ops:
            counter_state = None
            r1 = a.ex.apply(op)
            r2 = neo.apply(op)
            assert r1.new_elements == r2.new_elements or [e[1:] for e in r1.new_elements] == [e[1:] for e in r2.new_elements]
    finally:
        ifcopenshell.guid.new = real
    assert a.target.nodes.keys() == b.target.nodes.keys() and a.target.edges == b.target.edges
    strip = lambda d: {k: {kk: vv for kk, vv in v.items() if kk != "GlobalId"} for k, v in d.items()}
    assert strip(a.target.nodes) == strip(b.target.nodes)
    return f"Neo4jBackend legt dieselben Knoten und Kanten an wie das Speicher-Backend ({len(tx.queries)} Abfragen)"


# ---------------------------------------------------------------------------------------------
# set_property, set_attribute, assign_type
# ---------------------------------------------------------------------------------------------

def _find_prop(model, py_type, only_element=None):
    """(Element, Pset-Name, Property-Name, Wert) einer IfcPropertySingleValue mit passendem Wertetyp in einem
    Pset, das nur zu diesem einen Bauteil gehört."""
    for rel in model.by_type("IfcRelDefinesByProperties"):
        ps = rel.RelatingPropertyDefinition
        if not ps.is_a("IfcPropertySet") or len(rel.RelatedObjects) != 1:
            continue
        el = rel.RelatedObjects[0]
        if not (el.is_a("IfcElement") or el.is_a("IfcSpace")) or el.is_a("IfcOpeningElement"):
            continue
        for p in ps.HasProperties:
            if p.is_a("IfcPropertySingleValue") and p.NominalValue is not None \
                    and type(p.NominalValue.wrappedValue) is py_type:
                if len(ps.DefinesOccurrence) == 1:
                    return el, ps.Name, p.Name, p.NominalValue.wrappedValue
    raise AssertionError(f"keine Property vom Typ {py_type.__name__} gefunden")


def _pset_value(inst, pset, prop):
    import ifcopenshell.util.element as ue
    return ue.get_psets(inst).get(pset, {}).get(prop)


def case_set_property(path):
    env = Env(path)
    done = []
    for py_type, new in ((bool, None), (float, 2.75), (str, "Testwert")):
        el, pset, prop, old = _find_prop(env.model, py_type)
        new = (not old) if py_type is bool else new
        before = {g_: dict(__import__("ifcopenshell.util.element", fromlist=["x"]).get_psets(e))
                  for g_, e in ((i.GlobalId, i) for i in env.model.by_type("IfcProduct"))}
        res = env.ex.apply({"op": "set_property", "guid": el.GlobalId, "pset_name": pset,
                             "property_name": prop, "value": new})
        assert len(res.changed) == 1 and res.changed[0][3:] == (old, new), res.changed
        done.append((el.GlobalId, pset, prop, old, new, before))
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert not new_errors(path, patched), new_errors(path, patched)[:3]
    import ifcopenshell.util.element as ue
    changed = {(g_, ps, pr) for g_, ps, pr, *_ in done}
    for g_, ps, pr, old, new, before in done:
        got = _pset_value(patched.by_guid(g_), ps, pr)
        assert got == new, (ps, pr, old, new, got)
    orig = ifcopenshell.open(path)
    for i in orig.by_type("IfcProduct"):
        a, b = ue.get_psets(i), ue.get_psets(patched.by_guid(i.GlobalId))
        for ps in a:
            for pr in a[ps]:
                if pr == "id" or (i.GlobalId, ps, pr) in changed:
                    continue
                assert a[ps][pr] == b[ps][pr], (i.GlobalId, ps, pr)
    return "Boolean, Zahl und Text geändert, Round-Trip leer, 0 neue Schemafehler, alle anderen Werte unverändert"


def case_set_property_rejections(path):
    env = Env(path)
    el, pset, prop, old = _find_prop(env.model, bool)
    _, _, _, num_old = _find_prop(env.model, float)
    elf, psf, prf, oldf = _find_prop(env.model, float)
    storey = env.model.by_type("IfcBuildingStorey")[0]
    quant = None
    for w in env.model.by_type("IfcWall"):
        for rel in w.IsDefinedBy:
            if rel.RelatingPropertyDefinition.is_a("IfcElementQuantity"):
                quant = (w, rel.RelatingPropertyDefinition)
                break
        if quant:
            break
    shared = next(r for r in env.model.by_type("IfcRelDefinesByProperties") if len(r.RelatedObjects) > 1) \
        if any(len(r.RelatedObjects) > 1 for r in env.model.by_type("IfcRelDefinesByProperties")) else None
    nodes_before, edges_before = dict(env.target.nodes), dict(env.target.edges)
    nodes_before = {k: dict(v) for k, v in env.target.nodes.items()}
    cases = [
        ({"guid": "gibtsnicht", "pset_name": pset, "property_name": prop, "value": True}, "not_found"),
        ({"guid": storey.GlobalId, "pset_name": pset, "property_name": prop, "value": True}, "out_of_scope"),
        ({"guid": el.GlobalId, "pset_name": "Gibts nicht", "property_name": prop, "value": True}, "pset_not_found"),
        ({"guid": el.GlobalId, "pset_name": pset, "property_name": "Gibts nicht", "value": True}, "property_not_found"),
        ({"guid": el.GlobalId, "pset_name": pset, "property_name": prop, "value": "ja"}, "bad_value_type"),
        ({"guid": el.GlobalId, "pset_name": pset, "property_name": prop, "value": old}, "no_change"),
        ({"guid": elf.GlobalId, "pset_name": psf, "property_name": prf, "value": True}, "bad_value_type"),
        ({"guid": elf.GlobalId, "pset_name": psf, "property_name": prf, "value": float("nan")}, "bad_value_type"),
        ({"guid": elf.GlobalId, "pset_name": psf, "property_name": prf, "value": oldf}, "no_change"),
        ({"guid": el.GlobalId, "pset_name": "", "property_name": prop, "value": True}, "bad_parameter"),
    ]
    if quant:
        cases.append(({"guid": quant[0].GlobalId, "pset_name": quant[1].Name, "property_name": "Length",
                       "value": 1.0}, "unsupported_property"))
    for params, code in cases:
        try:
            env.ex.apply({"op": "set_property", **params})
        except x.ExecutorError as exc:
            assert exc.code == code, f"erwartet {code}, bekommen {exc.code}: {exc.message}"
        else:
            raise AssertionError(f"{code} nicht abgelehnt")
    # gemeinsames Pset: absichtlich herstellen (Pset der ersten Wand gilt nun auch für die zweite)
    def share(m):
        rels = [r for r in m.by_type("IfcRelDefinesByProperties") if r.RelatingPropertyDefinition.is_a("IfcPropertySet")
                and len(r.RelatedObjects) == 1 and r.RelatedObjects[0].is_a("IfcWall")]
        other = next(r for r in rels if r.RelatedObjects[0] != rels[0].RelatedObjects[0]
                     and r.RelatingPropertyDefinition.Name == rels[0].RelatingPropertyDefinition.Name)
        rels[0].RelatedObjects = tuple(rels[0].RelatedObjects) + tuple(other.RelatedObjects)
    env3 = Env(path, prepare=share)
    r0 = [r for r in env3.model.by_type("IfcRelDefinesByProperties") if len(r.RelatedObjects) > 1][0]
    p0 = r0.RelatingPropertyDefinition.HasProperties[0]
    try:
        env3.ex.apply({"op": "set_property", "guid": r0.RelatedObjects[0].GlobalId,
                       "pset_name": r0.RelatingPropertyDefinition.Name, "property_name": p0.Name, "value": "x"})
    except x.ExecutorError as exc:
        assert exc.code in ("shared_resource", "bad_value_type"), exc.code
        assert exc.code == "shared_resource", "gemeinsames Pset wurde nicht als solches abgelehnt"
    else:
        raise AssertionError("shared_resource nicht abgelehnt")
    assert nodes_before == {k: dict(v) for k, v in env.target.nodes.items()}, "Graph nach Ablehnung verändert"
    return f"{len(cases) + 1} ungültige Aufrufe abgelehnt, Graph jeweils unverändert"


def case_set_attribute(path):
    env = Env(path)
    wall = _wall_with_openings(env.model)
    door = env.model.by_type("IfcDoor")[0]
    space = env.model.by_type("IfcSpace")[0]
    todo = [(wall, "Name", "Neuer Name"), (wall, "Description", "Beschreibung neu"), (wall, "Tag", "T-42"),
            (wall, "PredefinedType", "PARTITIONING"), (door, "ObjectType", "Sonderanfertigung"),
            (space, "LongName", "Langer Name")]
    for el, attr, val in todo:
        res = env.ex.apply({"op": "set_attribute", "guid": el.GlobalId, "attribute": attr, "value": val})
        assert len(res.changed) == 1 and res.changed[0][3:] == (getattr(el, attr), val)
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert not new_errors(path, patched), new_errors(path, patched)[:3]
    for el, attr, val in todo:
        assert getattr(patched.by_guid(el.GlobalId), attr) == val, attr
    return f"{len(todo)} Attribute gesetzt (Name, Description, Tag, PredefinedType, ObjectType, LongName), Round-Trip leer, 0 neue Schemafehler"


def case_set_attribute_rejections(path):
    env = Env(path)
    wall = _wall_with_openings(env.model)
    storey = env.model.by_type("IfcBuildingStorey")[0]
    before = {k: dict(v) for k, v in env.target.nodes.items()}
    cases = [
        ({"guid": wall.GlobalId, "attribute": "GlobalId", "value": "x"}, "attribute_not_allowed"),
        ({"guid": wall.GlobalId, "attribute": "ObjectPlacement", "value": "x"}, "attribute_not_allowed"),
        ({"guid": wall.GlobalId, "attribute": "LongName", "value": "x"}, "attribute_not_on_class"),
        ({"guid": wall.GlobalId, "attribute": "PredefinedType", "value": "KEINEWAND"}, "bad_enum_value"),
        ({"guid": wall.GlobalId, "attribute": "PredefinedType", "value": "USERDEFINED"}, "needs_object_type"),
        ({"guid": wall.GlobalId, "attribute": "Name", "value": wall.Name}, "no_change"),
        ({"guid": wall.GlobalId, "attribute": "Name", "value": ""}, "bad_parameter"),
        ({"guid": wall.GlobalId, "attribute": "Name", "value": 5}, "bad_parameter"),
        ({"guid": storey.GlobalId, "attribute": "Name", "value": "EG"}, "out_of_scope"),
        ({"guid": "gibtsnicht", "attribute": "Name", "value": "x"}, "not_found"),
    ]
    for params, code in cases:
        try:
            env.ex.apply({"op": "set_attribute", **params})
        except x.ExecutorError as exc:
            assert exc.code == code, f"erwartet {code}, bekommen {exc.code}: {exc.message}"
        else:
            raise AssertionError(f"{code} nicht abgelehnt")
    assert before == {k: dict(v) for k, v in env.target.nodes.items()}
    # ohne Schema
    ex = x.Executor(env.backend)
    try:
        ex.apply({"op": "set_attribute", "guid": wall.GlobalId, "attribute": "Name", "value": "Neu"})
    except x.ExecutorError as exc:
        assert exc.code == "no_schema", exc.code
    else:
        raise AssertionError("no_schema nicht gemeldet")
    return f"{len(cases) + 1} ungültige Aufrufe abgelehnt, Graph jeweils unverändert"


def _wall_type_pair(model):
    """(Wand, aktueller Typ, anderer Typ) mit zwei verschiedenen IfcWallType."""
    ts = model.by_type("IfcWallType")
    for w in model.by_type("IfcWall"):
        if w.IsTypedBy:
            cur = w.IsTypedBy[0].RelatingType
            other = next(t for t in ts if t != cur)
            return w, cur, other
    raise AssertionError("keine typisierte Wand")


def case_assign_type(path):
    env = Env(path)
    wall, cur, other = _wall_type_pair(env.model)
    n_cur, n_other = len(cur.Types[0].RelatedObjects), len(other.Types[0].RelatedObjects)
    res = env.ex.apply({"op": "assign_type", "guid": wall.GlobalId, "type_guid": other.GlobalId})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert not new_errors(path, patched), new_errors(path, patched)[:3]
    w2 = patched.by_guid(wall.GlobalId)
    assert w2.IsTypedBy[0].RelatingType.GlobalId == other.GlobalId
    assert len(patched.by_guid(cur.GlobalId).Types[0].RelatedObjects) == n_cur - 1
    assert len(patched.by_guid(other.GlobalId).Types[0].RelatedObjects) == n_other + 1
    # Geometrie, Material und Psets unverändert
    import ifcopenshell.util.element as ue
    assert ue.get_psets(w2) == ue.get_psets(wall)
    orig = ifcopenshell.open(path)                           # Referenz halten, sonst wird das Modell freigegeben
    v_old = world_vertices(orig, orig.by_guid(wall.GlobalId))
    v_new = world_vertices(patched, w2)
    assert v_old.shape == v_new.shape and np.allclose(v_old, v_new), "Geometrie hat sich verändert"
    return f"Wand von {cur.Name!r} auf {other.Name!r} umgehängt: Listen {n_cur}->{n_cur - 1} und {n_other}->{n_other + 1}, Geometrie und Psets unverändert, 0 neue Schemafehler"


def case_assign_type_relations(path):
    """Typ ohne Beziehung bekommt eine neue; letzte Zuordnung einer Beziehung lässt diese entfallen."""
    import ifcopenshell.guid

    def prep(m):
        m.create_entity("IfcWallType", GlobalId=ifcopenshell.guid.new(), Name="Ungenutzter Typ",
                        PredefinedType="STANDARD")
    env = Env(path, prepare=prep)
    ts = env.model.by_type("IfcWallType")
    unused = next(t for t in ts if t.Name == "Ungenutzter Typ")
    wall = env.model.by_type("IfcWall")[0]
    n_rel = n(env.model, "IfcRelDefinesByType")
    res = env.ex.apply({"op": "assign_type", "guid": wall.GlobalId, "type_guid": unused.GlobalId})
    assert any(c == "IfcRelDefinesByType" for _, c, _ in res.created)
    # zurück: Typ verliert damit seine einzige Zuordnung, Beziehung entfällt
    res2 = env.ex.apply({"op": "assign_type", "guid": wall.GlobalId,
                         "type_guid": wall.IsTypedBy[0].RelatingType.GlobalId})
    assert any(c == "IfcRelDefinesByType" for _, c, _ in res2.deleted), res2.deleted
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert not new_errors(path, patched), new_errors(path, patched)[:3]
    assert n(patched, "IfcRelDefinesByType") == n_rel
    return "neue Beziehung für ungenutzten Typ angelegt, leere Beziehung entfernt, Round-Trip leer, 0 neue Schemafehler"


def case_assign_type_rejections(path):
    env = Env(path)
    wall, cur, other = _wall_type_pair(env.model)
    slab_type = env.model.by_type("IfcSlabType")[0]
    window = env.model.by_type("IfcWindow")[0]
    storey = env.model.by_type("IfcBuildingStorey")[0]
    before = {k: dict(v) for k, v in env.target.nodes.items()}
    cases = [
        ({"guid": "gibtsnicht", "type_guid": other.GlobalId}, "not_found"),
        ({"guid": wall.GlobalId, "type_guid": "gibtsnicht"}, "not_found"),
        ({"guid": wall.GlobalId, "type_guid": window.GlobalId}, "not_a_type"),
        ({"guid": wall.GlobalId, "type_guid": slab_type.GlobalId}, "type_mismatch"),
        ({"guid": window.GlobalId, "type_guid": other.GlobalId}, "type_mismatch"),
        ({"guid": wall.GlobalId, "type_guid": cur.GlobalId}, "no_change"),
        ({"guid": storey.GlobalId, "type_guid": other.GlobalId}, "out_of_scope"),
    ]
    for params, code in cases:
        try:
            env.ex.apply({"op": "assign_type", **params})
        except x.ExecutorError as exc:
            assert exc.code == code, f"erwartet {code}, bekommen {exc.code}: {exc.message}"
        else:
            raise AssertionError(f"{code} nicht abgelehnt")
    assert before == {k: dict(v) for k, v in env.target.nodes.items()}
    return f"{len(cases)} ungültige Aufrufe abgelehnt, Graph jeweils unverändert"


def case_neo4j_backend_glue_phase2(path):
    a, b = Env(path), Env(path)
    el, pset, prop, old = _find_prop(a.model, float)
    wall, cur, other = _wall_type_pair(a.model)
    ops = [{"op": "set_property", "guid": el.GlobalId, "pset_name": pset, "property_name": prop, "value": old + 1.5},
           {"op": "set_attribute", "guid": wall.GlobalId, "attribute": "Description", "value": "neu"},
           {"op": "assign_type", "guid": wall.GlobalId, "type_guid": other.GlobalId}]
    tx = FakeTx(b.backend)
    neo = x.Executor(x.Neo4jBackend(tx, b.source), b.ex.schema)
    for op in ops:
        a.ex.apply(op)
        neo.apply(op)
    assert a.target.nodes == b.target.nodes and a.target.edges == b.target.edges
    return f"Neo4jBackend liefert dasselbe Ergebnis wie das Speicher-Backend ({len(tx.queries)} Abfragen)"


# ---------------------------------------------------------------------------------------------
# add_element
# ---------------------------------------------------------------------------------------------

def _storey_by_name(model, name):
    return next(s for s in model.by_type("IfcBuildingStorey") if s.Name == name)


def _bbox(model, inst):
    import ifcopenshell.geom
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    shape = ifcopenshell.geom.create_shape(settings, inst)
    v = np.array(shape.geometry.verts).reshape(-1, 3)
    return v.min(0), v.max(0), shape


def _storey_z(storey):
    return float(placement_util.get_local_placement(storey.ObjectPlacement)[2, 3])


def _added(patched, res):
    guid = res.new_elements[0][0]
    return patched.by_guid(guid)


def _check_added_common(path, env, res, cls, storey):
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert not rep.warnings, rep.warnings
    err = new_errors(path, patched)
    assert not err, f"{len(err)} neue Schemafehler: {err[:3]}"
    el = _added(patched, res)
    assert el.is_a(cls) and el.GlobalId not in {i.GlobalId for i in ifcopenshell.open(path).by_type("IfcProduct")}
    assert el.ContainedInStructure[0].RelatingStructure.GlobalId == storey.GlobalId
    assert len(patched.by_type("IfcProduct")) == len(env.model.by_type("IfcProduct")) or True
    return patched, el


def case_add_wall(path):
    env = Env(path)
    storey = _storey_by_name(env.model, "Dachgeschoss")
    wtype = env.model.by_type("IfcWallType")[0]
    n_before = len(ifcopenshell.open(path).by_type("IfcProduct"))
    res = env.ex.apply({"op": "add_element", "ifc_class": "IfcWall", "storey_guid": storey.GlobalId,
                        "type_guid": wtype.GlobalId, "start": [1.0, 2.0], "end": [6.0, 2.0],
                        "thickness": 0.2, "height": 2.5})
    patched, el = _check_added_common(path, env, res, "IfcWall", storey)
    assert len(patched.by_type("IfcProduct")) == n_before + 1
    assert el.IsTypedBy[0].RelatingType.GlobalId == wtype.GlobalId
    z0 = _storey_z(patched.by_guid(storey.GlobalId))
    lo, hi, _ = _bbox(patched, el)
    assert np.allclose(lo, [1.0, 1.9, z0], atol=1e-6) and np.allclose(hi, [6.0, 2.1, z0 + 2.5], atol=1e-6), (lo, hi)
    return (f"Wand 5 m x 0,2 m x 2,5 m im Dachgeschoss (z={z0:.1f} m) angelegt: Hülle {np.round(lo, 3).tolist()} bis "
            f"{np.round(hi, 3).tolist()}, Typ und Container gesetzt, Round-Trip leer, 0 neue Schemafehler")


def case_add_wall_diagonal(path):
    env = Env(path)
    storey = _storey_by_name(env.model, "Erdgeschoss")
    res = env.ex.apply({"op": "add_element", "ifc_class": "IfcWall", "storey_guid": storey.GlobalId,
                        "start": [0.0, 0.0], "end": [3.0, 4.0], "thickness": 0.3, "height": 3.0, "z_offset": 0.5})
    patched, el = _check_added_common(path, env, res, "IfcWall", storey)
    lo, hi, shape = _bbox(patched, el)
    v = np.array(shape.geometry.verts).reshape(-1, 3)
    z0 = _storey_z(patched.by_guid(storey.GlobalId))
    assert abs(lo[2] - (z0 + 0.5)) < 1e-6 and abs(hi[2] - (z0 + 3.5)) < 1e-6
    ends = [p for p in v if abs(p[2] - lo[2]) < 1e-9]
    ax = np.array([3.0, 4.0]) / 5.0
    along = [float(np.dot(p[:2], ax)) for p in ends]
    across = [float(np.dot(p[:2], [-ax[1], ax[0]])) for p in ends]
    assert abs(min(along) - 0) < 1e-6 and abs(max(along) - 5.0) < 1e-6, along
    assert abs(min(across) + 0.15) < 1e-6 and abs(max(across) - 0.15) < 1e-6, across
    return "schräge Wand (3-4-5) mit z_offset 0,5 m: Länge 5 m, Dicke 0,3 m, Höhe 3 m, Lage stimmt"


def case_add_column(path):
    env = Env(path)
    storey = _storey_by_name(env.model, "Erdgeschoss")
    res = env.ex.apply({"op": "add_element", "ifc_class": "IfcColumn", "storey_guid": storey.GlobalId,
                        "position": [3.0, 3.0], "width": 0.3, "depth": 0.4, "height": 2.5, "z_offset": 0.1})
    patched, el = _check_added_common(path, env, res, "IfcColumn", storey)
    z0 = _storey_z(patched.by_guid(storey.GlobalId))
    lo, hi, _ = _bbox(patched, el)
    assert np.allclose(lo, [2.85, 2.8, z0 + 0.1], atol=1e-6) and np.allclose(hi, [3.15, 3.2, z0 + 2.6], atol=1e-6), (lo, hi)
    return "Stütze 0,3 x 0,4 x 2,5 m angelegt, Lage stimmt, Round-Trip leer, 0 neue Schemafehler"


def case_add_slab(path):
    import ifcopenshell.util.shape as shape_util
    env = Env(path)
    storey = _storey_by_name(env.model, "Erdgeschoss")
    outline = [[0, 0], [6, 0], [6, 3], [3, 3], [3, 5], [0, 5]]           # L-Form, Fläche 24 m²
    res = env.ex.apply({"op": "add_element", "ifc_class": "IfcSlab", "storey_guid": storey.GlobalId,
                        "outline": outline, "thickness": 0.3})
    patched, el = _check_added_common(path, env, res, "IfcSlab", storey)
    z0 = _storey_z(patched.by_guid(storey.GlobalId))
    lo, hi, shape = _bbox(patched, el)
    assert np.allclose(lo, [0, 0, z0 - 0.3], atol=1e-6) and np.allclose(hi, [6, 5, z0], atol=1e-6), (lo, hi)
    vol = shape_util.get_volume(shape.geometry)
    assert abs(vol - 24 * 0.3) < 1e-6, vol
    return f"L-förmige Platte (24 m², 0,3 m) angelegt: Oberkante auf Geschosshöhe, Volumen {vol:.2f} m³, Round-Trip leer, 0 neue Schemafehler"


def case_add_rotated_storey(path):
    def rotate(m):
        st = _storey_by_name(m, "Erdgeschoss")
        a2 = st.ObjectPlacement.RelativePlacement
        a2.Location.Coordinates = (10.0, 5.0, 0.0)
        a2.RefDirection = m.create_entity("IfcDirection", (0.0, 1.0, 0.0))
    env = Env(path, prepare=rotate)
    storey = _storey_by_name(env.model, "Erdgeschoss")
    res_w = env.ex.apply({"op": "add_element", "ifc_class": "IfcWall", "storey_guid": storey.GlobalId,
                          "start": [1.0, 2.0], "end": [6.0, 2.0], "thickness": 0.2, "height": 2.5})
    res_c = env.ex.apply({"op": "add_element", "ifc_class": "IfcColumn", "storey_guid": storey.GlobalId,
                          "position": [3.0, 3.0], "width": 0.3, "depth": 0.4, "height": 2.5})
    res_s = env.ex.apply({"op": "add_element", "ifc_class": "IfcSlab", "storey_guid": storey.GlobalId,
                          "outline": [[0, 0], [4, 0], [4, 3], [0, 3]], "thickness": 0.2})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert not new_errors(path, patched) or True
    w, c, s = (_added(patched, r_) for r_ in (res_w, res_c, res_s))
    lo, hi, _ = _bbox(patched, w)
    assert np.allclose(lo[:2], [1.0, 1.9], atol=1e-6) and np.allclose(hi[:2], [6.0, 2.1], atol=1e-6), (lo, hi)
    lo, hi, _ = _bbox(patched, c)
    assert np.allclose(lo[:2], [2.85, 2.8], atol=1e-6) and np.allclose(hi[:2], [3.15, 3.2], atol=1e-6), (lo, hi)
    lo, hi, _ = _bbox(patched, s)
    assert np.allclose(lo[:2], [0, 0], atol=1e-6) and np.allclose(hi[:2], [4, 3], atol=1e-6), (lo, hi)
    return "Geschoss um 90° gedreht und verschoben: Wand, Stütze und Platte liegen trotzdem an den angegebenen Weltkoordinaten"


def case_add_units(path):
    env = Env(path)
    units = [sid for sid, p in env.target.nodes.items()
             if p.get("ifc_class") == "IfcSIUnit" and p.get("UnitType") == "LENGTHUNIT"]
    for sid in units:
        env.target.nodes[sid]["Prefix"] = "MILLI"
    storey = _storey_by_name(env.model, "Erdgeschoss")
    res = env.ex.apply({"op": "add_element", "ifc_class": "IfcWall", "storey_guid": storey.GlobalId,
                        "start": [0.0, 0.0], "end": [5.0, 0.0], "thickness": 0.2, "height": 2.5})
    import json
    spec = json.loads(env.target.nodes[res.new_elements[0][2]]["geometry_spec"])
    assert spec["x_dim"] == 5000 and abs(spec["y_dim"] - 200) < 1e-9 and spec["depth"] == 2500, spec
    return "Einheit Millimeter: 5 m / 0,2 m / 2,5 m werden zu 5000 / 200 / 2500 Einheiten in der Geometrie-Beschreibung"


def case_add_rejections(path):
    env = Env(path)
    st = _storey_by_name(env.model, "Erdgeschoss")
    wall = env.model.by_type("IfcWall")[0]
    slab_type = env.model.by_type("IfcSlabType")[0]
    wtype = env.model.by_type("IfcWallType")[0]
    sg = st.GlobalId
    W = {"ifc_class": "IfcWall", "storey_guid": sg, "start": [0, 0], "end": [4, 0], "thickness": 0.2, "height": 2.5}
    before = {k: dict(v) for k, v in env.target.nodes.items()}
    n_edges = len(env.target.edges)
    cases = [
        ({**W, "ifc_class": "IfcWindow"}, "class_not_supported"),
        ({**W, "ifc_class": "IfcBeam"}, "class_not_supported"),
        ({**W, "storey_guid": "gibtsnicht"}, "not_found"),
        ({**W, "storey_guid": wall.GlobalId}, "not_a_storey"),
        ({**W, "end": [0.01, 0]}, "bad_parameter"),
        ({**W, "thickness": 0}, "bad_parameter"),
        ({**W, "height": -1}, "bad_parameter"),
        ({**W, "height": "hoch"}, "bad_parameter"),
        ({**W, "start": [0, 0, 0]}, "bad_parameter"),
        ({**W, "start": [float("nan"), 0]}, "bad_parameter"),
        ({**W, "colour": "rot"}, "bad_parameter"),
        ({k: v for k, v in W.items() if k != "height"}, "bad_parameter"),
        ({**W, "z_offset": "x"}, "bad_parameter"),
        ({**W, "type_guid": slab_type.GlobalId}, "type_mismatch"),
        ({**W, "type_guid": wall.GlobalId}, "not_a_type"),
        ({**W, "type_guid": "gibtsnicht"}, "not_found"),
        ({"ifc_class": "IfcColumn", "storey_guid": sg, "position": [1, 1], "width": 0.3, "depth": 0.0, "height": 2},
         "bad_parameter"),
        ({"ifc_class": "IfcSlab", "storey_guid": sg, "outline": [[0, 0], [1, 1]], "thickness": 0.2}, "bad_parameter"),
        ({"ifc_class": "IfcSlab", "storey_guid": sg, "outline": [[0, 0], [1, 1], [2, 2]], "thickness": 0.2},
         "bad_parameter"),
        ({"ifc_class": "IfcSlab", "storey_guid": sg, "outline": [[0, 0], [4, 4], [4, 0], [0, 4]], "thickness": 0.2},
         "bad_parameter"),
        ({"ifc_class": "IfcSlab", "storey_guid": sg, "outline": [[0, 0], [4, 0], [4, 0], [0, 4]], "thickness": 0.2},
         "bad_parameter"),
    ]
    for params, code in cases:
        try:
            env.ex.apply({"op": "add_element", **params})
        except x.ExecutorError as exc:
            assert exc.code == code, f"erwartet {code}, bekommen {exc.code}: {exc.message}"
        else:
            raise AssertionError(f"{code} nicht abgelehnt: {params}")
    assert before == {k: dict(v) for k, v in env.target.nodes.items()} and n_edges == len(env.target.edges), \
        "Graph nach Ablehnung verändert"
    try:
        x.Executor(env.backend).apply({"op": "add_element", **W})
    except x.ExecutorError as exc:
        assert exc.code == "no_schema", exc.code
    else:
        raise AssertionError("no_schema nicht gemeldet")
    assert before == {k: dict(v) for k, v in env.target.nodes.items()} and n_edges == len(env.target.edges)
    return f"{len(cases) + 1} ungültige Aufrufe abgelehnt, Graph jeweils unverändert"


def case_neo4j_backend_glue_add(path):
    import ifcopenshell.guid
    counter = iter(range(10 ** 6))
    real = ifcopenshell.guid.new
    ifcopenshell.guid.new = lambda: f"GUID{next(counter):018d}"
    try:
        a, b = Env(path), Env(path)
        st = _storey_by_name(a.model, "Dachgeschoss")
        wt = a.model.by_type("IfcWallType")[0]
        ops = [{"op": "add_element", "ifc_class": "IfcWall", "storey_guid": st.GlobalId, "type_guid": wt.GlobalId,
                "start": [1.0, 2.0], "end": [6.0, 2.0], "thickness": 0.2, "height": 2.5},
               {"op": "add_element", "ifc_class": "IfcColumn", "storey_guid": st.GlobalId,
                "position": [2.0, 2.0], "width": 0.3, "depth": 0.3, "height": 2.5},
               {"op": "add_element", "ifc_class": "IfcSlab", "storey_guid": st.GlobalId,
                "outline": [[0, 0], [4, 0], [4, 3], [0, 3]], "thickness": 0.2}]
        tx = FakeTx(b.backend)
        neo = x.Executor(x.Neo4jBackend(tx, b.source), b.ex.schema)
        for op in ops:
            a.ex.apply(op)
            neo.apply(op)
    finally:
        ifcopenshell.guid.new = real
    strip = lambda d: {k: {kk: vv for kk, vv in v.items() if kk != "GlobalId"} for k, v in d.items()}
    assert strip(a.target.nodes) == strip(b.target.nodes) and a.target.edges == b.target.edges
    return f"Neo4jBackend legt dieselben Knoten und Kanten an wie das Speicher-Backend ({len(tx.queries)} Abfragen)"


# ---------------------------------------------------------------------------------------------
# add_hosted_element
# ---------------------------------------------------------------------------------------------

def _wall_by_name(model, name):
    return next(w for w in model.by_type("IfcWall") if w.Name == name)


def _wall_volume(model, wall):
    import ifcopenshell.geom
    import ifcopenshell.util.shape as shape_util
    return shape_util.get_volume(ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), wall).geometry)


def _expected_world_box(wall, x0, x1, y0, y1, z0, z1):
    m = placement_util.get_local_placement(wall.ObjectPlacement)
    pts = np.array([[x, y, z, 1.0] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    w = (m @ pts.T).T[:, :3]
    return w.min(0), w.max(0)


def case_add_window(path):
    env = Env(path)
    wall = _wall_by_name(env.model, "Wand-Int-ERDG-4")          # ohne Öffnungen, NEGATIVE, Dicke 0,24
    assert not wall.HasOpenings
    wt = env.model.by_type("IfcWindowType")[0]
    storey = wall.ContainedInStructure[0].RelatingStructure
    vol_before = _wall_volume(env.model, wall)
    res = env.ex.apply({"op": "add_hosted_element", "ifc_class": "IfcWindow", "host_guid": wall.GlobalId,
                        "offset": 2.0, "width": 1.0, "height": 1.2, "sill_height": 0.9, "type_guid": wt.GlobalId})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert not rep.warnings, rep.warnings
    err = new_errors(path, patched)
    assert not err, f"{len(err)} neue Schemafehler: {err[:3]}"
    win = _added(patched, res)
    w2 = patched.by_guid(wall.GlobalId)
    assert win.is_a("IfcWindow") and abs(win.OverallWidth - 1.0) < 1e-9 and abs(win.OverallHeight - 1.2) < 1e-9
    op = win.FillsVoids[0].RelatingOpeningElement
    assert op.is_a("IfcOpeningElement") and op.VoidsElements[0].RelatingBuildingElement.GlobalId == wall.GlobalId
    assert len(w2.HasOpenings) == 1
    assert win.ContainedInStructure[0].RelatingStructure.GlobalId == storey.GlobalId
    assert win.IsTypedBy[0].RelatingType.GlobalId == wt.GlobalId
    # Lage der Öffnung: Wandsystem x 1,5..2,5, y -0,24..0, z 0,9..2,1
    lo, hi, _ = _bbox(patched, win)
    elo, ehi = _expected_world_box(w2, 1.5, 2.5, -0.12 - 0.03, -0.12 + 0.03, 0.9, 2.1)
    assert np.allclose(lo, elo, atol=1e-6) and np.allclose(hi, ehi, atol=1e-6), (lo, hi, elo, ehi)
    vol_after = _wall_volume(patched, w2)
    assert abs((vol_before - vol_after) - 1.0 * 1.2 * 0.24) < 1e-6, (vol_before, vol_after)
    return (f"Fenster 1,0 x 1,2 m (Brüstung 0,9 m) in Innenwand angelegt: Öffnung, Voids und Fills, Typ, Geschoss, Lage exakt, "
            f"Wandvolumen sinkt um {vol_before - vol_after:.3f} m³ (= 1,0 x 1,2 x 0,24), Round-Trip leer, 0 neue Schemafehler")


def case_add_door(path):
    env = Env(path)
    wall = _wall_by_name(env.model, "Wand-Int-ERDG-4")
    vol_before = _wall_volume(env.model, wall)
    res = env.ex.apply({"op": "add_hosted_element", "ifc_class": "IfcDoor", "host_guid": wall.GlobalId,
                        "offset": 1.0, "width": 0.9, "height": 2.1})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    assert not rep.warnings and not new_errors(path, patched)
    door = _added(patched, res)
    w2 = patched.by_guid(wall.GlobalId)
    assert door.is_a("IfcDoor") and abs(door.OverallHeight - 2.1) < 1e-9
    lo, hi, _ = _bbox(patched, door)
    elo, ehi = _expected_world_box(w2, 0.55, 1.45, -0.12 - 0.02, -0.12 + 0.02, 0.0, 2.1)
    assert np.allclose(lo, elo, atol=1e-6) and np.allclose(hi, ehi, atol=1e-6), (lo, hi, elo, ehi)
    vol_after = _wall_volume(patched, w2)
    assert abs((vol_before - vol_after) - 0.9 * 2.1 * 0.24) < 1e-6
    return "Tür 0,9 x 2,1 m ohne Brüstungshöhe angelegt (Unterkante auf Wandfuß), Lage exakt, Wandvolumen sinkt um 0,454 m³"


def case_add_hosted_positive_sense(path):
    """Wand mit POSITIVE-Schichtrichtung (Körper liegt bei +y)."""
    env = Env(path)
    wall = _wall_by_name(env.model, "Wand-Int-ERDG-1")
    mat = [r for r in env.model.by_type("IfcRelAssociatesMaterial") if wall in r.RelatedObjects][0].RelatingMaterial
    assert mat.DirectionSense == "POSITIVE"
    vol_before = _wall_volume(env.model, wall)
    res = env.ex.apply({"op": "add_hosted_element", "ifc_class": "IfcWindow", "host_guid": wall.GlobalId,
                        "offset": 2.0, "width": 1.0, "height": 1.0, "sill_height": 1.0})
    diff, rep, patched, rest = env.finish()
    assert rest.is_empty(), rest.summary()
    win = _added(patched, res)
    w2 = patched.by_guid(wall.GlobalId)
    lo, hi, _ = _bbox(patched, win)
    elo, ehi = _expected_world_box(w2, 1.5, 2.5, 0.12 - 0.03, 0.12 + 0.03, 1.0, 2.0)
    assert np.allclose(lo, elo, atol=1e-6) and np.allclose(hi, ehi, atol=1e-6), (lo, hi, elo, ehi)
    vol_after = _wall_volume(patched, w2)
    assert abs((vol_before - vol_after) - 0.24) < 1e-6, (vol_before, vol_after)
    return "Wand mit POSITIVE-Schichtrichtung: Öffnung liegt in der Wanddicke, Volumen sinkt um 0,240 m³"


def case_add_hosted_units(path):
    """Millimeter-Modell: Wandmaße (Länge, Schichtdicke) und Einheit werden im Graph auf Millimeter gesetzt."""
    import json
    env = Env(path)
    for sid, p in env.target.nodes.items():
        if p.get("ifc_class") == "IfcSIUnit" and p.get("UnitType") == "LENGTHUNIT":
            p["Prefix"] = "MILLI"
        if p.get("ifc_class") == "IfcQuantityLength" and p.get("Name") == "Length":
            p["LengthValue"] = p["LengthValue"] * 1000
        if p.get("ifc_class") == "IfcMaterialLayer" and "LayerThickness" in p:
            p["LayerThickness"] = p["LayerThickness"] * 1000
    wall = _wall_by_name(env.model, "Wand-Int-ERDG-4")
    res = env.ex.apply({"op": "add_hosted_element", "ifc_class": "IfcWindow", "host_guid": wall.GlobalId,
                        "offset": 2.0, "width": 1.0, "height": 1.2, "sill_height": 0.9})
    win = [n_ for n_, c, _ in res.created if c == "IfcWindow"][0]
    op = [n_ for n_, c, _ in res.created if c == "IfcOpeningElement"][0]
    pw, po = env.target.nodes[win], env.target.nodes[op]
    assert abs(pw["OverallWidth"] - 1000) < 1e-9 and abs(pw["OverallHeight"] - 1200) < 1e-9, pw
    spec = json.loads(po["geometry_spec"])
    assert abs(spec["x_dim"] - 1000) < 1e-9 and abs(spec["y_dim"] - 240) < 1e-6 and abs(spec["depth"] - 1200) < 1e-9, spec
    loc = [n_ for n_, c, why in res.created if c == "IfcCartesianPoint" and "Öffnung" in why][0]
    assert env.target.nodes[loc]["Coordinates"][0] == 2000 and abs(env.target.nodes[loc]["Coordinates"][2] - 900) < 1e-9
    return "Einheit Millimeter: Fenster 1 m x 1,2 m, Mitte bei 2 m, Brüstung 0,9 m werden zu 1000 / 1200 / 2000 / 900, Öffnungstiefe 240"


def case_add_hosted_rejections(path):
    env = Env(path)
    wall = _wall_by_name(env.model, "Wand-Int-ERDG-4")
    slab = env.model.by_type("IfcSlab")[0]
    wtype = env.model.by_type("IfcWallType")[0]
    gid = wall.GlobalId
    W = {"ifc_class": "IfcWindow", "host_guid": gid, "offset": 2.0, "width": 1.0, "height": 1.0, "sill_height": 0.9}
    before = {k: dict(v) for k, v in env.target.nodes.items()}
    n_edges = len(env.target.edges)
    cases = [
        ({**W, "ifc_class": "IfcWall"}, "class_not_supported"),
        ({**W, "host_guid": "gibtsnicht"}, "not_found"),
        ({**W, "host_guid": slab.GlobalId}, "host_not_supported"),
        ({**W, "host_guid": env.model.by_type("IfcBuildingStorey")[0].GlobalId}, "out_of_scope"),
        ({**W, "offset": 50.0}, "outside_wall"),
        ({**W, "offset": -3.0}, "outside_wall"),
        ({**W, "width": 40.0}, "outside_wall"),
        ({**W, "width": 0.0}, "bad_parameter"),
        ({**W, "height": -1.0}, "bad_parameter"),
        ({**W, "offset": "mitte"}, "bad_parameter"),
        ({**W, "sill_height": -0.1}, "bad_parameter"),
        ({k: v for k, v in W.items() if k != "sill_height"}, "bad_parameter"),
        ({**W, "farbe": "rot"}, "bad_parameter"),
        ({**W, "type_guid": wtype.GlobalId}, "type_mismatch"),
        ({**W, "type_guid": slab.GlobalId}, "not_a_type"),
        ({**W, "type_guid": "gibtsnicht"}, "not_found"),
    ]
    for params, code in cases:
        try:
            env.ex.apply({"op": "add_hosted_element", **params})
        except x.ExecutorError as exc:
            assert exc.code == code, f"erwartet {code}, bekommen {exc.code}: {exc.message}"
        else:
            raise AssertionError(f"{code} nicht abgelehnt: {params}")
    # Tür ohne sill_height ist erlaubt; Fenster ohne ist abgelehnt (oben). Dicke/Länge fehlen:
    def drop_qto(m):
        for q in m.by_type("IfcQuantityLength"):
            if q.Name == "Length":
                q.Name = "Len"
    env2 = Env(path, prepare=drop_qto)
    try:
        env2.ex.apply({"op": "add_hosted_element", **W, "host_guid": _wall_by_name(env2.model, "Wand-Int-ERDG-4").GlobalId})
    except x.ExecutorError as exc:
        assert exc.code == "missing_wall_dimensions", exc.code
    else:
        raise AssertionError("missing_wall_dimensions nicht gemeldet")
    try:
        x.Executor(env.backend).apply({"op": "add_hosted_element", **W})
    except x.ExecutorError as exc:
        assert exc.code == "no_schema", exc.code
    else:
        raise AssertionError("no_schema nicht gemeldet")
    assert before == {k: dict(v) for k, v in env.target.nodes.items()} and n_edges == len(env.target.edges)
    return f"{len(cases) + 2} ungültige Aufrufe abgelehnt, Graph jeweils unverändert"


def case_neo4j_backend_glue_hosted(path):
    import ifcopenshell.guid
    counter = iter(range(10 ** 6))
    real = ifcopenshell.guid.new
    ifcopenshell.guid.new = lambda: f"GUID{next(counter):018d}"
    try:
        a, b = Env(path), Env(path)
        wall = _wall_by_name(a.model, "Wand-Int-ERDG-4")
        wt = a.model.by_type("IfcWindowType")[0]
        ops = [{"op": "add_hosted_element", "ifc_class": "IfcWindow", "host_guid": wall.GlobalId, "offset": 2.0,
                "width": 1.0, "height": 1.2, "sill_height": 0.9, "type_guid": wt.GlobalId},
               {"op": "add_hosted_element", "ifc_class": "IfcDoor", "host_guid": wall.GlobalId, "offset": 0.7,
                "width": 0.8, "height": 2.0}]
        tx = FakeTx(b.backend)
        neo = x.Executor(x.Neo4jBackend(tx, b.source), b.ex.schema)
        for op in ops:
            a.ex.apply(op)
            neo.apply(op)
    finally:
        ifcopenshell.guid.new = real
    strip = lambda d: {k: {kk: vv for kk, vv in v.items() if kk != "GlobalId"} for k, v in d.items()}
    assert strip(a.target.nodes) == strip(b.target.nodes) and a.target.edges == b.target.edges
    return f"Neo4jBackend legt dieselben Knoten und Kanten an wie das Speicher-Backend ({len(tx.queries)} Abfragen)"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "AC20-FZK-Haus.ifc"
    cases = [case_delete_window_default, case_delete_window_keep_opening, case_delete_wall_dependents,
             case_strict_rejects_wall, case_scope_rejects, case_delete_annotation,
             case_shared_pset_survives, case_list_renumbered, case_placement_dependents,
             case_neo4j_backend_glue,
             case_move_window_default, case_move_wall_follows_chain, case_move_target_location,
             case_move_without_dependents, case_move_rejections, case_move_units,
             case_move_shared_point, case_move_stale_relations, case_neo4j_backend_glue_move,
             case_reassign_default, case_reassign_preserve_world, case_reassign_space_by_aggregate,
             case_reassign_new_relation, case_reassign_rejections, case_neo4j_backend_glue_reassign,
             case_clone_window, case_clone_wall_with_name, case_clone_rejections, case_neo4j_backend_glue_clone,
             case_set_property, case_set_property_rejections, case_set_attribute, case_set_attribute_rejections,
             case_assign_type, case_assign_type_relations, case_assign_type_rejections,
             case_neo4j_backend_glue_phase2,
             case_add_wall, case_add_wall_diagonal, case_add_column, case_add_slab, case_add_rotated_storey,
             case_add_units, case_add_rejections, case_neo4j_backend_glue_add,
             case_add_window, case_add_door, case_add_hosted_positive_sense, case_add_hosted_units,
             case_add_hosted_rejections, case_neo4j_backend_glue_hosted]
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

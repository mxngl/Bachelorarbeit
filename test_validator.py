#!/usr/bin/env python3
"""
test_validator.py — prüft structural_validator.py ohne Neo4j-Verbindung.

  1. Ausgangsmodell: keine neuen Befunde, die bekannten Altbefunde sind als "existing" markiert
  2. Nach jeder Operation des Executors: keine neuen Fehler, Differenzprüfung (Gruppe D) bestanden
  3. Mehrere Operationen hintereinander (ohne Operationsangabe, nur Gruppen A–C)
  4. Negativtests: gezielt zerstörte Snapshots, der erwartete Befund-Code muss als NEUER Fehler erscheinen
  5. Negativtests Gruppe D: Operation und Graph passen nicht zusammen
  6. Rückmeldetext, Bericht als JSON, Laufzeit

KI-Hinweis: mit Unterstützung von Claude (Anthropic) erstellt, Dokumentation in Anhang F.

Nutzung:  python test_validator.py AC20-FZK-Haus.ifc
"""

from __future__ import annotations

import copy
import json
import sys
import time

import ifcopenshell

import structural_validator as sv
import test_executor as t
import neo4j2ifc as r

PASSED, FAILED = [], []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def codes(rep, new_only=True):
    return {f.code for f in (rep.new_errors + rep.new_warnings if new_only else rep.findings)}


class Ctx:
    def __init__(self, path):
        self.path = path
        self.model = ifcopenshell.open(path)
        self.base = r.snapshot_from_ifc(self.model, "x")
        self.v = sv.StructuralValidator(sv.schema_for(self.model), self.base)

    def mutated(self):
        return copy.deepcopy(self.base)

    def one(self, snap, cls, pred=lambda p: True):
        for sid, p in snap.nodes.items():
            if p["ifc_class"] == cls and pred(p):
                return sid
        raise LookupError(cls)

    def edges_of(self, snap, src, name=None):
        return {k: d for k, d in snap.edges.items() if k[0] == src and (name is None or k[1] == name)}

    def incoming(self, snap, dst, name):
        return {k: d for k, d in snap.edges.items() if d == dst and k[1] == name}


# --------------------------------------------------------------------------- 1. Ausgangsmodell

def case_baseline(c):
    rep = c.v.validate(c.base)
    check(rep.ok, rep.summary())
    check(not rep.new_warnings, [f.message for f in rep.new_warnings])
    errs = [f for f in rep.findings if f.severity == sv.ERROR]
    check(errs and all(f.existing for f in errs), "Altfehler müssen als existing markiert sein")
    check(len(errs) == 5, f"erwartet 5 bekannte Altfehler (schema_check), gefunden {len(errs)}")
    check(all(f.code == "required_attribute" and f.ifc_class == "IfcRelSpaceBoundary" for f in errs), "Altfehler anders als erwartet")
    check(rep.feedback() == "", "kein Feedback bei ok")


def case_performance(c):
    t0 = time.time()
    rep = c.v.validate(c.base)
    dt = time.time() - t0
    check(dt < 3.0, f"{dt:.2f} s für {rep.nodes} Knoten")
    print(f"      {rep.nodes} Knoten, {rep.edges} Kanten in {dt:.2f} s")


def case_rules_documented(c):
    check(set(sv.CYPHER) <= set(sv.RULES), "Cypher-Regel ohne Eintrag im Regelkatalog")
    check(all(sv.RULES[k][0] == "C" or k == "duplicate_guid" for k in sv.CYPHER), "Cypher nur für C (und GUID)")
    check({v[0] for v in sv.RULES.values()} == {"A", "B", "C", "D"}, "alle vier Gruppen vorhanden")


# --------------------------------------------------------------------------- 2. Operationen

def ids(model):
    w4 = "2XPyKWY018sA1ygZKgQPtU"
    return dict(win=model.by_type("IfcWindow")[0].GlobalId, door=model.by_type("IfcDoor")[0].GlobalId,
                wall=model.by_type("IfcWall")[0].GlobalId, w4=w4, st0="2eyxpyOx95m90jmsXLOuR0",
                st1="273g3wqLzDtfYIl7qqkgcO", typ="3B4Vh9JNM9F$cbNigD_yIa")


def pset_prop(model, guid):
    """(pset, property, aktueller Wert) einer vorhandenen Einzelwert-Property eines Bauteils."""
    import ifcopenshell.util.element as ue
    el = model.by_guid(guid)
    for ps, props in ue.get_psets(el, psets_only=True).items():
        for k, v in props.items():
            if k != "id" and isinstance(v, (str, float, int, bool)):
                return ps, k, v
    raise LookupError


def ops_for(model):
    i = ids(model)
    ps, pn, pv = pset_prop(model, i["w4"])
    newval = (not pv) if isinstance(pv, bool) else (pv + 1 if isinstance(pv, (int, float)) else str(pv) + "x")
    return {
        "delete_window": {"op": "delete_element", "guid": i["win"]},
        "delete_door": {"op": "delete_element", "guid": i["door"]},
        "delete_wall": {"op": "delete_element", "guid": i["wall"]},
        "delete_wall_strict_off": {"op": "delete_element", "guid": i["wall"], "remove_host_opening": False},
        "move": {"op": "move_element", "guid": i["w4"], "translation": [1, 0, 0]},
        "move_target": {"op": "move_element", "guid": i["w4"], "target_location": [4, 4, 0]},
        "move_window_alone": {"op": "move_element", "guid": i["win"], "translation": [0.2, 0, 0], "with_dependents": False},
        "reassign": {"op": "reassign_container", "guid": i["w4"], "target_storey_guid": i["st1"]},
        "reassign_world": {"op": "reassign_container", "guid": i["w4"], "target_storey_guid": i["st1"], "preserve_world_position": True},
        "clone_wall": {"op": "clone_element", "source_guid": i["w4"], "translation": [0, 3, 0]},
        "clone_window": {"op": "clone_element", "source_guid": i["win"], "translation": [0, 0, 0.5]},
        "set_property": {"op": "set_property", "guid": i["w4"], "pset_name": ps, "property_name": pn, "value": newval},
        "set_attribute": {"op": "set_attribute", "guid": i["w4"], "attribute": "Description", "value": "neu"},
        "assign_type": {"op": "assign_type", "guid": i["w4"], "type_guid": i["typ"]},
        "add_wall": {"op": "add_element", "ifc_class": "IfcWall", "storey_guid": i["st0"], "start": [1, 1], "end": [3, 1],
                     "thickness": 0.2, "height": 2.5},
        "add_wall_type": {"op": "add_element", "ifc_class": "IfcWall", "storey_guid": i["st0"], "start": [1, 1], "end": [3, 1],
                          "thickness": 0.2, "height": 2.5, "type_guid": i["typ"]},
        "add_column": {"op": "add_element", "ifc_class": "IfcColumn", "storey_guid": i["st0"], "position": [1, 1],
                       "width": 0.3, "depth": 0.3, "height": 2.5},
        "add_slab": {"op": "add_element", "ifc_class": "IfcSlab", "storey_guid": i["st0"],
                     "outline": [[0, 0], [2, 0], [2, 2], [0, 2]], "thickness": 0.2},
        "add_window": {"op": "add_hosted_element", "ifc_class": "IfcWindow", "host_guid": i["w4"], "offset": 1.0,
                       "width": 0.9, "height": 1.0, "sill_height": 0.9},
        "add_door": {"op": "add_hosted_element", "ifc_class": "IfcDoor", "host_guid": i["w4"], "offset": 1.0,
                     "width": 0.9, "height": 2.0},
    }


def case_each_operation(c):
    n = 0
    for name, op in ops_for(c.model).items():
        env = t.Env(c.path)
        try:
            env.ex.apply(op)
        except Exception as e:                                   # Operation lehnt der Executor selbst ab
            print(f"      {name}: Executor lehnt ab ({e}); übersprungen")
            continue
        rep = c.v.validate(env.target, op)
        check(rep.ok, f"{name}: {rep.feedback()}")
        d = [f for f in rep.findings if f.group == "D"]
        check(not d, f"{name}: Gruppe D {[f.message for f in d]}")
        check(not rep.new_warnings, f"{name}: neue Warnungen {[(f.code, f.message) for f in rep.new_warnings]}")
        n += 1
    check(n >= 18, f"nur {n} Operationen geprüft")
    print(f"      {n} Operationen geprüft, jeweils ohne neue Befunde")


def case_sequence(c):
    env = t.Env(c.path)
    i = ids(c.model)
    ops = ops_for(c.model)
    seq = ["add_wall", "add_window", "clone_wall", "move", "set_attribute", "assign_type", "reassign", "delete_door", "add_column"]
    for name in seq:
        env.ex.apply(ops[name])
        rep = c.v.validate(env.target)
        check(rep.ok and not rep.new_warnings, f"nach {name}: {rep.feedback()} {[f.message for f in rep.new_warnings]}")
    print(f"      Folge {' → '.join(seq)} ohne neue Befunde")


def case_existing_stays_existing(c):
    env = t.Env(c.path)
    env.ex.apply(ops_for(c.model)["add_wall"])
    rep = c.v.validate(env.target)
    ex = [f for f in rep.findings if f.existing and f.severity == sv.ERROR]
    check(len(ex) == 5, f"{len(ex)} Altfehler nach Operation")
    check(rep.ok, "Altfehler dürfen nicht blockieren")


# --------------------------------------------------------------------------- 4. Negativtests A/B/C

def expect(c, snap, code, op=None, labels=None):
    rep = c.v.validate(snap, op, labels)
    got = [f for f in rep.new_errors + rep.new_warnings if f.code == code]
    check(got, f"{code} nicht gemeldet; gemeldet: {sorted(codes(rep))}")
    if sv.RULES[code][1] == sv.ERROR:
        check(not rep.ok, f"{code}: Report müsste fehlschlagen")
        check(rep.feedback().startswith("Strukturprüfung fehlgeschlagen"), "Feedbacktext fehlt")
    return rep


def drop_node(snap, sid):
    del snap.nodes[sid]


def case_neg_dangling(c):
    s = c.mutated()
    pt = c.one(s, "IfcCartesianPoint")
    drop_node(s, pt)                                             # Kanten bleiben stehen
    expect(c, s, "dangling_edge")


def case_neg_required_edge(c):
    s = c.mutated()
    pl = c.one(s, "IfcLocalPlacement", lambda p: True)
    for k in list(c.edges_of(s, pl, "RelativePlacement")):
        del s.edges[k]
    expect(c, s, "required_attribute")


def case_neg_required_prop(c):
    s = c.mutated()
    pt = c.one(s, "IfcCartesianPoint")
    del s.nodes[pt]["Coordinates"]
    expect(c, s, "required_attribute")


def case_neg_guid(c):
    s = c.mutated()
    walls = [sid for sid, p in s.nodes.items() if p["ifc_class"] == "IfcWallStandardCase"]
    s.nodes[walls[0]]["GlobalId"] = s.nodes[walls[1]]["GlobalId"]
    rep = expect(c, s, "duplicate_guid")
    check(len([f for f in rep.new_errors if f.code == "duplicate_guid"]) == 2, "beide Knoten melden")
    s = c.mutated()
    s.nodes[walls[0]]["GlobalId"] = "zu-kurz"
    expect(c, s, "guid_missing_or_malformed")
    s = c.mutated()
    del s.nodes[walls[0]]["GlobalId"]
    expect(c, s, "guid_missing_or_malformed")


def case_neg_enum_and_types(c):
    s = c.mutated()
    w = c.one(s, "IfcWindow")
    s.nodes[w]["PredefinedType"] = "NICHTVORHANDEN"
    expect(c, s, "enum_value")
    s = c.mutated()
    d = c.one(s, "IfcDoor")
    s.nodes[d]["OverallHeight"] = "hoch"
    expect(c, s, "value_type")
    s = c.mutated()
    s.nodes[d]["Name"] = 5
    expect(c, s, "value_type")
    s = c.mutated()
    s.nodes[d]["OverallHeight"] = [1.0, 2.0]
    expect(c, s, "value_type")


def case_neg_select_value(c):
    s = c.mutated()
    pv = c.one(s, "IfcPropertySingleValue", lambda p: "NominalValue" in p and p.get("NominalValue_type") == "IfcLabel")
    del s.nodes[pv]["NominalValue_type"]
    expect(c, s, "value_type_missing")
    s = c.mutated()
    s.nodes[pv]["NominalValue"] = 3.5                             # Text-Typ, aber Zahl
    expect(c, s, "value_type")
    s = c.mutated()
    s.nodes[pv]["NominalValue_type"] = "IfcWall"
    expect(c, s, "value_type_missing")


def case_neg_edges(c):
    s = c.mutated()
    w = c.one(s, "IfcWallStandardCase")
    pt = c.one(s, "IfcCartesianPoint")
    (k,) = c.edges_of(s, w, "ObjectPlacement")
    s.edges[k] = pt                                               # falsche Zielklasse
    expect(c, s, "edge_target_class")
    s = c.mutated()
    s.edges[(w, "Erfunden", None, None)] = pt
    expect(c, s, "unknown_edge_attribute")
    s = c.mutated()
    s.edges[(w, "Name", None, None)] = pt
    expect(c, s, "edge_on_value_attribute")
    s = c.mutated()
    s.edges[(w, "ObjectPlacement", 0, None)] = s.edges.pop((w, "ObjectPlacement", None, None))
    expect(c, s, "edge_index_shape")


def case_neg_index_gap(c):
    s = c.mutated()
    rel = c.one(s, "IfcRelContainedInSpatialStructure")
    items = sorted(c.edges_of(s, rel, "RelatedElements"))
    check(len(items) > 2, "Test braucht mehr als zwei Elemente")
    del s.edges[items[1]]                                         # Lücke bei idx 1
    expect(c, s, "edge_index_gap")


def case_neg_list_bounds(c):
    s = c.mutated()
    rel = c.one(s, "IfcRelContainedInSpatialStructure", lambda p: True)
    for k in list(c.edges_of(s, rel, "RelatedElements")):
        del s.edges[k]
    rep = c.v.validate(s)
    check({"required_attribute"} & codes(rep), "leere Pflichtliste wird gemeldet")
    s = c.mutated()
    pt = c.one(s, "IfcCartesianPoint")
    s.nodes[pt]["Coordinates"] = [1.0, 2.0, 3.0, 4.0]
    expect(c, s, "list_bounds")


def case_neg_placement_cycle(c):
    s = c.mutated()
    pls = [sid for sid, p in s.nodes.items() if p["ifc_class"] == "IfcLocalPlacement" and c.edges_of(s, sid, "PlacementRelTo")]
    a = pls[0]
    (k,) = c.edges_of(s, a, "PlacementRelTo")
    s.edges[k] = a
    expect(c, s, "placement_cycle")


def case_neg_labels_and_class(c):
    s = c.mutated()
    w = c.one(s, "IfcWallStandardCase")
    labels = {sid: sorted(set(c.v.si.chain(p["ifc_class"])) | {"IfcEntity"}) for sid, p in s.nodes.items()}
    labels[w] = ["IfcEntity", "IfcWallStandardCase"]
    expect(c, s, "label_mismatch", labels=labels)
    s = c.mutated()
    s.nodes[w]["ifc_class"] = "IfcWandStandardFall"
    expect(c, s, "unknown_class")


def case_neg_unknown_property(c):
    s = c.mutated()
    w = c.one(s, "IfcWallStandardCase")
    s.nodes[w]["Farbe"] = "rot"
    rep = c.v.validate(s)
    check("unknown_property" in codes(rep), "unknown_property erwartet")
    check(rep.ok, "unbekannte Property ist nur Warnung")


def case_neg_opening_rules(c):
    s = c.mutated()
    voids = c.one(s, "IfcRelVoidsElement")
    for k in list(c.edges_of(s, voids)):
        del s.edges[k]
    del s.nodes[voids]
    expect(c, s, "opening_no_host")

    s = c.mutated()
    voids = c.one(s, "IfcRelVoidsElement")
    opening = s.edges[(voids, "RelatedOpeningElement", None, None)]
    new = max(s.nodes) + 1
    s.nodes[new] = dict(s.nodes[voids], step_id=new, GlobalId="A" * 22)
    for k, d in list(c.edges_of(s, voids).items()):
        s.edges[(new, k[1], k[2], k[3])] = d
    expect(c, s, "opening_multiple_hosts")

    s = c.mutated()
    fills = c.one(s, "IfcRelFillsElement")
    opening = s.edges[(fills, "RelatingOpeningElement", None, None)]
    new = max(s.nodes) + 1
    s.nodes[new] = dict(s.nodes[fills], step_id=new, GlobalId="B" * 22)
    for k, d in list(c.edges_of(s, fills).items()):
        s.edges[(new, k[1], k[2], k[3])] = d
    expect(c, s, "opening_multiple_fillings")
    expect(c, s, "filling_multiple_openings")


def case_neg_filling_without_opening(c):
    s = c.mutated()
    fills = c.one(s, "IfcRelFillsElement")
    for k in list(c.edges_of(s, fills)):
        del s.edges[k]
    del s.nodes[fills]
    rep = c.v.validate(s)
    check("filling_without_opening" in codes(rep), "Warnung erwartet")


def case_neg_containers(c):
    s = c.mutated()
    rels = [sid for sid, p in s.nodes.items() if p["ifc_class"] == "IfcRelContainedInSpatialStructure"]
    check(len(rels) >= 2, "braucht zwei Geschosse")
    victim = s.edges[(rels[0], "RelatedElements", 0, None)]
    n = len(c.edges_of(s, rels[1], "RelatedElements"))
    s.edges[(rels[1], "RelatedElements", n, None)] = victim
    expect(c, s, "multiple_containers")
    s = c.mutated()
    for k, d in list(s.edges.items()):
        if k[0] == rels[0] and k[1] == "RelatedElements" and k[2] == 0:
            gone = d
            del s.edges[k]
    # Liste neu nummerieren, damit nur "nicht enthalten" bleibt
    items = [s.edges.pop(k) for k in sorted(c.edges_of(s, rels[0], "RelatedElements"))]
    for i, d in enumerate(items):
        s.edges[(rels[0], "RelatedElements", i, None)] = d
    rep = c.v.validate(s)
    check("element_not_contained" in codes(rep), "Warnung element_not_contained erwartet")


def case_neg_types(c):
    s = c.mutated()
    rels = [sid for sid, p in s.nodes.items() if p["ifc_class"] == "IfcRelDefinesByType"]
    check(len(rels) >= 2, "braucht zwei Typbeziehungen")
    victim = s.edges[(rels[0], "RelatedObjects", 0, None)]
    n = len(c.edges_of(s, rels[1], "RelatedObjects"))
    s.edges[(rels[1], "RelatedObjects", n, None)] = victim
    expect(c, s, "multiple_types")


def case_neg_placement_missing(c):
    s = c.mutated()
    w = c.one(s, "IfcWallStandardCase")
    for k in c.edges_of(s, w, "ObjectPlacement"):
        del s.edges[k]
    expect(c, s, "element_without_placement")


def case_neg_orphans(c):
    s = c.mutated()
    ps = c.one(s, "IfcPropertySet")
    for k in c.incoming(s, ps, "RelatingPropertyDefinition"):
        del s.edges[k]
    rep = c.v.validate(s)
    check("orphan_pset" in codes(rep), "orphan_pset erwartet")
    s = c.mutated()
    pv = c.one(s, "IfcPropertySingleValue")
    for k in c.incoming(s, pv, "HasProperties"):
        del s.edges[k]
    check("orphan_property" in codes(c.v.validate(s)), "orphan_property erwartet")
    s = c.mutated()
    new = max(s.nodes) + 1
    s.nodes[new] = {"step_id": new, "ifc_class": "IfcLocalPlacement"}
    rep = c.v.validate(s)
    check("orphan_placement" in codes(rep), "orphan_placement erwartet")
    check("required_attribute" in codes(rep), "RelativePlacement fehlt")


# --------------------------------------------------------------------------- 5. Negativtests D

def run_op(c, op):
    env = t.Env(c.path)
    env.ex.apply(op)
    return env


def case_neg_d(c):
    ops = ops_for(c.model)
    # Graph passt nicht zur angegebenen Operation
    env = run_op(c, ops["move"])
    expect(c, env.target, "expected_change_missing", op=ops["set_attribute"])
    expect(c, env.target, "unexpected_prop_change", op=ops["set_property"])
    env = run_op(c, ops["delete_window"])
    rep = expect(c, env.target, "unexpected_removed", op=ops["move"])
    env = run_op(c, ops["add_wall"])
    expect(c, env.target, "unexpected_added", op=ops["move"])
    # Operation ohne Wirkung
    expect(c, c.mutated(), "expected_change_missing", op=ops["delete_window"])
    expect(c, c.mutated(), "expected_change_missing", op=ops["move"])
    expect(c, c.mutated(), "expected_change_missing", op=ops["add_wall"])
    expect(c, c.mutated(), "expected_change_missing", op=ops["set_attribute"])
    # Ziel unbekannt
    bad = dict(ops["delete_window"], guid="X" * 22)
    expect(c, c.mutated(), "target_missing", op=bad)
    # zusätzlich zerstörte Raumstruktur
    s = c.mutated()
    storey = c.one(s, "IfcBuildingStorey")
    for k in list(c.edges_of(s, storey)):
        del s.edges[k]
    del s.nodes[storey]
    expect(c, s, "unexpected_removed", op=ops["delete_window"])


def case_neg_d_details(c):
    ops = ops_for(c.model)
    # set_property: falsche Property angegeben
    env = run_op(c, ops["set_property"])
    wrong = dict(ops["set_property"], property_name="GibtEsNicht")
    expect(c, env.target, "unexpected_prop_change", op=wrong)
    # set_attribute: anderes Attribut angegeben
    env = run_op(c, ops["set_attribute"])
    expect(c, env.target, "unexpected_prop_change", op=dict(ops["set_attribute"], attribute="Name"))
    # assign_type: anderes Typobjekt angegeben
    env = run_op(c, ops["assign_type"])
    other = [x.GlobalId for x in c.model.by_type("IfcWallType") if x.GlobalId != ops["assign_type"]["type_guid"]][0]
    expect(c, env.target, "expected_change_missing", op=dict(ops["assign_type"], type_guid=other))
    # add_hosted: falsche Wirtswand
    env = run_op(c, ops["add_window"])
    other_wall = [w.GlobalId for w in c.model.by_type("IfcWall") if w.GlobalId != ops["add_window"]["host_guid"]][0]
    expect(c, env.target, "expected_change_missing", op=dict(ops["add_window"], host_guid=other_wall))
    # add_element: falsche Klasse erwartet
    env = run_op(c, ops["add_wall"])
    expect(c, env.target, "expected_change_missing", op=dict(ops["add_wall"], ifc_class="IfcColumn"))
    # reassign: anderes Geschoss
    env = run_op(c, ops["reassign"])
    expect(c, env.target, "expected_change_missing", op=dict(ops["reassign"], target_storey_guid=ids(c.model)["st0"]))
    # clone: Vorlage verändert
    env = run_op(c, ops["clone_wall"])
    src = c.v.validate(env.target, ops["clone_wall"])
    check(src.ok, "Klon zuerst in Ordnung")
    w = c.v.validate(env.target, {"op": "erfunden"})
    check("unknown_operation" in codes(w), "unbekannte Operation: Warnung")
    check(w.ok, "unbekannte Operation blockiert nicht allein")


def case_report_json(c):
    env = run_op(c, ops_for(c.model)["add_wall"])
    s = copy.deepcopy(env.target)
    w = c.one(s, "IfcWall")
    del s.nodes[w]["GlobalId"]
    rep = c.v.validate(s, ops_for(c.model)["add_wall"])
    js = json.dumps(rep.to_json(), ensure_ascii=False)
    back = json.loads(js)
    check(back["ok"] is False and back["new_errors"], "JSON-Bericht")
    check("GlobalId" in rep.feedback(), rep.feedback())


CASES = [v for k, v in sorted(globals().items(), key=lambda kv: getattr(kv[1], "__code__", None) and kv[1].__code__.co_firstlineno or 0)
         if k.startswith("case_") and callable(v)]


def main(path):
    c = Ctx(path)
    for fn in CASES:
        try:
            fn(c)
            PASSED.append(fn.__name__)
            print(f"  ok    {fn.__name__}")
        except Exception as e:
            FAILED.append(fn.__name__)
            import traceback
            print(f"  FEHL  {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} bestanden")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "AC20-FZK-Haus.ifc"))

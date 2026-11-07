#!/usr/bin/env python3
"""
test_sg_validator.py — prüft semantic_geometric_validator.py ohne Neo4j-Verbindung.

Der Weg ist der echte: Executor ändert den Graphen im Speicher, die Reconstruction patcht eine Kopie der
Original-IFC (test_executor.Env.finish), der Validator liest das Ergebnis und das Original.

  1. Hilfsfunktionen: Durchdringungsvolumen gegen bekannte Körper, Maßregeln
  2. Operationen des Executors: keine neuen Fehler, erwartete Warnungen
  3. Negativtests: das rekonstruierte Modell wird gezielt verdorben, der erwartete Code muss als NEUER Befund erscheinen
  4. Kollision gegen berechnetes Volumen (zwei neue Stützen)
  5. G5 Bestand unverändert

KI-Hinweis: mit Unterstützung von Claude (Anthropic) erstellt, Dokumentation in Anhang F.

Nutzung:  python test_sg_validator.py AC20-FZK-Haus.ifc
"""

from __future__ import annotations

import sys
import time

import numpy as np

import ifcopenshell

import semantic_geometric_validator as sg
import test_executor as t

PASSED, FAILED = [], []
W4 = "2XPyKWY018sA1ygZKgQPtU"
ST0 = "2eyxpyOx95m90jmsXLOuR0"
ST1 = "273g3wqLzDtfYIl7qqkgcO"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


class Ctx:
    def __init__(self, path):
        self.path = path
        self.orig = ifcopenshell.open(path)
        self.v = sg.SemanticGeometricValidator(self.orig)

    def run(self, ops):
        """Operationen nacheinander ausführen, rekonstruieren. Gibt (Ergebnis-IFC, Operationsliste) zurück."""
        env = t.Env(self.path)
        for op in ops:
            env.ex.apply(op)
        _, _, patched, rest = env.finish()
        return patched


def codes(rep, kind="new"):
    fs = rep.new_errors + rep.new_warnings if kind == "new" else rep.findings
    return sorted({f.code for f in fs})


def expect(rep, code, severity=None):
    got = [f for f in rep.new_errors + rep.new_warnings if f.code == code]
    check(got, f"{code} nicht gemeldet; neu: {codes(rep)}")
    if severity:
        check(all(f.severity == severity for f in got), f"{code} sollte {severity} sein")
    return got


def new_guids(c, model, cls):
    old = {e.GlobalId for e in c.orig.by_type(cls)}
    return [e for e in model.by_type(cls) if e.GlobalId not in old]


ADD_WALL = {"op": "add_element", "ifc_class": "IfcWall", "storey_guid": ST0, "start": [1, 1], "end": [3, 1],
            "thickness": 0.2, "height": 2.5}
ADD_WINDOW = {"op": "add_hosted_element", "ifc_class": "IfcWindow", "host_guid": W4, "offset": 1.0,
              "width": 0.9, "height": 1.0, "sill_height": 0.9}
ADD_DOOR = {"op": "add_hosted_element", "ifc_class": "IfcDoor", "host_guid": W4, "offset": 3.0, "width": 0.9, "height": 2.0}


def column(x, y, w=0.4, h=2.5):
    return {"op": "add_element", "ifc_class": "IfcColumn", "storey_guid": ST0, "position": [x, y],
            "width": w, "depth": w, "height": h}


# --------------------------------------------------------------------------- 1. Hilfsfunktionen

def box(lo, hi):
    lo, hi = np.array(lo, float), np.array(hi, float)
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = []
    for a, b, c, d in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        f += [(a, b, c), (a, c, d)]
    return v, np.array(f)


def case_overlap_volume(c):
    a, b = box((0, 0, 0), (2, 1, 1)), box((1, 0.5, 0), (3, 2, 1))
    vol, dims = sg.overlap_volume(a, b)
    check(abs(vol - 0.5) < 0.03, f"{vol}")
    vol, _ = sg.overlap_volume(box((0, 0, 0), (1, 1, 1)), box((1, 0, 0), (2, 1, 1)))
    check(vol == 0.0, "Berührung ist keine Durchdringung")
    vol, _ = sg.overlap_volume(box((0, 0, 0), (1, 1, 1)), box((0.995, 0, 0), (2, 1, 1)))
    check(vol == 0.0, "Durchdringung unter 1 cm zählt nicht")
    v, f = box((0, 0, 0), (1, 1, 1))
    th = np.pi / 4
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    shift = np.array([0.5, -0.2, 0])
    vol, _ = sg.overlap_volume((v, f), (v @ R.T + shift, f))
    g = (np.arange(1500) + 0.5) / 1500
    X, Y = np.meshgrid(g, g)                                   # Referenz: unabhängiger Flächentest
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    q = (P - shift) @ R                                          # zurück in den Rahmen des gedrehten Quadrats
    ref = ((q[:, 0] >= 0) & (q[:, 0] <= 1) & (q[:, 1] >= 0) & (q[:, 1] <= 1)).mean()
    check(abs(vol - ref) < 0.04, f"gedreht {vol} gegen Referenz {ref:.3f}")
    inner = box((0.2, 0.2, 0.2), (0.8, 0.8, 0.8))
    vol, _ = sg.overlap_volume(box((0, 0, 0), (1, 1, 1)), inner)
    check(abs(vol - 0.216) < 0.01, f"enthalten {vol}")


def case_dimension_rules(c):
    check(not sg._dims_wall(5, 0.2, 2.5), "normale Wand")
    check(sg._dims_wall(5, 2.0, 2.5), "Dicke 2 m")
    check(sg._dims_door(0.9, 0.05, 2.0) == [], "normale Tür")
    check(sg._dims_door(0.2, 0.05, 2.0), "Tür 20 cm breit")
    check(sg._dims_column(0.05, 0.05, 3), "Stütze 5 cm")
    check(sg._dims_slab(5, 5, 0.01), "Platte 1 cm")


def case_rules_catalog(c):
    check({v[0] for v in sg.RULES.values()} == {"G1", "G2", "G3", "G4", "G5"}, "alle Gruppen")
    check(all(v[1] in (sg.ERROR, sg.WARNING) for v in sg.RULES.values()), "Schweregrade")


def case_unchanged_model(c):
    rep = c.v.validate(c.orig)
    check(rep.ok and not rep.findings, rep.summary())
    check("0 neue" in rep.diff_summary, rep.diff_summary)


# --------------------------------------------------------------------------- 2. Operationen

def case_op_delete_and_attribute(c):
    for ops in ([{"op": "delete_element", "guid": c.orig.by_type("IfcWindow")[0].GlobalId}],
                [{"op": "set_attribute", "guid": W4, "attribute": "Description", "value": "x"}]):
        rep = c.v.validate(c.run(ops), ops[0])
        check(rep.ok and not rep.findings, rep.summary())


def case_op_move(c):
    op = {"op": "move_element", "guid": W4, "translation": [0.5, 0, 0]}
    res = c.run([op])
    rep = c.v.validate(res, op)
    check(rep.ok, rep.feedback())
    check("1 verschobene" in rep.diff_summary or "verschobene" in rep.diff_summary, rep.diff_summary)
    check("existing_changed" not in codes(rep, "all"), "Ziel darf sich bewegen")


def case_op_add_wall_and_window(c):
    res = c.run([ADD_WALL, ADD_WINDOW, ADD_DOOR])
    rep = c.v.validate(res)
    check(rep.ok, rep.feedback())
    check(not [f for f in rep.findings if f.group == "G2"], [f.message for f in rep.findings if f.group == "G2"])
    check("5 neue" in rep.diff_summary, rep.diff_summary)      # Wand, Fenster, Tür, zwei Öffnungen


def case_op_clone_and_assign(c):
    ops = [{"op": "clone_element", "source_guid": W4, "translation": [0, 3, 0]}]
    rep = c.v.validate(c.run(ops), ops[0])
    check(rep.ok, rep.feedback())


# --------------------------------------------------------------------------- 3. Negativtests

def shifted(model, el, dx=0.0, dy=0.0, dz=0.0):
    loc = el.ObjectPlacement.RelativePlacement.Location
    x, y, z = loc.Coordinates
    loc.Coordinates = (x + dx, y + dy, z + dz)


def case_neg_opening_outside_wall(c):
    res = c.run([ADD_WINDOW])
    o = new_guids(c, res, "IfcOpeningElement")[0]
    shifted(res, o, dx=10.0)
    rep = c.v.validate(res)
    expect(rep, "opening_outside_wall", sg.ERROR)
    check(not rep.ok, "muss fehlschlagen")
    check("Geometrieprüfung fehlgeschlagen" in rep.feedback(), rep.feedback())


def case_neg_opening_too_high(c):
    res = c.run([ADD_WINDOW])
    o = new_guids(c, res, "IfcOpeningElement")[0]
    shifted(res, o, dz=5.0)
    expect(c.v.validate(res), "opening_outside_wall", sg.ERROR)


def case_neg_filling_outside_opening(c):
    res = c.run([ADD_WINDOW])
    w = new_guids(c, res, "IfcWindow")[0]
    shifted(res, w, dx=0.8)
    expect(c.v.validate(res), "filling_outside_opening", sg.ERROR)


def case_neg_openings_overlap(c):
    second = dict(ADD_WINDOW, offset=1.3)
    res = c.run([ADD_WINDOW, second])
    rep = c.v.validate(res)
    got = expect(rep, "openings_overlap", sg.ERROR)
    check(got[0].detail["overlap"][0] > 0.5, got[0].detail)


def case_neg_geometry_missing(c):
    res = c.run([ADD_WALL])
    w = new_guids(c, res, "IfcWall")[0]
    w.Representation = None
    expect(c.v.validate(res), "geometry_missing", sg.ERROR)


def case_neg_geometry_degenerate(c):
    res = c.run([ADD_WALL])
    w = new_guids(c, res, "IfcWall")[0]
    for sol in res.by_type("IfcExtrudedAreaSolid"):
        if sol.Depth == 2.5:
            sol.Depth = 1e-9
    rep = c.v.validate(res)
    check({"geometry_degenerate", "geometry_missing"} & set(codes(rep)), codes(rep))
    check(not rep.ok, "muss fehlschlagen")


def case_neg_dimensions(c):
    thick = dict(ADD_WALL, thickness=2.0)
    res = c.run([thick, column(40, 40, w=0.05)])
    rep = c.v.validate(res)
    got = expect(rep, "dimension_range", sg.WARNING)
    check({f.ifc_class for f in got} >= {"IfcWall", "IfcColumn"}, [f.ifc_class for f in got])
    check(rep.ok, "Maßplausibilität ist nur Warnung")


def case_neg_storey(c):
    res = c.run([dict(ADD_WALL, z_offset=10.0), column(100, 100)])
    rep = c.v.validate(res)
    expect(rep, "outside_storey_height", sg.WARNING)
    expect(rep, "outside_building", sg.WARNING)
    check(rep.ok, "Lage ist nur Warnung")


# --------------------------------------------------------------------------- 4. Kollision

def case_collision_volume(c):
    res = c.run([column(3.0, 3.0), column(3.2, 3.0)])
    cols = new_guids(c, res, "IfcColumn")
    check(len(cols) == 2, "zwei Stützen")
    rep = c.v.validate(res)
    pair = {x.GlobalId for x in cols}
    got = [f for f in expect(rep, "collision", sg.WARNING) if f.guid in pair and pair <= set(f.subject.split("#")[1].split("|"))]
    check(len(got) == 1, [f.message for f in rep.findings if f.code == "collision"])
    check(abs(got[0].detail["volume"] - 0.2) < 0.02, got[0].detail)        # 0,2 m × 0,4 m × 2,5 m
    check(rep.ok, "Kollision ist nur Warnung")


def case_collision_slab_known(c):
    slab = {"op": "add_element", "ifc_class": "IfcSlab", "storey_guid": ST0,
            "outline": [[0, 0], [2, 0], [2, 2], [0, 2]], "thickness": 0.2}
    rep = c.v.validate(c.run([slab]))
    got = expect(rep, "collision", sg.WARNING)
    check(any(f.ifc_class == "IfcSlab" for f in got), "Platte überlappt die vorhandene")


def case_no_collision_for_touching(c):
    res = c.run([column(3.0, 3.0), column(3.4, 3.0)])           # berühren sich exakt
    cols = {x.GlobalId for x in new_guids(c, res, "IfcColumn")}
    rep = c.v.validate(res)
    hits = [f for f in rep.findings if f.code == "collision" and f.guid in cols
            and cols <= set(f.subject.split("#")[1].split("|"))]
    check(not hits, "Berührung ist keine Kollision")


# --------------------------------------------------------------------------- 5. Bestand (G5)

def case_existing_changed(c):
    move = {"op": "move_element", "guid": W4, "translation": [0.5, 0, 0]}
    res = c.run([move])
    rep = c.v.validate(res, {"op": "set_attribute", "guid": W4, "attribute": "Name", "value": "x"})
    expect(rep, "existing_changed", sg.ERROR)
    check(not rep.ok, "muss fehlschlagen")
    other = [w.GlobalId for w in c.orig.by_type("IfcWall") if w.GlobalId != W4][0]
    rep = c.v.validate(res, {"op": "move_element", "guid": other, "translation": [0.5, 0, 0]})
    expect(rep, "existing_changed", sg.ERROR)
    rep = c.v.validate(res, move)
    check("existing_changed" not in codes(rep, "all"), "passende Operation: kein G5-Befund")


def case_existing_marked(c):
    """Eine Kollision, die schon im Ausgangsmodell bestand, blockiert nicht und ist 'existing'."""
    res = c.run([column(3.0, 3.0), column(3.2, 3.0)])
    v2 = sg.SemanticGeometricValidator(res)                       # Ausgangsmodell mit der Kollision
    res2 = c.run([column(3.0, 3.0), column(3.2, 3.0)])
    shifted(res2, new_guids(c, res2, "IfcColumn")[0], dz=0.0)
    for e in res2.by_type("IfcColumn"):
        pass
    rep = v2.validate(res)                                        # unverändert
    check(rep.ok and not rep.findings, rep.summary())
    # Stütze um 1 cm verschieben: Paar bleibt bestehen -> existing
    for e in res.by_type("IfcColumn"):
        if e.GlobalId in {x.GlobalId for x in new_guids(c, res, "IfcColumn")}:
            shifted(res, e, dy=0.01)
            break
    rep = v2.validate(res)
    col = [f for f in rep.findings if f.code == "collision"]
    check(col and all(f.existing for f in col), [(f.existing, f.message) for f in rep.findings])
    check(rep.ok, "bestehende Befunde blockieren nicht")


def case_report_json(c):
    import json
    res = c.run([ADD_WINDOW])
    shifted(res, new_guids(c, res, "IfcOpeningElement")[0], dx=10.0)
    rep = c.v.validate(res)
    back = json.loads(json.dumps(rep.to_json(), ensure_ascii=False))
    check(back["ok"] is False and back["new_errors"], "JSON")
    check(back["new_errors"][0]["code"] == "opening_outside_wall", back["new_errors"][0])


def case_runtime(c):
    t0 = time.time()
    res = c.run([ADD_WALL, ADD_WINDOW, ADD_DOOR])
    t1 = time.time()
    c.v.validate(res)
    dt = time.time() - t1
    check(dt < 60, f"{dt:.1f} s")
    print(f"      Prüfung eines Ergebnisses: {dt:.1f} s")


CASES = [v for k, v in sorted(globals().items(), key=lambda kv: getattr(kv[1], "__code__", None) and kv[1].__code__.co_firstlineno or 0)
         if k.startswith("case_") and callable(v)]


def main(path):
    c = Ctx(path)
    for fn in CASES:
        t0 = time.time()
        try:
            fn(c)
            PASSED.append(fn.__name__)
            print(f"  ok    {fn.__name__} ({time.time() - t0:.0f} s)", flush=True)
        except Exception as e:
            FAILED.append(fn.__name__)
            import traceback
            print(f"  FEHL  {fn.__name__}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(limit=3)
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} bestanden")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "AC20-FZK-Haus.ifc"))

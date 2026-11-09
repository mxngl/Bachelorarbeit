#!/usr/bin/env python3
"""kandidaten.py: Kandidatenliste fuer den Aufgabenentwurf (ohne LLM, nur ifcopenshell).

KI-Hinweis (Anhang F): erzeugt mit Claude Sonnet 5.5 (Anthropic) im Chat vom 09.11.2026.

Pro Modell eine CSV (kandidaten_<modell>.csv) mit einer Zeile je Element der gewaehlten Klassen:
GUID, Klasse, Name, Typ, Geschoss, Raum (Name und Langname), Host-Wand (bei Tueren/Fenstern),
Position (Placement-Ursprung in Metern) und Eindeutigkeitsmasse:
  n_name        wie oft der Name im Modell vorkommt (1 = Name eindeutig)
  n_geschoss    Elemente derselben Klasse im selben Geschoss
  n_raum        Elemente derselben Klasse im selben Raum (1 = "das Fenster im Raum X" ist eindeutig)
  n_typ_raum    Elemente derselben Klasse und desselben Typs im selben Raum
Raumzuordnung: zuerst ueber IfcRelSpaceBoundary, sonst Placement-Ursprung in der Bounding Box eines Raums
desselben Geschosses (Spalte raum_quelle). Aendert nichts am Modell.

Aufruf:
  python3 kandidaten.py test-models/duplex.ifc test-models/smiley_west.ifc test-models/dental_clinic.ifc
  optional: --tasks tasks_residential1.json  (zeigt je Kategorie 2 Beispielaufgaben aus dem Pilotkatalog)
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as E
import ifcopenshell.util.placement as P
import ifcopenshell.util.unit as U

CLASSES = ["IfcWall", "IfcDoor", "IfcWindow", "IfcFurnishingElement", "IfcColumn", "IfcBeam", "IfcSlab",
           "IfcStair", "IfcRailing", "IfcCovering", "IfcCurtainWall", "IfcFlowTerminal"]


def parent_storey(el):
    c = E.get_container(el)
    if c is None and hasattr(E, "get_aggregate"):
        c = E.get_aggregate(el)  # Raeume haengen per Aggregation am Geschoss
    seen = 0
    while c is not None and not c.is_a("IfcBuildingStorey") and seen < 6:
        c = E.get_aggregate(c) if hasattr(E, "get_aggregate") else None
        seen += 1
    return c


def origin(el, scale):
    try:
        if el.ObjectPlacement is None:
            return None
        m = P.get_local_placement(el.ObjectPlacement)
        return tuple(float(v) * scale for v in m[:3, 3])
    except Exception:
        return None


def space_boxes(model):
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    boxes = []
    for sp in model.by_type("IfcSpace"):
        try:
            sh = ifcopenshell.geom.create_shape(s, sp)
            v = np.array(sh.geometry.verts).reshape(-1, 3)
            boxes.append((sp, v.min(axis=0), v.max(axis=0)))
        except Exception:
            pass
    return boxes


def boundary_map(model):
    """element id -> Liste von Raeumen (ueber IfcRelSpaceBoundary)."""
    out = collections.defaultdict(list)
    for rel in model.by_type("IfcRelSpaceBoundary"):
        sp, el = rel.RelatingSpace, rel.RelatedBuildingElement
        if sp is not None and el is not None and sp.is_a("IfcSpace"):
            if sp not in out[el.id()]:
                out[el.id()].append(sp)
    return out


def host_of(el):
    for rel in getattr(el, "FillsVoids", None) or []:
        for v in rel.RelatingOpeningElement.VoidsElements or []:
            return v.RelatingBuildingElement
    return None


def build(path, outdir):
    model = ifcopenshell.open(path)
    scale = U.calculate_unit_scale(model)
    bmap = boundary_map(model)
    boxes = space_boxes(model)
    box_storey = {sp.id(): (parent_storey(sp).Name if parent_storey(sp) else None) for sp, _, _ in boxes}

    rows = []
    for cls in CLASSES:
        for el in model.by_type(cls):
            st = parent_storey(el)
            host = host_of(el) if cls in ("IfcDoor", "IfcWindow") else None
            if st is None and host is not None:
                st = parent_storey(host)
            stn = st.Name if st else None
            pos = origin(el, scale)
            room, quelle = None, ""
            # Raumzuordnung: Grenzen, bei Fuellungen ueber die Hostwand
            spaces = bmap.get(el.id()) or (bmap.get(host.id()) if host is not None else None) or []
            if len(spaces) == 1:
                room, quelle = spaces[0], "boundary"
            elif len(spaces) > 1:
                room, quelle = spaces[0], "boundary_mehrere"
            elif pos is not None:
                best = None
                for sp, lo, hi in boxes:
                    if box_storey.get(sp.id()) != stn:
                        continue
                    if lo[0] - 0.05 <= pos[0] <= hi[0] + 0.05 and lo[1] - 0.05 <= pos[1] <= hi[1] + 0.05 \
                            and lo[2] - 0.2 <= pos[2] <= hi[2] + 0.05:
                        vol = float(np.prod(hi - lo))
                        if best is None or vol < best[0]:
                            best = (vol, sp)
                if best:
                    room, quelle = best[1], "bbox"
            t = E.get_type(el)
            rows.append({
                "guid": el.GlobalId, "basis": cls, "klasse": el.is_a(), "name": el.Name or "", "typ": t.Name if t else "",
                "geschoss": stn or "", "raum": room.Name if room else "", "raum_lang": (room.LongName or "") if room else "",
                "raum_quelle": quelle, "host": (host.Name if host is not None else ""),
                "x": round(pos[0], 3) if pos else "", "y": round(pos[1], 3) if pos else "", "z": round(pos[2], 3) if pos else "",
            })

    cname = collections.Counter((r["basis"], r["name"]) for r in rows)
    cstor = collections.Counter((r["basis"], r["geschoss"]) for r in rows)
    croom = collections.Counter((r["basis"], r["geschoss"], r["raum"]) for r in rows if r["raum"])
    ctyp = collections.Counter((r["basis"], r["geschoss"], r["raum"], r["typ"]) for r in rows if r["raum"])
    for r in rows:
        r["n_name"] = cname[(r["basis"], r["name"])]
        r["n_geschoss"] = cstor[(r["basis"], r["geschoss"])]
        r["n_raum"] = croom[(r["basis"], r["geschoss"], r["raum"])] if r["raum"] else ""
        r["n_typ_raum"] = ctyp[(r["basis"], r["geschoss"], r["raum"], r["typ"])] if r["raum"] else ""

    stem = os.path.splitext(os.path.basename(path))[0]
    fn = os.path.join(outdir, f"kandidaten_{stem}.csv")
    cols = ["guid", "basis", "klasse", "name", "typ", "geschoss", "raum", "raum_lang", "raum_quelle", "host",
            "x", "y", "z", "n_name", "n_geschoss", "n_raum", "n_typ_raum"]
    with open(fn, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\n==== {path} [{model.schema}] -> {fn}")
    print(f"{'Klasse':22}{'n':>6}{'Raum':>7}{'bbox':>6}{'Name=1':>8}{'Raum=1':>8}{'Raum+Typ=1':>11}")
    for cls in CLASSES:
        rs = [r for r in rows if r["basis"] == cls]
        if not rs:
            continue
        print(f"{cls:22}{len(rs):>6}{sum(1 for r in rs if r['raum']):>7}{sum(1 for r in rs if r['raum_quelle'] == 'bbox'):>6}"
              f"{sum(1 for r in rs if r['n_name'] == 1):>8}{sum(1 for r in rs if r['n_raum'] == 1):>8}"
              f"{sum(1 for r in rs if r['n_typ_raum'] == 1):>11}")
    # Raeume mit genau einem Fenster/Tuer/Moebel: gute Ziele fuer eindeutige raeumliche Referenzen
    per_room = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["raum"]:
            per_room[(r["geschoss"], r["raum"], r["raum_lang"])][r["basis"]] += 1
    print("Beispielraeume (Geschoss, Raum, Langname -> Elemente):")
    shown = 0
    for k, c in sorted(per_room.items(), key=lambda kv: -sum(kv[1].values())):
        if shown >= 8:
            break
        print("  ", k, dict(c))
        shown += 1
    rooms = {(sp.Name, sp.LongName) for sp, _, _ in boxes} or set()
    langn = collections.Counter(l for _, l in rooms if l)
    print("Raeume gesamt mit Geometrie:", len(boxes), "| Langnamen, die bei mehreren Raeumen vorkommen:",
          [(n, c) for n, c in langn.most_common() if c > 1][:6])
    return rows


def show_tasks(path):
    d = json.load(open(path, encoding="utf-8"))
    d = d if isinstance(d, list) else d.get("tasks", d)
    by = collections.defaultdict(list)
    for t in d:
        by[t.get("category")].append(t)
    print(f"\n==== Beispielaufgaben aus {path}")
    for cat, ts in by.items():
        for t in ts[:2]:
            print(json.dumps(t, ensure_ascii=False)[:600])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ifc", nargs="+")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--tasks")
    a = ap.parse_args()
    for p in a.ifc:
        try:
            build(p, a.outdir)
        except Exception as e:
            import traceback
            print(f"FEHLER {p}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(limit=3)
    if a.tasks:
        show_tasks(a.tasks)

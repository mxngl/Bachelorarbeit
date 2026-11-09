#!/usr/bin/env python3
"""inventar.py: Modellinventar fuer den Aufgabenentwurf (ohne LLM, nur ifcopenshell).

KI-Hinweis (Anhang F): erzeugt mit Claude Sonnet 5.5 (Anthropic) im Chat vom 09.11.2026.

Pro Modell: Schema, Geschosse mit Elementanzahl je Klasse, Raeume, Klassenzaehlung,
Namenseindeutigkeit (wichtig fuer Lokalisierung), Typen, Fuellungen (Tueren/Fenster) mit Host-Wand,
Geometrie-Check je Klasse (Mesh berechenbar ja/nein) und vorhandene Property Sets.
Aendert nichts am Modell. Aufruf:
  python3 inventar.py test-models/duplex.ifc [weitere.ifc ...] [--json inventar.json] [--max-geom 40]
"""
import argparse
import collections
import json
import sys

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as E

CLASSES = ["IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcRoof", "IfcColumn", "IfcBeam", "IfcStair",
           "IfcStairFlight", "IfcRailing", "IfcDoor", "IfcWindow", "IfcCovering", "IfcFurnishingElement",
           "IfcCurtainWall", "IfcPlate", "IfcMember", "IfcSpace", "IfcOpeningElement"]


def storey_of(el):
    c = E.get_container(el)
    while c is not None and not c.is_a("IfcBuildingStorey"):
        c = E.get_aggregate(c) if hasattr(E, "get_aggregate") else None
    return c.Name if c is not None else None


def name_stats(items):
    names = [(i.Name or "") for i in items]
    cnt = collections.Counter(names)
    return {
        "n": len(items),
        "ohne_name": cnt.get("", 0),
        "eindeutig": sum(1 for v in cnt.values() if v == 1),
        "doppelt": sum(1 for v in cnt.values() if v > 1),
        "haeufigste": cnt.most_common(4),
    }


def geom_ok(model, els, limit):
    s = ifcopenshell.geom.settings()
    ok = fail = 0
    for el in els[:limit]:
        try:
            if ifcopenshell.geom.create_shape(s, el) is not None:
                ok += 1
            else:
                fail += 1
        except Exception:
            fail += 1
    return ok, fail


def inventory(path, max_geom):
    m = ifcopenshell.open(path)
    inv = {"datei": path, "schema": m.schema}
    prod = m.by_type("IfcElement")
    inv["elemente_gesamt"] = len(prod)
    inv["mit_globalid"] = sum(1 for e in prod if e.GlobalId)

    counts = collections.Counter(e.is_a() for e in prod)
    counts["IfcSpace"] = len(m.by_type("IfcSpace"))
    inv["klassen"] = dict(counts.most_common())

    storeys = []
    for st in m.by_type("IfcBuildingStorey"):
        per = collections.Counter()
        for el in prod:
            c = E.get_container(el)
            if c is not None and c.id() == st.id():
                per[el.is_a()] += 1
        storeys.append({"name": st.Name, "elevation": st.Elevation, "elemente": dict(per.most_common(8)),
                        "raeume": sum(1 for sp in m.by_type("IfcSpace")
                                      if any(r.RelatingObject.id() == st.id() for r in (sp.Decomposes or [])))})
    inv["geschosse"] = storeys

    sp = m.by_type("IfcSpace")
    inv["raeume"] = [{"name": s.Name, "long": s.LongName} for s in sp[:40]]
    inv["raeume_namen"] = name_stats(sp) if sp else None

    ns = {}
    for cls in ("IfcWall", "IfcDoor", "IfcWindow", "IfcSlab", "IfcColumn", "IfcStair"):
        items = m.by_type(cls)
        if items:
            ns[cls] = name_stats(items)
    inv["namen"] = ns

    types = collections.Counter()
    for cls in ("IfcWall", "IfcDoor", "IfcWindow", "IfcSlab"):
        for el in m.by_type(cls):
            t = E.get_type(el)
            types[(cls, t.Name if t else None)] += 1
    inv["typen"] = [{"klasse": k[0], "typ": k[1], "anzahl": v} for k, v in types.most_common(20)]

    fill = []
    for cls in ("IfcDoor", "IfcWindow"):
        for el in m.by_type(cls):
            host = None
            for rel in el.FillsVoids or []:
                op = rel.RelatingOpeningElement
                for v in op.VoidsElements or []:
                    host = v.RelatingBuildingElement
            fill.append((el, host))
    inv["fuellungen"] = {"gesamt": len(fill), "mit_hostwand": sum(1 for _, h in fill if h is not None),
                         "beispiele": [{"fuellung": f.Name, "host": h.Name if h else None} for f, h in fill[:8]]}

    wl = [w for w in m.by_type("IfcWall")]
    inv["waende"] = {
        "gesamt": len(wl),
        "mit_oeffnungen": sum(1 for w in wl if w.HasOpenings),
        "mit_ifcrelconnects": sum(1 for w in wl if getattr(w, "ConnectedTo", None) or getattr(w, "ConnectedFrom", None)),
        "mit_material": sum(1 for w in wl if E.get_material(w)),
    }
    rep = collections.Counter()
    for w in wl:
        if w.Representation:
            rep[tuple(sorted({r.RepresentationIdentifier for r in w.Representation.Representations if r.RepresentationIdentifier}))] += 1
    inv["wand_darstellungen"] = [{"identifier": list(k), "anzahl": v} for k, v in rep.most_common(5)]

    geo = {}
    for cls in ("IfcWall", "IfcSlab", "IfcDoor", "IfcWindow", "IfcColumn", "IfcSpace"):
        items = [e for e in m.by_type(cls) if e.Representation]
        if items:
            ok, fail = geom_ok(m, items, max_geom)
            geo[cls] = {"geprueft": min(len(items), max_geom), "ok": ok, "fehler": fail}
    inv["geometrie_stichprobe"] = geo

    psets = collections.Counter()
    for w in wl[:50]:
        for k in E.get_psets(w):
            psets[k] += 1
    inv["psets_waende_stichprobe"] = dict(psets.most_common(8))
    inv["einheit_hinweis"] = [u.Name for u in m.by_type("IfcSIUnit") if u.UnitType == "LENGTHUNIT"] or "siehe IfcUnitAssignment"
    return inv


def show(inv):
    print(f"\n==== {inv['datei']}  [{inv['schema']}]  Elemente: {inv['elemente_gesamt']}")
    print("Klassen:", ", ".join(f"{k}={v}" for k, v in list(inv["klassen"].items())[:14]))
    for s in inv["geschosse"]:
        print(f"  Geschoss {s['name']!r} (z={s['elevation']}) Raeume={s['raeume']} Elemente={s['elemente']}")
    if inv["raeume_namen"]:
        print("Raeume:", inv["raeume_namen"])
        print("  Beispiele:", [(r['name'], r['long']) for r in inv["raeume"][:10]])
    print("Namenseindeutigkeit (eindeutig = Name kommt genau 1x vor):")
    for k, v in inv["namen"].items():
        print(f"  {k}: n={v['n']} ohne_name={v['ohne_name']} eindeutig={v['eindeutig']} doppelt={v['doppelt']} haeufig={v['haeufigste'][:3]}")
    print("Typen:", [(t['klasse'], t['typ'], t['anzahl']) for t in inv["typen"][:8]])
    print("Fuellungen:", inv["fuellungen"]["gesamt"], "mit Hostwand:", inv["fuellungen"]["mit_hostwand"], inv["fuellungen"]["beispiele"][:4])
    print("Waende:", inv["waende"], "Darstellungen:", inv["wand_darstellungen"])
    print("Geometrie-Stichprobe:", inv["geometrie_stichprobe"])
    print("Psets Waende:", inv["psets_waende_stichprobe"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ifc", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--max-geom", type=int, default=40)
    a = ap.parse_args()
    out = []
    for p in a.ifc:
        try:
            inv = inventory(p, a.max_geom)
            show(inv)
            out.append(inv)
        except Exception as e:
            print(f"FEHLER {p}: {type(e).__name__}: {e}", file=sys.stderr)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False, default=str)
        print("\nJSON:", a.json)

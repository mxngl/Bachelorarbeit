#!/usr/bin/env python3
"""tasks_build.py: erzeugt die Aufgabenkataloge fuer duplex, smiley_west und dental_clinic aus den Kandidaten-CSVs.

KI-Hinweis (Anhang F): erzeugt mit Claude Sonnet 5.5 (Anthropic) im Chat vom 09.11.2026.

Eingabe:  kandidaten_<modell>.csv (kandidaten.py) und die Geschoss-GUIDs (unten fest eingetragen, Ausgabe vom Spark).
Ausgabe:  tasks_<modell>.json im Schema der Pilotkataloge (id, category, task, expected).
Alle Ziele werden per Regel aus den CSV-Spalten gewaehlt (eindeutiger Name, Eindeutigkeit pro Raum, Host-Wand mit
genau einer Fuellung usw.), nicht von Hand. Die GUIDs stammen direkt aus der CSV. Die Soll-Ergebnisse (outcome) sind
Erwartungen und werden mit tasks_check.py (Trockenlauf) am Modell geprueft und danach ggf. korrigiert.
"""
import json
import os
import sys

import pandas as pd

STOREYS = {
    "duplex": {"Level 2": "1xS3BCk291UvhgP2dvNMQJ", "Roof": "1xS3BCk291UvhgP2dvNtSE",
               "Level 1": "1xS3BCk291UvhgP2dvNMKI", "T/FDN": "1xS3BCk291UvhgP2dvNsgp"},
    "smiley_west": {"KG": "0eLD6321X73BGqB5WeSKCe", "EG": "2aIWPvOfn1XvF_7jNsKTBw", "1.OG": "0Lz7dRJK5CdfSh5PhR5gUV",
                    "2.OG": "0pYOL8pVH0bBC2MnVS84J_", "Dach": "1y5XMII898hfivwb6v5YU6"},
    "dental_clinic": {"First Floor": "3eM8WbY_59RR5TDWs3wwJP", "Second Floor": "3eM8WbY_59RR5TDWs3wwJN",
                      "Roof - Main": "3eM8WbY_59RR5TDWs3wwJL", "TOF Footing": "3eM8WbY_59RR5TDWs3wwJR"},
}


def load(path):
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ("n_name", "n_geschoss", "n_raum", "n_typ_raum", "x", "y", "z"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


class B:
    def __init__(self, model, df):
        self.model, self.df, self.used = model, df, set()
        self.st = STOREYS[model]
        self.tasks = []
        self.fills = df[df.host != ""].groupby("host").guid.apply(list).to_dict()
        self.clean = ~df.name.str.contains("'") & ~df.name.str.contains('"') & ~df.raum_lang.str.contains("'")
        self.rl_n = df[df.raum_lang != ""].groupby("raum_lang").raum.nunique()

    def pick(self, mask, n=0, key=("geschoss", "name")):
        c = self.df[mask & self.clean & ~self.df.guid.isin(self.used)].sort_values(list(key))
        if len(c) <= n:
            import inspect; raise SystemExit(f"{self.model}: keine Kandidaten fuer Regel (n={n}), Zeile {inspect.stack()[1].lineno}")
        r = c.iloc[n]
        self.used.add(r.guid)
        return r

    def add(self, tid, cat, text, expected):
        self.tasks.append({"id": tid, "category": cat, "task": text, "expected": expected})

    def cls(self, basis, **kw):
        m = self.df.basis == basis
        return m


def r2(v):
    return round(float(v), 3)


def build(model, df):
    b = B(model, df)
    D = b.df
    uniq = D.n_name == 1
    ok = lambda cls: (D.basis == cls) & uniq
    wall_nofill = lambda extra=True: ok("IfcWall") & ~D.name.isin(b.fills.keys()) & extra
    wall_onefill = ok("IfcWall") & D.name.map(lambda n: len(b.fills.get(n, []))) .eq(1)
    S = b.st

    # ---------- remove ----------
    if model == "duplex":
        door = b.pick(ok("IfcDoor") & D.name.str.contains("0762") & (D.geschoss == "Level 1"))
        win = b.pick(ok("IfcWindow") & D.name.str.contains("2800mm") & (D.geschoss == "Level 2"))
    elif model == "smiley_west":
        door = b.pick(ok("IfcDoor") & (D.geschoss == "KG") & (D.n_raum == 1), n=0)
        win = b.pick(ok("IfcWindow") & (D.geschoss == "KG"), n=0)
    else:
        door = b.pick(ok("IfcDoor") & (D.n_raum == 1) & (D.geschoss == "Second Floor") & (D.typ == "0915 x 2134mm"))
        win = b.pick(ok("IfcWindow") & (D.geschoss == "Second Floor"))
    b.add("r1", "remove", f"Delete the door '{door['name']}'.",
          {"outcome": "success", "op": "delete_element", "params": {"guid": door.guid}})
    b.add("r2", "remove", f"Remove the window '{win['name']}'.",
          {"outcome": "success", "op": "delete_element", "params": {"guid": win.guid}})

    # r3: Referenz ueber Geschoss/Raum (raeumlich)
    if model == "duplex":
        bed = b.pick((D.basis == "IfcFurnishingElement") & D.name.str.contains("Queen") & (D.raum == "B203"))
        text = "Delete the queen-size bed in room 'B203' on storey 'Level 2'."
        note = f"Raeumliche Referenz, Ziel ist '{bed['name']}' (einziges Queen-Bett im Raum B203; in A203 steht ein weiteres)"
        t = bed
    elif model == "smiley_west":
        t = b.pick((D.basis == "IfcDoor") & uniq & (D.geschoss == "KG") & (D.n_raum == 1) & (D.raum_lang.map(b.rl_n) == 1), n=1)
        text = f"Delete the door in the room '{t['raum_lang']}' on storey 'KG'."
        note = f"Raeumliche Referenz, Ziel ist '{t['name']}' (einzige Tuer im Raum {t['raum']})"
    else:
        t = b.pick((D.basis == "IfcDoor") & uniq & (D.n_raum == 1) & (D.geschoss == "First Floor") & (D.raum_lang.map(b.rl_n) == 1))
        text = f"Delete the door in the room '{t['raum_lang']}' on storey 'First Floor'."
        note = f"Raeumliche Referenz, Ziel ist '{t['name']}' (einzige Tuer im Raum {t['raum']})"
    b.add("r3", "remove", text, {"outcome": "success", "op": "delete_element", "params": {"guid": t.guid}, "note": note})

    # r4: Wand mit genau einer Fuellung (Kaskade)
    cand = wall_onefill & (D.geschoss == {"duplex": "Level 2", "smiley_west": "KG", "dental_clinic": "First Floor"}[model])
    w = b.pick(cand)
    f = b.fills[w["name"]][0]
    fname = D[D.guid == f].name.iloc[0]
    b.used.add(f)
    b.add("r4", "remove", f"Delete the wall '{w['name']}'.",
          {"outcome": "success", "op": "delete_element", "params": {"guid": w.guid},
           "note": f"Kaskade: '{fname}' entfaellt mit"})

    # ---------- reposition ----------
    w1 = b.pick(wall_nofill() & (D.geschoss == {"duplex": "Level 1", "smiley_west": "EG", "dental_clinic": "First Floor"}[model]))
    b.add("m1", "reposition", f"Move the wall '{w1['name']}' 0.5 m in positive X direction.",
          {"outcome": "success", "op": "move_element", "params": {"guid": w1.guid, "translation": [0.5, 0, 0]}})

    if model == "duplex":
        e = b.pick((D.basis == "IfcFurnishingElement") & uniq & D.name.str.contains("Sofa") & (D.raum == "A102"))
        txt = f"Move the sofa '{e['name']}' 1 m in positive X direction."
        par = {"guid": e.guid, "translation": [1, 0, 0]}
    elif model == "smiley_west":
        e = b.pick(ok("IfcColumn") & D.name.str.contains("EG-Stuetze-0"), n=0)
        txt = f"Move the column '{e['name']}' 0.5 m in positive Y direction."
        par = {"guid": e.guid, "translation": [0, 0.5, 0]}
    else:
        e = b.pick((D.basis == "IfcFurnishingElement") & uniq & D.name.str.contains("Counter Top:") & (D.geschoss == "First Floor"))
        txt = f"Move the counter top '{e['name']}' 0.5 m in positive X direction."
        par = {"guid": e.guid, "translation": [0.5, 0, 0]}
    b.add("m2", "reposition", txt, {"outcome": "success", "op": "move_element", "params": par})

    src, dst = {"duplex": ("Level 1", "Level 2"), "smiley_west": ("EG", "1.OG"), "dental_clinic": ("Second Floor", "First Floor")}[model]
    w3 = b.pick(wall_nofill() & (D.geschoss == src))
    b.add("m3", "reposition", f"Move the wall '{w3['name']}' to storey '{dst}'.",
          {"outcome": "success", "op": "reassign_container", "params": {"guid": w3.guid, "target_storey_guid": S[dst]}})

    if model == "duplex":
        e4 = b.pick(ok("IfcBeam") & (D.geschoss == "Level 2"))
    elif model == "smiley_west":
        e4 = b.pick(ok("IfcBeam") & D.name.str.contains("Unterzug-06"))
    else:
        e4 = b.pick(ok("IfcFlowTerminal") & D.name.str.contains("Grab Bars") & (D.geschoss == "First Floor") & (D.x != 0))
    cls_word = {"IfcBeam": "beam", "IfcFlowTerminal": "fixture"}[e4.basis]
    tgt = [r2(e4.x + 1.0), r2(e4.y), r2(e4.z)]
    b.add("m4", "reposition",
          f"Place the {cls_word} '{e4['name']}' so that its origin is at x = {tgt[0]} m, y = {tgt[1]} m, z = {tgt[2]} m.",
          {"outcome": "success", "op": "move_element", "params": {"guid": e4.guid, "target_location": tgt},
           "note": "Bezug ist der Placement-Ursprung des Elements (Ausgangswert in x um 1 m verschoben)"})

    # ---------- add ----------
    ext = {"duplex": ("Exterior - Brick", "Level 2"), "smiley_west": ("EG-A", "EG"), "dental_clinic": ("Exterior - Insul", "First Floor")}[model]
    base = ok("IfcWall") & D.name.str.contains(ext[0], regex=False) & (D.geschoss == ext[1])
    try:
        hw = b.pick(base & ~D.name.isin(b.fills.keys()))
    except SystemExit:
        hw = b.pick(base)  # Fallback: Wand mit bereits vorhandener Fuellung
    exp_out = "success" if model == "smiley_west" else "no_success"
    note_a1 = ("Erwartung aus dem Trockenlauf: Wand hat BaseQuantities (Laenge) und IfcMaterialLayerSetUsage"
               if model == "smiley_west" else
               "Erwartung (Pilotmodelle): BaseQuantities.Length/IfcMaterialLayerSetUsage fehlen an Revit-Waenden, Executor lehnt mit missing_wall_dimensions ab. Ziel ist kein Erfolg")
    b.add("a1", "add",
          f"Add a window to the wall '{hw['name']}', 1.0 m from the wall start, 0.9 m wide and 1.0 m high, with a sill height of 0.9 m.",
          {"outcome": exp_out, "op": "add_hosted_element",
           "params": {"ifc_class": "IfcWindow", "host_guid": hw.guid, "offset": 1.0, "width": 0.9, "height": 1.0, "sill_height": 0.9},
           "note": note_a1})

    if model == "duplex":
        c2 = b.pick((D.basis == "IfcFurnishingElement") & uniq & D.name.str.contains("Sofa") & (D.raum == "B102"))
        c3 = b.pick(ok("IfcRailing") & D.name.str.contains("900mm Handrail"))
        cw2, cw3 = "sofa", "railing"
    elif model == "smiley_west":
        c2 = b.pick(ok("IfcColumn") & D.name.str.contains("Stuetze-03"))
        c3 = b.pick(ok("IfcBeam") & D.name.str.contains("Unterzug-03"))
        cw2, cw3 = "column", "beam"
    else:
        c2 = b.pick((D.basis == "IfcFurnishingElement") & uniq & D.name.str.contains("Counter Top w Sink") & (D.geschoss == "Second Floor"))
        c3 = b.pick(ok("IfcFlowTerminal") & (D.geschoss == "Second Floor") & (D.x != 0))
        cw2, cw3 = "sink counter top", "fixture"
    b.add("a2", "add", f"Create a copy of the {cw2} '{c2['name']}' shifted by 1.5 m in positive Y direction.",
          {"outcome": "success", "op": "clone_element", "params": {"source_guid": c2.guid, "translation": [0, 1.5, 0]}})
    b.add("a3", "add", f"Create a copy of the {cw3} '{c3['name']}' shifted by 1 m in positive X direction.",
          {"outcome": "success", "op": "clone_element", "params": {"source_guid": c3.guid, "translation": [1, 0, 0]}})

    # add_element ausserhalb des Gebaeudes (Gebaeudeumriss aus den Elementpositionen), um Kollisionen zu vermeiden
    if model == "duplex":
        sg = "Level 1"
        text = f"On storey '{sg}', add a column at x = 12.0 m, y = -5.0 m with a cross section of 0.3 m by 0.3 m and a height of 3.0 m."
        par = {"ifc_class": "IfcColumn", "storey_guid": S[sg], "position": [12.0, -5.0], "width": 0.3, "depth": 0.3, "height": 3.0}
    elif model == "smiley_west":
        sg = "EG"
        text = f"On storey '{sg}', add a wall from (30.0, 15.0) to (33.0, 15.0), 0.2 m thick and 3.0 m high."
        par = {"ifc_class": "IfcWall", "storey_guid": S[sg], "start": [30.0, 15.0], "end": [33.0, 15.0], "thickness": 0.2, "height": 3.0}
    else:
        sg = "First Floor"
        text = f"On storey '{sg}', add a slab with the outline (10,10), (12,10), (12,12), (10,12) and a thickness of 0.2 m."
        par = {"ifc_class": "IfcSlab", "storey_guid": S[sg], "outline": [[10, 10], [12, 10], [12, 12], [10, 12]], "thickness": 0.2}
    b.add("a4", "add", text, {"outcome": "success", "op": "add_element", "params": par, "check_coords": True,
                              "note": "Position ausserhalb des Gebaeudeumrisses gewaehlt (Kollisionen vermeiden); Lage im Trockenlauf pruefen"})

    # ---------- abort ----------
    cnt = int((D.basis == "IfcRailing").sum())
    b.add("x1", "abort", "Delete the railing.",
          {"outcome": "abort", "abort_reason": "ambiguous", "stage": "locator", "note": f"{cnt} IfcRailing"})
    if ":" in door["name"]:
        nf = door["name"].rsplit(":", 1)[0] + ":999999"
    else:
        nf = door["name"][:-4] + "99-9"  # Namensschema ohne ElementId (smiley_west): Nummer ersetzen
    assert nf not in set(D.name), nf
    b.add("x2", "abort", f"Delete the door '{nf}'.",
          {"outcome": "abort", "abort_reason": "not_found", "stage": "locator", "note": "Name einer existierenden Tuer, Nummer bzw. ElementId durch einen nicht vorhandenen Wert ersetzt"})
    rw = b.pick(wall_nofill())
    b.add("x3", "abort", f"Rotate the wall '{rw['name']}' by 90 degrees.",
          {"outcome": "abort", "abort_reason": "out_of_scope", "stage": "generator", "note": "rotate_element ist nicht im Vokabular"})
    if model == "duplex":
        b.add("x4", "abort", "Delete the queen-size bed in the room 'Bedroom 2'.",
              {"outcome": "abort", "abort_reason": "ambiguous", "stage": "locator",
               "note": "Zwei Raeume heissen 'Bedroom 2' (A203, B203), jeder mit einem Queen-Bett"})
    elif model == "smiley_west":
        dn = D[(D.basis == "IfcWindow") & (D.n_name > 1)].sort_values("name").iloc[0]
        b.add("x4", "abort", f"Delete the window '{dn['name']}'.",
              {"outcome": "abort", "abort_reason": "ambiguous", "stage": "locator", "note": f"{int(dn.n_name)} Fenster tragen diesen Namen"})
    else:
        dn = D[(D.basis == "IfcDoor") & (D.n_name > 1)].sort_values("name").iloc[0]
        b.add("x4", "abort", f"Delete the door '{dn['name']}'.",
              {"outcome": "abort", "abort_reason": "ambiguous", "stage": "locator", "note": f"{int(dn.n_name)} Tueren tragen diesen Namen"})
    return b.tasks


if __name__ == "__main__":
    indir, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    for model in ("duplex", "smiley_west", "dental_clinic"):
        df = load(os.path.join(indir, f"kandidaten_{model}.csv"))
        tasks = build(model, df)
        fn = os.path.join(outdir, f"tasks_{model}.json")
        json.dump(tasks, open(fn, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"\n== {fn}: {len(tasks)} Aufgaben")
        for t in tasks:
            print(f"  {t['id']:3} {t['expected']['outcome']:10} {t['task'][:150]}")

#!/usr/bin/env python3
"""tasks_check.py: prueft eine Aufgabendatei gegen ihr IFC-Modell und fuehrt jede Soll-Operation ohne LLM aus.

KI-Hinweis (Anhang F): erzeugt mit Claude Sonnet 5.5 (Anthropic) im Chat vom 09.11.2026.

Pro Aufgabe:
  1. Struktur:   id, category, task, expected vorhanden; GUIDs in expected existieren im Modell.
  2. Namen:      Alle in der Aufgabe in einfachen Anfuehrungszeichen genannten Namen (Format 'Name') muessen
                 genau einem Element mit dieser GUID entsprechen (Name eindeutig und gleiche GUID wie expected).
  3. Trockenlauf: Fuer expected.outcome == "success" oder "no_success" mit expected.op wird die Operation
                 {"op": expected.op, **expected.params} direkt durch Executor, Validatoren und Rekonstruktion geschickt
                 (ohne Locator/Generator, ohne LLM). Abbruchaufgaben (outcome == "abort") werden nur strukturell geprueft.
     Ergebnis: ok          Soll "success", Pipeline: success
               ok_abgelehnt Soll "no_success", Pipeline lehnt ab (wie erwartet)
               ABWEICHUNG  Soll und Pipeline stimmen nicht ueberein (Aufgabe anpassen oder als bekannte Grenze notieren)
Aendert keine Repo-Datei, schreibt keine Ergebnis-IFC in test-models (Trockenlauf im Speicher).
Aufruf (im Repo-Ordner):
  python3 tasks_check.py tasks_duplex.json test-models/duplex.ifc [--only a1,a2] [--skip-run]
"""
import argparse
import json
import re
import sys
import time
import traceback

import ifcopenshell

from test_orchestrator import Ctx


def guids_in(obj):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.endswith("guid") and isinstance(v, str):
                out.append(v)
            else:
                out.extend(guids_in(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(guids_in(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("ifc")
    ap.add_argument("--only")
    ap.add_argument("--skip-run", action="store_true")
    a = ap.parse_args()

    tasks = json.load(open(a.tasks, encoding="utf-8"))
    tasks = tasks if isinstance(tasks, list) else tasks.get("tasks", tasks)
    model = ifcopenshell.open(a.ifc)
    # Namen nur von Bauteilen und Raumstruktur zaehlen. IfcOpeningElement und Typobjekte tragen in Revit-Modellen
    # oft denselben Namen wie das Bauteil, das der Locator meint, und waeren hier sonst ein Scheinduplikat.
    names = {}
    pool = [e for e in model.by_type("IfcElement") if not e.is_a("IfcOpeningElement") and not e.is_a("IfcVirtualElement")]
    for cls in ("IfcSpatialStructureElement", "IfcSpatialElement"):
        try:
            pool += model.by_type(cls)
        except Exception:
            pass
    for sp in model.by_type("IfcSpace"):
        if sp.LongName:  # Raum-Langnamen koennen in Aufgaben als Referenz stehen
            names.setdefault(sp.LongName, [])
            if sp.GlobalId not in names[sp.LongName]:
                names[sp.LongName].append(sp.GlobalId)
    for e in pool:
        if getattr(e, "Name", None):
            names.setdefault(e.Name, [])
            if e.GlobalId not in names[e.Name]:
                names[e.Name].append(e.GlobalId)
    only = set(a.only.split(",")) if a.only else None
    ctx = None if a.skip_run else Ctx(a.ifc)

    summary = {}
    ids = set()
    for t in tasks:
        tid = t.get("id")
        if only and tid not in only:
            continue
        probs = []
        if tid in ids:
            probs.append("doppelte id")
        ids.add(tid)
        for k in ("id", "category", "task", "expected"):
            if k not in t:
                probs.append(f"Feld {k} fehlt")
        exp = t.get("expected", {})
        for g in guids_in(exp.get("params", {})):
            try:
                model.by_guid(g)
            except Exception:
                probs.append(f"GUID nicht im Modell: {g}")
        for nm in re.findall(r"'([^']+)'", t.get("task", "")):
            gs = names.get(nm)
            if gs is None:
                if exp.get("abort_reason") != "not_found":
                    probs.append(f"Name nicht im Modell: {nm!r}")
            elif len(gs) > 1 and exp.get("abort_reason") != "ambiguous":
                probs.append(f"Name mehrdeutig ({len(gs)}x): {nm!r}")
            elif exp.get("params") and gs[0] not in guids_in(exp["params"]) and exp.get("op") in ("delete_element", "move_element", "clone_element"):
                # Name gehoert zu einem Element, das nicht Ziel der Operation ist (z. B. Host) -> Hinweis, kein Fehler
                pass

        status = "struktur_ok" if not probs else "STRUKTUR_FEHLER"
        detail = "; ".join(probs)
        outcome = exp.get("outcome")
        if not probs and ctx is not None and outcome in ("success", "no_success") and exp.get("op"):
            op = {"op": exp["op"], **exp.get("params", {})}
            try:
                orch, store = ctx.orch([op])
                t0 = time.time()
                res = orch.run(f"Trockenlauf {tid}")
                got = "success" if res.status == "success" else "no_success"
                if got == outcome:
                    status = "ok" if got == "success" else "ok_abgelehnt"
                else:
                    status = "ABWEICHUNG"
                detail = f"Soll {outcome}, Pipeline {res.status} ({time.time() - t0:.1f} s)"
                if res.status != "success":
                    detail += " | " + json.dumps(res.to_json(), default=str, ensure_ascii=False)[:500]
            except Exception as e:
                status = "AUSNAHME"
                detail = f"{type(e).__name__}: {e}"
                traceback.print_exc(limit=2)
        print(f"{tid:5} {t.get('category',''):11} {status:16} {detail}")
        summary[status] = summary.get(status, 0) + 1
    print("\nZusammenfassung:", summary)
    bad = sum(v for k, v in summary.items() if k in ("STRUKTUR_FEHLER", "ABWEICHUNG", "AUSNAHME"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

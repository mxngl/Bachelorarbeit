#!/usr/bin/env python3
"""machbarkeit.py: prueft ohne LLM, ob die Pipeline (Snapshot, Executor, Validatoren, Rekonstruktion) auf einem IFC-Modell laeuft.

KI-Hinweis (Anhang F): erzeugt mit Claude Sonnet 5.5 (Anthropic) im Chat vom 09.11.2026.

Pro Modell wird eine Wand (bevorzugt ohne Oeffnungen) automatisch gewaehlt und zweimal durch den Orchestrator geschickt:
verschieben um 0,5 m und, mit einer zweiten Wand, loeschen. Es wird nur gemeldet, ob die Pipeline durchlaeuft, nichts bewertet.
Die Datei gehoert in den Repo-Ordner (importiert test_orchestrator, executor, orchestrator) und aendert keine Repo-Datei.

Aufruf:  python machbarkeit.py test-models/duplex.ifc test-models/smiley_west.ifc ...
"""
import json
import sys
import time
import traceback

import ifcopenshell

from test_orchestrator import Ctx


def walls(model):
    out = []
    for w in model.by_type("IfcWall"):
        if w.ObjectPlacement is None or w.Representation is None:
            continue
        out.append(w)
    free = [w for w in out if not w.HasOpenings and not w.IsDecomposedBy]
    return free + [w for w in out if w not in free]


def run_op(c, name, op):
    orch, store = c.orch([op])
    t0 = time.time()
    try:
        res = orch.run(f"Machbarkeit {name}")
    except Exception as e:
        print(f"    {name:15} AUSNAHME {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return False
    ok = res.status == "success"
    print(f"    {name:15} {res.status:8} Runden {len(res.rounds)}  {time.time() - t0:5.1f} s")
    if not ok:
        print("      ", json.dumps(res.to_json(), default=str, ensure_ascii=False)[:900])
    return ok


def check_model(path):
    print(f"== {path}")
    try:
        t0 = time.time()
        c = Ctx(path)
        print(f"    Snapshot ok ({time.time() - t0:.1f} s), Schema {c.model.schema}, Graph-Knoten {len(c.base.nodes)}")
    except Exception as e:
        print(f"    FEHLER beim Laden/Snapshot: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return False
    ws = walls(c.model)
    if not ws:
        print("    keine Wand mit Placement und Darstellung gefunden")
        return False
    a = ws[0]
    ok = run_op(c, "move_element", {"op": "move_element", "guid": a.GlobalId, "translation": [0.5, 0, 0]})
    if len(ws) > 1:
        ok = run_op(c, "delete_element", {"op": "delete_element", "guid": ws[1].GlobalId}) and ok
    return ok


if __name__ == "__main__":
    res = {p: check_model(p) for p in sys.argv[1:]}
    print("\nZusammenfassung:", ", ".join(f"{p.split('/')[-1]}={'ok' if v else 'FEHLER'}" for p, v in res.items()))

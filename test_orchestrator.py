#!/usr/bin/env python3
"""
test_orchestrator.py — prüft orchestrator.py ohne Neo4j-Verbindung (Graph im Speicher, MemoryStore).

  1. Erfolg in einer Runde, Graph danach unverändert (Standard) bzw. geändert (keep)
  2. Reparaturschleife: Executor lehnt ab → Generator bekommt die Meldung → zweite Operation gelingt
  3. Structural Validator lehnt ab (Graph absichtlich verdorben) → Rollback → Reparatur
  4. Semantic–Geometric Validator lehnt ab (Fenster ragt über die Wand) → Rollback → Reparatur
  5. Gescheitert nach max. Runden: Protokoll, Graph unverändert; Abbruch durch den Generator
  6. Mehrere Aufgaben nacheinander mit keep; Locator wird aufgerufen

KI-Hinweis: mit Unterstützung von Claude (Anthropic) erstellt, Dokumentation in Anhang F.

Nutzung:  python test_orchestrator.py AC20-FZK-Haus.ifc
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

import ifcopenshell

import executor as x
import neo4j2ifc as r
import orchestrator as o

PASSED, FAILED = [], []
W4 = "2XPyKWY018sA1ygZKgQPtU"
ST1 = "273g3wqLzDtfYIl7qqkgcO"


def check(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


class Ctx:
    def __init__(self, path):
        self.src = path
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, os.path.basename(path))
        shutil.copy(path, self.path)
        self.model = ifcopenshell.open(self.path)
        self.base = r.snapshot_from_ifc(self.model, os.path.basename(self.path))

    def orch(self, ops, **kw):
        store = o.MemoryStore(__import__("copy").deepcopy(self.base), o.sv.schema_for(self.model))
        orch = o.Orchestrator(self.path, store, generator=o.ScriptedGenerator(ops), **kw)
        orch.out_path = os.path.join(self.tmp, "out.ifc")
        return orch, store

    def same_graph(self, store):
        return store.committed.nodes == self.base.nodes and store.committed.edges == self.base.edges


def move(guid=W4, dx=0.5):
    return {"op": "move_element", "guid": guid, "translation": [dx, 0, 0]}


def case_success_reset(c):
    orch, store = c.orch([move()])
    res = orch.run("Verschiebe die Wand um 0,5 m")
    check(res.status == "success" and len(res.rounds) == 1, res.to_json())
    check(os.path.exists(res.out_ifc) and ifcopenshell.open(res.out_ifc), "Ergebnis-IFC")
    check(c.same_graph(store), "Graph muss unverändert sein (Standard)")
    check(not res.kept, "nicht behalten")
    rd = res.rounds[0]
    check(set(rd.seconds) >= {"generate", "execute", "structural", "reconstruct", "semantic"}, rd.seconds)
    json.dumps(res.to_json())
    out = ifcopenshell.open(res.out_ifc)
    import ifcopenshell.util.placement as pl
    a = pl.get_local_placement(c.model.by_guid(W4).ObjectPlacement)[0][3]
    b = pl.get_local_placement(out.by_guid(W4).ObjectPlacement)[0][3]
    check(abs((b - a) - 0.5) < 1e-6, f"Wand um {b - a} verschoben")


def case_success_keep(c):
    orch, store = c.orch([move()], keep=True)
    res = orch.run("Verschiebe die Wand")
    check(res.status == "success" and res.kept, res.to_json())
    check(not c.same_graph(store), "Graph muss die Änderung enthalten")
    # zweite Aufgabe baut auf dem neuen Zustand auf; Validatoren vergleichen mit dem Zustand davor
    orch.generator = o.ScriptedGenerator([move(dx=0.25)])
    res2 = orch.run("Noch einmal 0,25 m")
    check(res2.status == "success", res2.to_json())
    out = ifcopenshell.open(res2.out_ifc)
    import ifcopenshell.util.placement as pl
    a = pl.get_local_placement(c.model.by_guid(W4).ObjectPlacement)[0][3]
    b = pl.get_local_placement(out.by_guid(W4).ObjectPlacement)[0][3]
    check(abs((b - a) - 0.75) < 1e-6, f"kumuliert {b - a}")


def case_repair_executor(c):
    bad = {"op": "delete_element", "guid": "X" * 22}
    orch, store = c.orch([bad, move()])
    res = orch.run("Aufgabe")
    check(res.status == "success" and len(res.rounds) == 2, res.to_json())
    check(res.rounds[0].stage_failed == "executor", res.rounds[0].stage_failed)
    gen = orch.generator
    check(gen.calls[0]["feedback"] == [] and len(gen.calls[1]["feedback"]) == 1, gen.calls)
    check("fail: Executor (not_found)" in gen.calls[1]["feedback"][0], gen.calls[1]["feedback"])
    check(c.same_graph(store), "Graph unverändert")


def corrupting_factory(times):
    """Executor, der in den ersten `times` Aufrufen zusätzlich den Graphen verdirbt (Platzierung entfernen)."""
    state = {"n": 0}

    def make(backend, schema):
        ex = x.Executor(backend, schema)
        real = ex.apply

        def apply(op):
            res = real(op)
            state["n"] += 1
            if state["n"] <= times:
                guid = op["guid"]
                sid = backend.find_by_guid(guid)
                for e in backend.out_edges(sid):
                    if e.type == "ObjectPlacement":
                        backend.remove_edge(sid, "ObjectPlacement")
            return res
        ex.apply = apply
        return ex
    return make


def case_repair_structural(c):
    orch, store = c.orch([move(), move()], executor_factory=corrupting_factory(1))
    res = orch.run("Aufgabe")
    check(res.status == "success" and len(res.rounds) == 2, res.to_json())
    check(res.rounds[0].stage_failed == "structural", res.rounds[0].stage_failed)
    fb = orch.generator.calls[1]["feedback"][0]
    check(fb.startswith("fail: Strukturprüfung fehlgeschlagen") and "ObjectPlacement" in fb, fb)
    check(c.same_graph(store), "Graph unverändert")
    check("reconstruct" not in res.rounds[0].seconds, "Reconstruction läuft erst nach dem Structural Validator")


def window(offset):
    return {"op": "add_hosted_element", "ifc_class": "IfcWindow", "host_guid": W4, "offset": offset,
            "width": 0.9, "height": 1.0, "sill_height": 0.9}


def case_repair_semantic(c):
    wall = c.model.by_guid(W4)
    import ifcopenshell.util.element as ue
    length = None
    for ps, props in ue.get_psets(wall).items():
        if "Length" in props:
            length = props["Length"]
    check(length, "Wandlänge")
    # Der Executor lässt 0,5 m Überstand über die Mengenlänge zu; das Fenster ragt 0,4 m über das Wandende hinaus
    orch, store = c.orch([window(length - 0.05), window(1.0)])
    res = orch.run("Fenster einsetzen")
    check(res.status == "success" and len(res.rounds) == 2, res.to_json())
    check(res.rounds[0].stage_failed == "semantic", res.rounds[0].stage_failed)
    fb = orch.generator.calls[1]["feedback"][0]
    check("Geometrieprüfung fehlgeschlagen" in fb and "opening_outside_wall" in fb, fb)
    check(c.same_graph(store), "Graph unverändert")


def case_failed_logged(c):
    log = os.path.join(c.tmp, "failed.jsonl")
    bad = {"op": "delete_element", "guid": "X" * 22}
    orch, store = c.orch([bad], max_rounds=3, log_path=log)
    res = orch.run("Unlösbar")
    check(res.status == "failed" and len(res.rounds) == 3, res.to_json())
    check(res.out_ifc is None, "keine Ergebnis-IFC")
    check(c.same_graph(store), "Graph unverändert")
    lines = [json.loads(l) for l in open(log, encoding="utf-8")]
    check(len(lines) == 1 and lines[0]["status"] == "failed" and len(lines[0]["rounds"]) == 3, lines)
    check(lines[0]["rounds"][2]["feedback"].startswith("fail: Executor"), lines[0]["rounds"][2])
    check(len(orch.generator.calls[2]["feedback"]) == 2, "Generator sieht die beiden bisherigen Meldungen")


def case_abort(c):
    orch, store = c.orch([{"op": "abort", "reason": "Aufgabe nicht ausführbar"}])
    res = orch.run("Mach etwas Unmögliches")
    check(res.status == "aborted" and "nicht ausführbar" in res.reason, res.to_json())
    check(res.out_ifc is None and c.same_graph(store), "nichts geschrieben")


def case_locator_and_sequence(c):
    seen = []

    class Loc(o.LocatorBase):
        def locate(self, task, snapshot):
            seen.append((task, len(snapshot.nodes)))
            return {"guid": W4}

    orch, store = c.orch([move()], keep=True)
    orch.locator = Loc()
    r1 = orch.run("Aufgabe 1")
    r2 = orch.run("Aufgabe 2")
    check(r1.status == r2.status == "success", (r1.status, r2.status))
    check([s[0] for s in seen] == ["Aufgabe 1", "Aufgabe 2"] and seen[0][1] == len(c.base.nodes), seen)
    check(orch.generator.calls[0]["located"] == {"guid": W4}, orch.generator.calls[0])


def case_unknown_operation(c):
    orch, store = c.orch([{"op": "erfinde_etwas"}], max_rounds=2)
    res = orch.run("Aufgabe")
    check(res.status == "failed" and "unknown_op" in res.rounds[0].feedback, res.rounds[0].feedback)


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
            traceback.print_exc(limit=4)
    shutil.rmtree(c.tmp, ignore_errors=True)
    print(f"\n{len(PASSED)}/{len(PASSED) + len(FAILED)} bestanden")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "AC20-FZK-Haus.ifc"))

#!/usr/bin/env python3
"""
orchestrator.py — Orchestrator: ruft die Stationen des Pipeline-Diagramms der Reihe nach auf.

  Natural-language task → Locator (LLM, nur lesend) → Generator (LLM, eine Operation oder Abbruch)
  → Executor → Structural Validator → Reconstruction → Semantic–Geometric Validator → Ergebnis

Fehler von Executor und beiden Validatoren gehen als "fail: error message" an den Generator zurück
(Repair loop), höchstens `max_rounds` Runden. Danach gilt die Aufgabe als gescheitert und wird protokolliert
(Failed task, logged).

Entscheidungen (Max, 02.11.2026):
  * Der Graph wird in einer offenen Transaktion geändert. Structural Validator und Reconstruction lesen den
    Zustand aus derselben Transaktion. Erst wenn beide Validatoren bestanden haben, wird committet (mit
    keep=True), sonst Rollback. Der Graph in der Datenbank ist nie halb geändert.
  * Nach Erfolg wird standardmäßig zurückgerollt (jede Aufgabe unabhängig, für Benchmarks); die Ergebnis-IFC wird
    geschrieben. Mit keep=True bleibt die Änderung im Graphen (fortlaufende Bearbeitung).

Locator und Generator sind austauschbare Objekte (siehe LocatorBase, GeneratorBase). Dieser Stand enthält nur
Platzhalter ohne Sprachmodell (ScriptedGenerator: feste Operationen je Runde), damit die Schleife getestet werden
kann. Die Anbindung des lokalen Modells folgt.

Baseline der Validatoren: Zustand VOR der Operation (Graph bzw. zuletzt gültiges IFC). "Neu" heißt damit: von
dieser Operation verursacht.

KI-Hinweis (TUHH-Handreichung "KI-Tools in Studium und Lehre"): mit Unterstützung von Claude (Anthropic)
erstellt. Prompts und Ausgaben gehören in Anhang F (AI usage documentation).

Nutzung (Neo4j, Passwort in NEO4J_PASSWORD):
  python orchestrator.py test-models\\AC20-FZK-Haus.ifc --op "{\\"op\\":\\"delete_element\\",\\"guid\\":\\"...\\"}" --out ergebnis_orch.ifc
  python orchestrator.py test-models\\AC20-FZK-Haus.ifc --ops-file ops.json --rounds 3 --keep --log aufgaben.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import ifcopenshell
from ifcopenshell import ifcopenshell_wrapper as wrapper

import executor as x
import ifc2neo4j as g
import neo4j2ifc as r
import structural_validator as sv
import semantic_geometric_validator as sg


# =========================================================================== Speicher (Transaktion)

class MemoryStore:
    """Graph im Speicher (für Tests und ohne Datenbank). begin() arbeitet auf einer Kopie."""

    def __init__(self, snapshot: r.Snapshot, schema):
        self.committed = snapshot
        self.schema = schema
        self._work: r.Snapshot | None = None
        self._cache: dict = {}

    def snapshot(self) -> r.Snapshot:
        return copy.deepcopy(self._work if self._work is not None else self.committed)

    def begin(self) -> x.MemoryBackend:
        self._work = copy.deepcopy(self.committed)
        return x.MemoryBackend(self._work, lambda cls: set(g.label_chain(self.schema, cls, self._cache)) | {"IfcEntity"})

    def commit(self) -> None:
        self.committed, self._work = self._work, None

    def rollback(self) -> None:
        self._work = None

    def close(self) -> None:
        pass


def snapshot_from_tx(tx, source: str) -> r.Snapshot:
    """Wie neo4j2ifc.snapshot_from_neo4j, liest aber innerhalb einer offenen Transaktion (sieht ungespeicherte Änderungen)."""
    snap = r.Snapshot()
    for rec in tx.run("MATCH (n:IfcEntity {source_file: $f}) RETURN properties(n) AS p", f=source):
        p = dict(rec["p"])
        p.pop("source_file", None)
        snap.nodes[p["step_id"]] = p
    for rec in tx.run(
        "MATCH (a:IfcEntity {source_file: $f})-[r]->(b:IfcEntity {source_file: $f}) WHERE r.inverse IS NULL "
        "RETURN a.step_id AS src, b.step_id AS dst, type(r) AS t, r.idx AS i1, r.idx2 AS i2", f=source):
        key = (rec["src"], rec["t"], rec["i1"], rec["i2"])
        if key in snap.edges:
            snap.collisions += 1
        snap.edges[key] = rec["dst"]
    return snap


class Neo4jStore:
    def __init__(self, uri: str, user: str, password: str, database: str | None, source: str):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password), notifications_min_severity="OFF")
        self.driver.verify_connectivity()
        self.database, self.source = database, source
        self.session = None
        self.tx = None

    def _read(self):
        if self.tx is not None:
            return snapshot_from_tx(self.tx, self.source)
        with self.driver.session(database=self.database) as s:
            tx = s.begin_transaction()
            try:
                return snapshot_from_tx(tx, self.source)
            finally:
                tx.rollback()

    def snapshot(self) -> r.Snapshot:
        return self._read()

    def begin(self) -> x.Neo4jBackend:
        self.session = self.driver.session(database=self.database)
        self.tx = self.session.begin_transaction()
        return x.Neo4jBackend(self.tx, self.source)

    def commit(self) -> None:
        self.tx.commit()
        self._end()

    def rollback(self) -> None:
        if self.tx is not None:
            try:
                self.tx.rollback()
            finally:
                self._end()

    def _end(self) -> None:
        self.tx = None
        if self.session is not None:
            self.session.close()
            self.session = None

    def close(self) -> None:
        self.rollback()
        self.driver.close()


# =========================================================================== Platzhalter für Locator und Generator

class LocatorBase:
    """Locator (LLM, nur Lesezugriff): liefert Kontext für den Generator (z. B. GlobalIds der Zielbauteile)."""

    def locate(self, task: str, snapshot: r.Snapshot) -> dict:
        return {}


class GeneratorBase:
    """Generator (LLM): genau eine Operation als dict, oder {"op": "abort", "reason": ...}."""

    def generate(self, task: str, located: dict, feedback: list[str]) -> dict:
        raise NotImplementedError


class ScriptedGenerator(GeneratorBase):
    """Feste Operationen je Runde (Platzhalter ohne Modell). Nach der letzten wird die letzte wiederholt."""

    def __init__(self, ops: list[dict]):
        self.ops = ops
        self.calls: list[dict] = []

    def generate(self, task, located, feedback):
        self.calls.append({"task": task, "located": located, "feedback": list(feedback)})
        return self.ops[min(len(self.calls) - 1, len(self.ops) - 1)]


# =========================================================================== Ergebnis

@dataclass
class RoundLog:
    round: int
    op: dict | None = None
    stage_failed: str | None = None          # executor | structural | semantic | generator
    feedback: str = ""
    seconds: dict = field(default_factory=dict)
    exec_summary: str = ""
    structural: str = ""
    semantic: str = ""

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class TaskResult:
    task: str
    status: str                              # success | failed | aborted
    rounds: list[RoundLog] = field(default_factory=list)
    out_ifc: str | None = None
    kept: bool = False
    reason: str = ""
    seconds: float = 0.0

    @property
    def op(self) -> dict | None:
        return self.rounds[-1].op if self.rounds else None

    def to_json(self) -> dict:
        return {"task": self.task, "status": self.status, "reason": self.reason, "out_ifc": self.out_ifc,
                "kept": self.kept, "seconds": round(self.seconds, 3), "rounds": [r_.to_json() for r_ in self.rounds]}


# =========================================================================== Orchestrator

class Orchestrator:
    def __init__(self, ifc_path: str, store, locator: LocatorBase | None = None, generator: GeneratorBase | None = None,
                 max_rounds: int = 3, keep: bool = False, log_path: str | None = None,
                 executor_factory: Callable | None = None):
        self.ifc_path = ifc_path
        self.source = os.path.basename(ifc_path)
        self.store = store
        self.locator = locator or LocatorBase()
        self.generator = generator
        self.max_rounds = max_rounds
        self.keep = keep
        self.log_path = log_path
        self.executor_factory = executor_factory or (lambda backend, schema: x.Executor(backend, schema))
        self.original = ifcopenshell.open(ifc_path)
        self.schema = sv.schema_for(self.original)
        self.base_snapshot = r.snapshot_from_ifc(self.original, self.source)    # "so wurde der Graph geladen"
        self.current_model = self.original                                      # zuletzt gültiges IFC
        self._sg: sg.SemanticGeometricValidator | None = None

    # -- eine Aufgabe -------------------------------------------------------------------------
    def run(self, task: str, generator: GeneratorBase | None = None) -> TaskResult:
        gen = generator or self.generator
        if gen is None:
            raise ValueError("kein Generator angegeben")
        t_task = time.time()
        res = TaskResult(task=task, status="failed")
        prev = self.store.snapshot()                                   # Zustand vor der Aufgabe
        located = self.locator.locate(task, prev)
        if isinstance(located, dict) and located.get("status") == "abort":      # Locator-Abbruch: getrennt protokolliert
            res.rounds.append(RoundLog(0, stage_failed="locator", feedback=str(located.get("abort", ""))))
            res.status, res.reason = "aborted", "Locator: " + str(located.get("abort", {}).get("reason", ""))
            res.seconds = time.time() - t_task
            self._log(res)
            return res
        structural = sv.StructuralValidator(self.schema, prev)
        semantic = self._semantic()
        feedback: list[str] = []
        for n in range(1, self.max_rounds + 1):
            rd = RoundLog(n)
            res.rounds.append(rd)
            t0 = time.time()
            op = gen.generate(task, located, feedback)
            rd.seconds["generate"] = time.time() - t0
            rd.op = op
            if not isinstance(op, dict) or op.get("op") == "abort":
                rd.stage_failed = "generator"
                res.status, res.reason = "aborted", (op or {}).get("reason", "Generator hat abgebrochen") if isinstance(op, dict) else "keine Operation"
                break
            ok, msg = self._attempt(op, rd, prev, structural, semantic, res)
            if ok:
                res.status = "success"
                break
            rd.feedback = msg
            feedback.append(msg)
        else:
            res.reason = f"Nach {self.max_rounds} Runden nicht bestanden"
        res.seconds = time.time() - t_task
        self._log(res)
        return res

    # -- ein Versuch (Executor → Structural → Reconstruction → Semantic–Geometric) --------------
    def _attempt(self, op, rd: RoundLog, prev, structural, semantic, res) -> tuple[bool, str]:
        t = time.time()
        backend = self.store.begin()
        committed = False
        try:
            try:
                er = self.executor_factory(backend, self.schema).apply(op)
            except x.ExecutorError as e:
                rd.stage_failed = "executor"
                return False, f"fail: Executor ({e.code}): {e.message}"
            rd.seconds["execute"] = time.time() - t
            rd.exec_summary = er.summary()

            t = time.time()
            snap = self.store.snapshot()                               # aus derselben Transaktion
            srep = structural.validate(snap, op)
            rd.seconds["structural"] = time.time() - t
            rd.structural = srep.summary()
            if not srep.ok:
                rd.stage_failed = "structural"
                return False, "fail: " + srep.feedback()

            t = time.time()
            model = ifcopenshell.open(self.ifc_path)
            diff = r.diff_snapshots(self.base_snapshot, snap)
            r.apply_diff(model, self.base_snapshot, snap, diff)
            rd.seconds["reconstruct"] = time.time() - t

            t = time.time()
            grep = semantic.validate(model, op)
            rd.seconds["semantic"] = time.time() - t
            rd.semantic = grep.summary()
            if not grep.ok:
                rd.stage_failed = "semantic"
                return False, "fail: " + grep.feedback()

            out = self._out_path()
            model.write(out)
            res.out_ifc = out
            if self.keep:
                self.store.commit()
                committed = True
                self.current_model = model
                self._sg = None
                res.kept = True
            return True, ""
        finally:
            if not committed:
                self.store.rollback()

    # -- Hilfen --------------------------------------------------------------------------------
    def _semantic(self) -> sg.SemanticGeometricValidator:
        if self._sg is None:
            self._sg = sg.SemanticGeometricValidator(self.current_model)
        return self._sg

    out_path: str | None = None

    def _out_path(self) -> str:
        if self.out_path:
            return self.out_path
        stem = os.path.splitext(self.ifc_path)[0]
        return stem + "_orchestrator.ifc"

    def _log(self, res: TaskResult) -> None:
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res.to_json(), ensure_ascii=False) + "\n")


# =========================================================================== CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Orchestrator: Aufgabe durch die Pipeline schicken (Platzhalter-Generator)")
    ap.add_argument("ifc", help="IFC-Datei (Ausgangsmodell; der Dateiname ist der source_file im Graph)")
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    ap.add_argument("--database", default=None)
    ap.add_argument("--task", default="(ohne Aufgabentext)", help="Aufgabentext (für Protokoll und späteren Generator)")
    ap.add_argument("--op", help="eine Operation als JSON (wird in jeder Runde wiederholt)")
    ap.add_argument("--ops-file", help="JSON-Liste von Operationen, je Runde die nächste (Reparaturschleife testen)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--keep", action="store_true", help="Änderung nach Erfolg im Graphen behalten (Standard: zurückrollen)")
    ap.add_argument("--out", help="Ergebnis-IFC (Standard: <Modell>_orchestrator.ifc)")
    ap.add_argument("--log", help="JSONL-Protokoll der Aufgaben")
    args = ap.parse_args(argv)

    if args.ops_file:
        ops = json.load(open(args.ops_file, encoding="utf-8"))
    elif args.op:
        ops = [json.loads(args.op)]
    else:
        ap.error("--op oder --ops-file angeben")
    source = os.path.basename(args.ifc)
    store = Neo4jStore(args.uri, args.user, args.password, args.database, source)
    try:
        orch = Orchestrator(args.ifc, store, generator=ScriptedGenerator(ops), max_rounds=args.rounds,
                            keep=args.keep, log_path=args.log)
        orch.out_path = args.out
        res = orch.run(args.task)
    finally:
        store.close()
    for rd in res.rounds:
        t = ", ".join(f"{k} {v:.1f} s" for k, v in rd.seconds.items())
        print(f"Runde {rd.round}: {rd.op.get('op') if rd.op else '-'} → "
              f"{'bestanden' if rd.stage_failed is None else 'abgelehnt von ' + rd.stage_failed} ({t})")
        if rd.feedback:
            print("  " + rd.feedback.replace("\n", "\n  "))
    print(f"Ergebnis: {res.status}" + (f" ({res.reason})" if res.reason else "") + f", {res.seconds:.1f} s")
    if res.out_ifc:
        print("Ergebnis-IFC:", res.out_ifc)
    print("Graph:", "Änderung behalten (committet)" if res.kept else "unverändert (zurückgerollt)")
    return 0 if res.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())

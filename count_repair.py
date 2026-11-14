#!/usr/bin/env python3
"""
count_repair.py — zählt aus den Aufgaben-Protokollen (JSONL von orchestrator/llm_stations), wie oft die
Reparaturschleife benutzt wurde und wie die Aufgaben danach endeten.

Aufruf ohne Argumente: liest alle Dateien runs*.jsonl im aktuellen Ordner (ohne *.llm.jsonl).
Aufruf mit Argumenten: liest genau diese Dateien.
Hinweis: "success" heißt nur, dass alle Prüfungen bestanden wurden. Ob das Ergebnis zur Ground Truth passt,
sagt auswertung.py.
"""
import glob
import json
import sys
from collections import Counter

files = sys.argv[1:] or [f for f in sorted(glob.glob("runs*.jsonl")) if not f.endswith(".llm.jsonl")]
if not files:
    sys.exit("Keine Dateien runs*.jsonl gefunden.")

for path in files:
    tasks = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    in_loop = [t for t in tasks if len([r for r in t["rounds"] if r["round"] >= 1]) > 1]
    ok_after = [t for t in in_loop if t["status"] == "success"]
    first_try_fail = Counter()
    for t in in_loop:
        first = next(r for r in t["rounds"] if r["round"] == 1)
        first_try_fail[first.get("stage_failed")] += 1
    print(f"== {path}")
    print(f"Aufgaben gesamt:                           {len(tasks)}")
    print(f"  mit Reparaturschleife (mehr als 1 Runde): {len(in_loop)}")
    print(f"    davon am Ende success:                  {len(ok_after)}")
    print(f"    davon am Ende failed/aborted:           {len(in_loop) - len(ok_after)}")
    print(f"  Auslöser in Runde 1 (Station):            {dict(first_try_fail)}")
    for t in in_loop:
        print(f"    - {t['status']:8s} {len(t['rounds'])} Runden | {t['task'][:70]}")

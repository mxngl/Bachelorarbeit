#!/usr/bin/env python3
"""vergleich_laeufe.py: vergleicht zwei Pipeline-Logs (runs_*.jsonl) Aufgabe fuer Aufgabe.

KI-Hinweis (Anhang F): erzeugt mit Claude Sonnet 5.5 (Anthropic) im Chat vom 09.11.2026.

Aufruf:
    python vergleich_laeufe.py runs_A.jsonl runs_B.jsonl [--ohne-freitext]

Verglichen werden Status, Grund und die Operationen aller Runden. Mit --ohne-freitext wird das Freitextfeld 'detail'
der Abbruch-Operationen ausgeklammert (das Modell formuliert es jedes Mal neu). Zuordnung ueber den Aufgabentext,
bei mehreren Logzeilen je Text zaehlt die letzte.
"""
import argparse
import json


def load(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            d[r["task"]] = r
    return d


def sig(run, ohne_freitext):
    ops = []
    for x in run.get("rounds", []):
        op = x.get("op")
        if ohne_freitext and isinstance(op, dict):
            op = {k: v for k, v in op.items() if k != "detail"}
        ops.append(op)
    return run["status"], run.get("reason"), json.dumps(ops, sort_keys=True, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--ohne-freitext", action="store_true")
    o = ap.parse_args()
    a, b = load(o.a), load(o.b)
    common = [t for t in a if t in b]
    diff = [t for t in common if sig(a[t], o.ohne_freitext) != sig(b[t], o.ohne_freitext)]
    print(f"Aufgaben in beiden Logs: {len(common)} (nur in A: {len(set(a) - set(b))}, nur in B: {len(set(b) - set(a))})")
    print(f"identisch: {len(common) - len(diff)} | abweichend: {len(diff)}" + (" (ohne Freitext 'detail')" if o.ohne_freitext else ""))
    for t in diff:
        print("-", t[:80])
        print("   A:", sig(a[t], o.ohne_freitext)[:2])
        print("   B:", sig(b[t], o.ohne_freitext)[:2])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""auswertung.py: vergleicht ein Pipeline-Log (runs_*.jsonl) mit der Ground Truth (tasks_*.json).

KI-Hinweis (Anhang F): erzeugt mit Claude Sonnet 5.5 (Anthropic) im Chat vom 09.11.2026.

Aufruf:
    python auswertung.py tasks_fzk.json runs_MODELTAG.jsonl [--tol 0.01] [--csv out.csv] [--json out.json]
                         [--original test-models/AC20-FZK-Haus.ifc --results-dir ergebnisse/MODELTAG/fzk]

Mit --original und --results-dir wird bei move_element die Wirkung geprueft: Weicht die Operation des Modells nur in den
Parametern von expected ab (z. B. translation statt target_location), zaehlt sie als richtig, wenn der Ursprung des
Elements im Ergebnis-IFC dort liegt, wo expected ihn hinstellt. Die Pruefung braucht ifcopenshell.

Zuordnung: Das Log enthaelt keine Aufgaben-ID, nur den Aufgabentext. Die Zuordnung laeuft ueber den Text.
Enthaelt das Log mehrere Laeufe (--log haengt an), zaehlt je Aufgabentext die letzte Zeile.

Ergebnis je Aufgabe:
    richtig                      Ausgang wie erwartet (bei success zusaetzlich Operation und Parameter gleich)
    falsche_operation            success, aber Operation/Parameter weichen von expected ab
    pipeline_lehnt_ab            erwartet success, erste Operation des Modells stimmt, Pipeline lehnt trotzdem ab
    falsche_operation_abgelehnt  erwartet success, erste Operation weicht ab und Pipeline lehnt ab
    unerwarteter_abbruch         erwartet success, Locator/Generator brechen ab
    kein_abbruch                 erwartet abort, Pipeline liefert aber success
    abbruch_abweichend           erwartet abort, Abbruch mit anderem Grund oder anderer Stufe
    abbruch_erwartet_failed      erwartet abort, Aufgabe scheitert nach den Runden
    unerwarteter_erfolg          erwartet no_success, Pipeline liefert success
    nicht_gelaufen               kein Logeintrag zur Aufgabe
Nur die in expected.params genannten Parameter werden verglichen, Zahlen mit Toleranz --tol (Meter).
"""
import argparse
import csv
import json
import re
import os
import sys
from collections import defaultdict

KNOWN_REASONS = ("ambiguous", "not_found", "out_of_scope", "insufficient_information")
CTX = {"original": None, "results_dir": None}
_MODELS = {}


def _model(path):
    if path not in _MODELS:
        import ifcopenshell
        _MODELS[path] = ifcopenshell.open(path)
    return _MODELS[path]


def origin_m(path, guid):
    """Welt-Ursprung des Placements eines Elements in Metern, None wenn nicht ermittelbar."""
    import ifcopenshell.util.placement as placement
    import ifcopenshell.util.unit as unit
    m = _model(path)
    try:
        el = m.by_guid(guid)
    except RuntimeError:
        return None
    if el is None or getattr(el, "ObjectPlacement", None) is None:
        return None
    sc = unit.calculate_unit_scale(m)
    mat = placement.get_local_placement(el.ObjectPlacement)
    return [float(mat[i][3]) * sc for i in range(3)]


def move_effect(exp, run, tol):
    """Prueft die Wirkung einer Verschiebung im Ergebnis-IFC. Rueckgabe (ok, Text) oder (None, Grund)."""
    if not (CTX["original"] and CTX["results_dir"]):
        return None, "Wirkungspruefung nicht aktiv"
    p = exp.get("params") or {}
    out = run.get("out_ifc") or ""
    cand = [os.path.join(CTX["results_dir"], os.path.basename(out)), out]
    path = next((c for c in cand if c and os.path.isfile(c)), None)
    if path is None:
        return None, "Ergebnis-IFC nicht gefunden"
    before, after = origin_m(CTX["original"], p["guid"]), origin_m(path, p["guid"])
    if before is None or after is None:
        return None, "Ursprung nicht ermittelbar"
    if "target_location" in p:
        want = [float(v) for v in p["target_location"]]
    elif "translation" in p:
        want = [b + float(t) for b, t in zip(before, p["translation"])]
    else:
        return None, "expected ohne translation/target_location"
    fmt = lambda v: "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
    return all(abs(a - w) <= tol for a, w in zip(after, want)), f"Ursprung {fmt(after)}, erwartet {fmt(want)}"


def close(exp, act, tol):
    """exp (Soll) ist Teilmenge von act (Ist). Zahlen mit Toleranz, Listen elementweise, Dicts per Schluessel."""
    if isinstance(exp, bool) or isinstance(act, bool):
        return exp == act
    if isinstance(exp, (int, float)) and isinstance(act, (int, float)):
        return abs(exp - act) <= tol
    if isinstance(exp, (list, tuple)) and isinstance(act, (list, tuple)):
        return len(exp) == len(act) and all(close(a, b, tol) for a, b in zip(exp, act))
    if isinstance(exp, dict) and isinstance(act, dict):
        return all(k in act and close(v, act[k], tol) for k, v in exp.items())
    return exp == act


def op_matches(op, exp, tol):
    if not op:
        return False, ["keine Operation"]
    diffs = []
    if op.get("op") != exp.get("op"):
        diffs.append(f"op: {op.get('op')} statt {exp.get('op')}")
    for k, v in (exp.get("params") or {}).items():
        if k not in op:
            diffs.append(f"{k}: fehlt")
        elif not close(v, op[k], tol):
            diffs.append(f"{k}: {op[k]} statt {v}")
    return not diffs, diffs


def ops_of(run):
    return [r["op"] for r in run.get("rounds", []) if r.get("op")]


def codes(run):
    """Befund-Codes wie [G5/existing_changed] aus dem Feedback aller Runden, in Reihenfolge, ohne Dubletten."""
    seen = []
    for r in run.get("rounds", []):
        for c in re.findall(r"\[([A-Za-z0-9_]+/[A-Za-z0-9_]+)\]", str(r.get("feedback", ""))):
            if c not in seen:
                seen.append(c)
    return seen


def abort_info(run):
    rounds = run.get("rounds") or [{}]
    last = rounds[-1]
    reason = (run.get("reason") or "").lower()
    stage = "locator" if reason.startswith("locator") or last.get("stage_failed") == "locator" else "generator"
    blob = reason + " " + str(last.get("feedback", "")).lower()
    why = next((k for k in KNOWN_REASONS if k in reason), None) or next((k for k in KNOWN_REASONS if k in blob), reason)
    return stage, why


def classify(exp, run, tol):
    st, outcome = run["status"], exp.get("outcome")
    ops = ops_of(run)
    if outcome == "success":
        if st == "success":
            ok, d = op_matches(ops[-1] if ops else None, exp, tol)
            last = ops[-1] if ops else {}
            if (not ok and exp.get("op") == "move_element" and last.get("op") == "move_element"
                    and last.get("guid") == (exp.get("params") or {}).get("guid")):
                eff, txt = move_effect(exp, run, tol)
                if eff:
                    return "richtig", ["Wirkung geprueft: " + txt], True
                d = d + ["Wirkung: " + txt]
            return ("richtig" if ok else "falsche_operation"), d, True if ok else None
        first_ok, d = op_matches(ops[0] if ops else None, exp, tol)
        if st == "aborted":
            stage, why = abort_info(run)
            return "unerwarteter_abbruch", [f"{stage}: {why}"], None
        extra = ["Codes: " + " > ".join(codes(run))] if codes(run) else [run.get("reason", "")]
        return ("pipeline_lehnt_ab" if first_ok else "falsche_operation_abgelehnt"), d + extra, first_ok
    if outcome == "abort":
        if st == "aborted":
            stage, why = abort_info(run)
            d = []
            if why != exp.get("abort_reason"):
                d.append(f"grund: {why} statt {exp.get('abort_reason')}")
            if exp.get("stage") and stage != exp["stage"]:
                d.append(f"stufe: {stage} statt {exp['stage']}")
            return ("richtig" if not d else "abbruch_abweichend"), d, None
        if st == "success":
            return "kein_abbruch", [f"stattdessen {ops[-1] if ops else None}"], None
        return "abbruch_erwartet_failed", [run.get("reason", "")], None
    if outcome == "no_success":
        return ("richtig" if st != "success" else "unerwarteter_erfolg"), [], None
    return "unbekannt", [f"expected.outcome={outcome}"], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks")
    ap.add_argument("runs")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--csv")
    ap.add_argument("--json")
    ap.add_argument("--original", help="Original-IFC (fuer die Wirkungspruefung bei move_element)")
    ap.add_argument("--results-dir", help="Ordner mit den Ergebnis-IFCs dieses Laufs")
    a = ap.parse_args()
    CTX["original"], CTX["results_dir"] = a.original, a.results_dir

    tasks = json.load(open(a.tasks, encoding="utf-8"))
    rows = [json.loads(l) for l in open(a.runs, encoding="utf-8") if l.strip()]
    by_text = defaultdict(list)
    for r in rows:
        by_text[r["task"]].append(r)
    for t, q in by_text.items():
        if len(q) > 1:
            print(f"Hinweis: {len(q)} Logzeilen zu '{t}', nehme die letzte", file=sys.stderr)
    known = {t["task"] for t in tasks}
    for t in by_text:
        if t not in known:
            print(f"Hinweis: Logzeile ohne Aufgabe in {a.tasks}: '{t}'", file=sys.stderr)

    res = []
    for t in tasks:
        q = by_text.get(t["task"])
        if not q:
            res.append(dict(id=t["id"], kategorie=t["category"], erwartet=t["expected"].get("outcome"), status="-",
                            runden=0, sekunden=0.0, ergebnis="nicht_gelaufen", details="", erste_op_ok=None))
            continue
        run = q[-1]
        label, diffs, first_ok = classify(t["expected"], run, a.tol)
        res.append(dict(id=t["id"], kategorie=t["category"], erwartet=t["expected"].get("outcome"),
                        status=run["status"], runden=len([r for r in run.get("rounds", []) if r.get("round", 0) >= 1]),
                        sekunden=round(run.get("seconds", 0.0), 1), ergebnis=label, details="; ".join(diffs),
                        erste_op_ok=first_ok))

    print(f"{'id':4} {'kategorie':11} {'erwartet':10} {'status':8} {'R':>2} {'s':>6}  {'ergebnis':27} details")
    for r in res:
        print(f"{r['id']:4} {r['kategorie']:11} {str(r['erwartet']):10} {r['status']:8} {r['runden']:>2} "
              f"{r['sekunden']:>6}  {r['ergebnis']:27} {r['details']}")

    n = len(res)
    ok = sum(r["ergebnis"] == "richtig" for r in res)
    summary = {"aufgaben": n, "richtig": ok, "quote": round(ok / n, 3) if n else None,
               "gesamtzeit_s": round(sum(r["sekunden"] for r in res), 1), "kategorien": {}, "ergebnisse": {}}
    cats = defaultdict(list)
    for r in res:
        cats[r["kategorie"]].append(r)
        summary["ergebnisse"][r["ergebnis"]] = summary["ergebnisse"].get(r["ergebnis"], 0) + 1
    print("\nGesamt: %d von %d richtig (%.1f %%), Gesamtzeit %.0f s" % (ok, n, 100 * ok / n if n else 0, summary["gesamtzeit_s"]))
    for c, lst in cats.items():
        k = sum(r["ergebnis"] == "richtig" for r in lst)
        summary["kategorien"][c] = {"aufgaben": len(lst), "richtig": k}
        print(f"  {c:11} {k} von {len(lst)}")
    print("Ergebnisse:", ", ".join(f"{k}={v}" for k, v in sorted(summary["ergebnisse"].items())))
    ab = [r for r in res if r["erwartet"] == "abort"]
    if ab:
        ex = sum(r["ergebnis"] == "richtig" for r in ab)
        ue = sum(r["status"] == "aborted" for r in ab)
        print(f"Abbrueche (erwartet abort): exakt (Grund und Stufe) {ex} von {len(ab)}, ueberhaupt abgebrochen {ue} von {len(ab)}")
        lenient = ok + sum(r["ergebnis"] == "abbruch_abweichend" for r in res)
        print(f"Gesamt, wenn jeder Abbruch zaehlt: {lenient} von {n} ({100 * lenient / n:.1f} %)")
        summary["abbrueche"] = {"erwartet": len(ab), "exakt": ex, "ueberhaupt": ue, "gesamt_wenn_jeder_abbruch_zaehlt": lenient}
    rep = [r for r in res if r["ergebnis"] == "richtig" and r["runden"] > 1]
    print(f"Richtig erst nach Reparatur (mehr als 1 Runde): {len(rep)}")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
            w.writeheader()
            w.writerows(res)
    if a.json:
        json.dump({"summary": summary, "aufgaben": res}, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
structural_validator.py — Structural Validator (Stufe 1 der zweistufigen Prüfung nach dem Executor).

Aufgabe: Ist die Graph-Darstellung nach einer Operation noch *strukturell* ein gültiges IFC-Modell?
Geometrie und Bedeutung (Überlappung, Lage der Öffnung in der Wand usw.) prüft der Semantic-Geometric
Validator. Dieser Validator liest nur den Graph (Knoten, Properties, Kanten), nie die Datei.

Vier Regelgruppen (Befund-Codes siehe RULES):
  A  Referenzen und Listen      Kante verweist auf vorhandenen Knoten, Attributname gehört zur Klasse,
                                Zielklasse passt zum Attributtyp, idx/idx2 lückenlos und zur Listentiefe passend
  B  Pflichtangaben und Werte   Pflichtattribute, GlobalId (22 Zeichen, eindeutig), Enum-Werte, Werttypen,
                                Listengrenzen, keine Platzierungszyklen, unbekannte Properties
  C  Beziehungsregeln           Öffnung hat genau eine Wand und höchstens eine Füllung, Bauteil in genau einem
                                Geschoss, höchstens ein Typ, Platzierung vorhanden, Tür/Fenster mit Füllung,
                                verwaiste Property-Sets/Properties/Platzierungen (Eigenbeitrag, Kap. 4.6)
  D  Differenzprüfung           Was die Operation an Knoten und Kanten ändern darf (Fußabdruck je Operation),
                                gemessen an der Differenz zum Ausgangsmodell

Nur *neue* Befunde zählen: Der Validator bildet beim Anlegen das Ausgangsmodell (Baseline) ab und merkt sich
dessen Befunde. Tritt ein Befund mit gleichem Schlüssel (Code, Knoten, Gegenstand) auch nach der Operation auf,
ist er "existing" und blockiert nicht. Das Original-FZK-Haus ist nicht fehlerfrei (schema_check: 5 Fehler);
ein Validator, der das nicht berücksichtigt, würde jede Operation ablehnen.

Die Regeln laufen in Python auf einem Snapshot (zwei Abfragen an Neo4j, oder im Speicher). Für die Regeln
aus Gruppe C gibt es zu jeder Regel eine gleichwertige Cypher-Abfrage (CYPHER); `--cypher-check` führt sie
auf der Datenbank aus und vergleicht die Treffer mit den Python-Befunden.

Nutzung:
  python structural_validator.py AC20-FZK-Haus.ifc                      # Graph gegen Ausgangsmodell prüfen
  python structural_validator.py AC20-FZK-Haus.ifc --op '{"op":"delete_element","guid":"..."}'
  python structural_validator.py AC20-FZK-Haus.ifc --json bericht.json --cypher-check
  python structural_validator.py --list-rules

KI-Hinweis (TUHH-Handreichung "KI-Tools in Studium und Lehre"): mit Unterstützung von Claude (Anthropic)
erstellt. Prompts und Ausgaben gehören in Anhang F (AI usage documentation).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import ifcopenshell
from ifcopenshell import ifcopenshell_wrapper as w

import ifc2neo4j as g
import neo4j2ifc as r

META_KEYS = r.META_KEYS
CUT = g.GEOMETRY_CUT_ATTRIBUTES
GUID_RE = re.compile(r"^[0-9A-Za-z_$]{22}$")
ERROR, WARNING = "error", "warning"


# =========================================================================== Regelkatalog

# code -> (Gruppe, Schweregrad, Kurzbeschreibung)
RULES: dict[str, tuple[str, str, str]] = {
    # A
    "unknown_class":          ("A", ERROR,   "ifc_class ist im Schema nicht definiert"),
    "dangling_edge":          ("A", ERROR,   "Kante zeigt auf einen nicht vorhandenen Knoten"),
    "unknown_edge_attribute": ("A", ERROR,   "Kantenname ist kein (abgeleitetes oder inverses) Attribut der Klasse"),
    "edge_on_value_attribute": ("A", ERROR,  "Kante auf einem Attribut, das keine Referenz tragen darf"),
    "edge_target_class":      ("A", ERROR,   "Zielklasse passt nicht zum Typ des Attributs"),
    "edge_index_shape":       ("A", ERROR,   "idx/idx2 passen nicht zur Listentiefe des Attributs"),
    "edge_index_gap":         ("A", ERROR,   "idx/idx2 sind nicht lückenlos ab 0"),
    "edge_key_collision":     ("A", ERROR,   "Zwei Kanten mit gleichem Schlüssel (Quelle, Attribut, idx, idx2)"),
    "label_mismatch":         ("A", ERROR,   "Neo4j-Labels weichen von der Klassenkette des Schemas ab"),
    # B
    "required_attribute":     ("B", ERROR,   "Pflichtattribut fehlt"),
    "prop_on_reference_attribute": ("B", ERROR, "Property auf einem Attribut, das eine Referenz sein muss"),
    "unknown_property":       ("B", WARNING, "Property gehört zu keinem Attribut der Klasse"),
    "guid_missing_or_malformed": ("B", ERROR, "GlobalId fehlt oder hat nicht 22 gültige Zeichen"),
    "duplicate_guid":         ("B", ERROR,   "GlobalId kommt mehrfach vor"),
    "enum_value":             ("B", ERROR,   "Wert ist kein Element der Enumeration"),
    "value_type":             ("B", ERROR,   "Wert passt nicht zum Typ des Attributs"),
    "value_type_missing":     ("B", ERROR,   "Typ des Auswahlwerts (<Attribut>_type) fehlt oder ist unzulässig"),
    "list_bounds":            ("B", ERROR,   "Listenlänge außerhalb der Schemagrenzen"),
    "placement_cycle":        ("B", ERROR,   "PlacementRelTo-Kette enthält einen Zyklus"),
    # C
    "opening_no_host":        ("C", ERROR,   "Öffnung ohne IfcRelVoidsElement"),
    "opening_multiple_hosts": ("C", ERROR,   "Öffnung in mehreren Wänden (mehrere IfcRelVoidsElement)"),
    "opening_multiple_fillings": ("C", ERROR, "Öffnung mit mehr als einer Füllung"),
    "filling_multiple_openings": ("C", ERROR, "Füllung steckt in mehr als einer Öffnung"),
    "filling_without_opening": ("C", WARNING, "Tür/Fenster ohne Öffnung (IfcRelFillsElement fehlt)"),
    "multiple_containers":    ("C", ERROR,   "Bauteil in mehr als einem räumlichen Container"),
    "element_not_contained":  ("C", WARNING, "Bauteil ohne Container und ohne übergeordnetes Bauteil"),
    "multiple_types":         ("C", ERROR,   "Objekt hat mehr als ein Typobjekt (IfcRelDefinesByType)"),
    "element_without_placement": ("C", ERROR, "Bauteil ohne ObjectPlacement"),
    "orphan_pset":            ("C", WARNING, "Property-Set/Mengen ohne Zuordnung (kein IfcRelDefinesByProperties, kein Typ)"),
    "orphan_property":        ("C", WARNING, "Property/Menge ohne übergeordnetes Set"),
    "orphan_placement":       ("C", WARNING, "Platzierung, auf die nichts verweist"),
    # D
    "class_changed":          ("D", ERROR,   "Klasse eines Knotens hat sich geändert"),
    "unexpected_added":       ("D", ERROR,   "Neuer Knoten, den die Operation nicht anlegen darf"),
    "unexpected_removed":     ("D", ERROR,   "Knoten entfernt, den die Operation nicht entfernen darf"),
    "unexpected_prop_change": ("D", ERROR,   "Property geändert, die die Operation nicht ändern darf"),
    "unexpected_ref_change":  ("D", ERROR,   "Referenzattribut geändert, das die Operation nicht ändern darf"),
    "target_missing":         ("D", ERROR,   "Zielobjekt der Operation fehlt im Ausgangsmodell"),
    "expected_change_missing": ("D", ERROR,  "Erwartete Änderung der Operation fehlt im Graph"),
    "unknown_operation":      ("D", WARNING, "Für diese Operation gibt es keinen Fußabdruck"),
}

# Hinweise für den Generator (Rückmeldung bei Fehlern)
HINTS: dict[str, str] = {
    "opening_no_host": "Eine Öffnung braucht genau eine Wand (IfcRelVoidsElement).",
    "filling_multiple_openings": "Ein Fenster/eine Tür darf nur in einer Öffnung sitzen.",
    "multiple_containers": "Ein Bauteil gehört zu genau einem Geschoss.",
    "required_attribute": "Pflichtattribute dürfen nicht entfallen.",
    "unexpected_added": "Die Operation hat Knoten angelegt, die zu ihr nicht passen; Parameter prüfen.",
    "unexpected_removed": "Die Operation hat zu viel entfernt; Ziel und Kaskade prüfen.",
}


@dataclass
class Finding:
    group: str
    code: str
    severity: str
    step_id: int | None
    guid: str | None
    ifc_class: str | None
    message: str
    subject: str = ""            # Gegenstand (z. B. Attributname), Teil des Schlüssels
    detail: dict = field(default_factory=dict)
    existing: bool = False       # schon im Ausgangsmodell vorhanden -> blockiert nicht

    @property
    def key(self) -> tuple:
        return (self.code, self.step_id, self.subject)

    def to_json(self) -> dict:
        return {"group": self.group, "code": self.code, "severity": self.severity, "step_id": self.step_id,
                "guid": self.guid, "ifc_class": self.ifc_class, "message": self.message,
                "subject": self.subject, "detail": self.detail, "existing": self.existing}


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    nodes: int = 0
    edges: int = 0
    seconds: float = 0.0
    diff_summary: str | None = None

    @property
    def new_errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR and not f.existing]

    @property
    def new_warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING and not f.existing]

    @property
    def existing(self) -> list[Finding]:
        return [f for f in self.findings if f.existing]

    @property
    def ok(self) -> bool:
        return not self.new_errors

    def summary(self) -> str:
        e = Counter((f.severity) for f in self.findings if not f.existing)
        ex = Counter((f.severity) for f in self.findings if f.existing)
        return (f"{'OK' if self.ok else 'FEHLER'}: neu {e[ERROR]} Fehler, {e[WARNING]} Warnungen | "
                f"schon im Ausgangsmodell {ex[ERROR]} Fehler, {ex[WARNING]} Warnungen | "
                f"{self.nodes} Knoten, {self.edges} Kanten, {self.seconds:.2f} s")

    def feedback(self, limit: int = 8) -> str:
        """Text für den Generator (Reparaturschleife 'fail: Fehlermeldung')."""
        if self.ok:
            return ""
        errs = self.new_errors
        lines = [f"Strukturprüfung fehlgeschlagen: {len(errs)} neue Fehler."]
        for f in errs[:limit]:
            who = f"{f.ifc_class} {f.guid or f.step_id}" if f.ifc_class else f"Knoten {f.step_id}"
            hint = HINTS.get(f.code)
            lines.append(f"- [{f.group}/{f.code}] {who}: {f.message}" + (f" ({hint})" if hint else ""))
        if len(errs) > limit:
            lines.append(f"- ... und {len(errs) - limit} weitere")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {"ok": self.ok, "summary": self.summary(), "diff": self.diff_summary,
                "new_errors": [f.to_json() for f in self.new_errors],
                "new_warnings": [f.to_json() for f in self.new_warnings],
                "existing": [f.to_json() for f in self.existing]}


# =========================================================================== Schema

@dataclass
class AttrInfo:
    name: str
    optional: bool
    derived: bool
    depth: int                       # 0 Einzelwert, 1 Liste, 2 Liste von Listen
    bounds: tuple                    # je Ebene (min, max); max -1 = unbegrenzt
    kind: str                        # entity | select | enum | simple | other
    entities: frozenset = frozenset()   # entity/select: erlaubte Klassen (inkl. Unterklassen über die Kette)
    typedefs: dict = field(default_factory=dict)  # select: Typname -> Python-Typen
    items: frozenset = frozenset()   # enum
    pytypes: tuple = ()              # simple: erlaubte Python-Typen


_SIMPLE = {
    "string": (str,), "binary": (str,), "boolean": (bool,), "logical": (bool, str),
    "integer": (int,), "real": (int, float), "number": (int, float),
}


def _simple_pytypes(t) -> tuple:
    return _SIMPLE.get(str(t).strip("<>").strip(), ())


class SchemaInfo:
    def __init__(self, schema):
        self.schema = schema
        self._chain: dict[str, frozenset] = {}
        self._attrs: dict[str, dict[str, AttrInfo]] = {}

    def known(self, cls: str) -> bool:
        try:
            self.schema.declaration_by_name(cls)
            return True
        except Exception:
            return False

    def chain(self, cls: str) -> frozenset:
        c = self._chain.get(cls)
        if c is None:
            c = frozenset(g.label_chain(self.schema, cls, {}))
            self._chain[cls] = c
        return c

    def attrs(self, cls: str) -> dict[str, AttrInfo]:
        a = self._attrs.get(cls)
        if a is None:
            decl = self.schema.declaration_by_name(cls)
            a = {}
            for at, der in zip(decl.all_attributes(), decl.derived()):
                a[at.name()] = self._describe(at, bool(der))
            self._attrs[cls] = a
        return a

    # -- Typauflösung ------------------------------------------------------------------------
    def _expand_select(self, sel, ents: set, tds: dict, seen: set) -> None:
        if sel.name() in seen:
            return
        seen.add(sel.name())
        for m in sel.select_list():
            if isinstance(m, w.select_type):
                self._expand_select(m, ents, tds, seen)
            elif isinstance(m, w.type_declaration):
                base = m.declared_type()
                while isinstance(base, w.named_type):
                    d = base.declared_type()
                    base = d.declared_type() if isinstance(d, w.type_declaration) else d
                tds[m.name()] = _simple_pytypes(base) if isinstance(base, w.simple_type) else ()
            elif isinstance(m, w.enumeration_type):
                continue
            else:
                ents.add(m.name())

    def _describe(self, at, derived: bool) -> AttrInfo:
        t = at.type_of_attribute()
        bounds = []
        while isinstance(t, w.aggregation_type):
            b2 = t.bound2()
            bounds.append((t.bound1(), -1 if b2 is None else b2))
            t = t.type_of_element()
        info = AttrInfo(at.name(), bool(at.optional()), derived, len(bounds), tuple(bounds), "other")
        base = t
        while isinstance(base, w.named_type):
            base = base.declared_type()
        # base ist jetzt Deklaration (entity/select/enum/typedef) oder einfacher Typ
        while isinstance(base, w.type_declaration):
            inner = base.declared_type()
            if isinstance(inner, w.aggregation_type):
                return info                      # z. B. IfcCompoundPlaneAngleMeasure: nicht geprüft
            while isinstance(inner, w.named_type):
                inner = inner.declared_type()
            base = inner
        if isinstance(base, w.entity):
            info.kind, info.entities = "entity", frozenset([base.name()])
        elif isinstance(base, w.select_type):
            ents: set = set()
            tds: dict = {}
            self._expand_select(base, ents, tds, set())
            info.kind, info.entities, info.typedefs = "select", frozenset(ents), tds
        elif isinstance(base, w.enumeration_type):
            info.kind, info.items = "enum", frozenset(base.enumeration_items())
        elif isinstance(base, w.simple_type):
            info.kind, info.pytypes = "simple", _simple_pytypes(base)
        return info


def _type_ok(value: Any, pytypes: tuple) -> bool:
    if not pytypes:
        return True
    if isinstance(value, bool):
        return bool in pytypes
    return isinstance(value, pytypes)


# =========================================================================== Index

class Index:
    """Hilfsstrukturen über einen Snapshot: geordnete Ausgangskanten, Eingangskanten, Klassenketten."""

    def __init__(self, snap: r.Snapshot, si: SchemaInfo):
        self.snap, self.si = snap, si
        self.nodes = snap.nodes
        self.out: dict[tuple, list[int]] = defaultdict(list)       # (src, attr) -> Ziele in idx-Reihenfolge
        self.inn: dict[tuple, list[int]] = defaultdict(list)       # (dst, attr) -> Quellen
        self.in_any: dict[int, int] = defaultdict(int)             # dst -> Zahl eingehender Kanten
        raw: dict[tuple, list] = defaultdict(list)
        for (src, name, i1, i2), dst in snap.edges.items():
            raw[(src, name)].append((-1 if i1 is None else i1, -1 if i2 is None else i2, dst))
            self.inn[(dst, name)].append(src)
            self.in_any[dst] += 1
        for key, items in raw.items():
            items.sort()
            self.out[key] = [d for _, _, d in items]
        self.by_class: dict[str, list[int]] = defaultdict(list)
        for sid, p in self.nodes.items():
            self.by_class[p.get("ifc_class", "?")].append(sid)

    def cls(self, sid: int) -> str:
        return self.nodes[sid].get("ifc_class", "?")

    def isa(self, sid: int, name: str) -> bool:
        p = self.nodes.get(sid)
        return p is not None and name in self.si.chain(p.get("ifc_class", "?"))

    def of_type(self, name: str) -> list[int]:
        """Alle Knoten, deren Klasse `name` ist oder davon erbt."""
        out = []
        for c, ids in self.by_class.items():
            if name in self.si.chain(c):
                out.extend(ids)
        return sorted(out)

    def first(self, sid: int, attr: str) -> int | None:
        t = self.out.get((sid, attr))
        return t[0] if t else None

    def guid(self, sid: int) -> str | None:
        p = self.nodes.get(sid)
        return p.get("GlobalId") if p else None


# =========================================================================== Regeln

class Checker:
    """Führt die Gruppen A, B, C auf einem Snapshot aus."""

    def __init__(self, si: SchemaInfo, snap: r.Snapshot, labels: dict | None = None):
        self.si, self.snap, self.labels = si, snap, labels
        self.ix = Index(snap, si)
        self.findings: list[Finding] = []

    def add(self, code: str, sid: int | None, message: str, subject: str = "", **detail) -> None:
        group, sev, _ = RULES[code]
        p = self.ix.nodes.get(sid) if sid is not None else None
        self.findings.append(Finding(group, code, sev, sid, p.get("GlobalId") if p else None,
                                     p.get("ifc_class") if p else None, message, subject, detail))

    def run(self) -> list[Finding]:
        self.group_a()
        self.group_b()
        self.group_c()
        return self.findings

    # ---- A ------------------------------------------------------------------------------------
    def group_a(self) -> None:
        ix, si = self.ix, self.si
        if self.snap.collisions:
            self.add("edge_key_collision", None, f"{self.snap.collisions} Kanten mit gleichem Schlüssel")
        for sid, p in ix.nodes.items():
            cls = p.get("ifc_class")
            if not cls or not si.known(cls):
                self.add("unknown_class", sid, f"Klasse {cls!r} ist im Schema nicht definiert")
            elif self.labels is not None and sid in self.labels:
                want = set(si.chain(cls)) | {"IfcEntity"}
                if set(self.labels[sid]) != want:
                    self.add("label_mismatch", sid, f"Labels {sorted(self.labels[sid])} statt {sorted(want)}")
        # Kanten je (Quelle, Attribut) gruppiert
        groups: dict[tuple, list] = defaultdict(list)
        for (src, name, i1, i2), dst in self.snap.edges.items():
            if src not in ix.nodes:
                self.add("dangling_edge", None, f"Kante {name} hat die nicht vorhandene Quelle {src}",
                         subject=f"{src}.{name}")
                continue
            if dst not in ix.nodes:
                self.add("dangling_edge", src, f"Kante {name} zeigt auf den nicht vorhandenen Knoten {dst}",
                         subject=f"{name}->{dst}")
                continue
            groups[(src, name)].append((i1, i2, dst))
        for (src, name), items in groups.items():
            cls = ix.cls(src)
            if not si.known(cls):
                continue
            info = si.attrs(cls).get(name)
            if info is None or info.derived:
                self.add("unknown_edge_attribute", src, f"{name} ist kein Attribut von {cls}", subject=name)
                continue
            if info.kind not in ("entity", "select") or (info.kind == "select" and not info.entities):
                self.add("edge_on_value_attribute", src, f"{cls}.{name} darf keine Referenz tragen", subject=name)
                continue
            for i1, i2, dst in items:
                dc = ix.cls(dst)
                if not (info.entities & si.chain(dc)):
                    self.add("edge_target_class", src,
                             f"{cls}.{name} verweist auf {dc} ({dst}); erlaubt: {sorted(info.entities)[:4]}",
                             subject=f"{name}->{dst}")
            self._check_indices(src, cls, name, info, items)

    def _check_indices(self, src, cls, name, info: AttrInfo, items) -> None:
        depth = info.depth
        for i1, i2, _ in items:
            shape = (0 if i1 is None else 1) + (0 if i2 is None else 1)
            if shape != depth:
                self.add("edge_index_shape", src,
                         f"{cls}.{name}: Listentiefe {depth} laut Schema, Kante hat idx={i1}, idx2={i2}",
                         subject=name)
                return
        if depth == 0:
            if len(items) > 1:
                self.add("edge_key_collision", src, f"{cls}.{name} ist ein Einzelwert, hat aber {len(items)} Kanten",
                         subject=name)
        elif depth == 1:
            idx = sorted(i1 for i1, _, _ in items)
            if idx != list(range(len(idx))):
                self.add("edge_index_gap", src, f"{cls}.{name}: idx {idx[:8]} nicht lückenlos ab 0", subject=name)
            self._bounds(src, cls, name, info, 0, len(idx))
        else:
            rows: dict[int, list] = defaultdict(list)
            for i1, i2, _ in items:
                rows[i1].append(i2)
            if sorted(rows) != list(range(len(rows))):
                self.add("edge_index_gap", src, f"{cls}.{name}: Zeilenindex nicht lückenlos", subject=name)
            for k, cols in rows.items():
                if sorted(cols) != list(range(len(cols))):
                    self.add("edge_index_gap", src, f"{cls}.{name}[{k}]: idx2 nicht lückenlos", subject=name)
            self._bounds(src, cls, name, info, 0, len(rows))
            for cols in rows.values():
                self._bounds(src, cls, name, info, 1, len(cols))

    def _bounds(self, src, cls, name, info: AttrInfo, level: int, n: int) -> None:
        lo, hi = info.bounds[level]
        if n < lo or (hi != -1 and n > hi):
            self.add("list_bounds", src, f"{cls}.{name}: {n} Elemente, erlaubt {lo}..{'*' if hi == -1 else hi}",
                     subject=name)

    # ---- B ------------------------------------------------------------------------------------
    def group_b(self) -> None:
        ix, si = self.ix, self.si
        guids: dict[str, list[int]] = defaultdict(list)
        for sid, p in ix.nodes.items():
            cls = p.get("ifc_class")
            if not cls or not si.known(cls):
                continue
            attrs = si.attrs(cls)
            is_root = "IfcRoot" in si.chain(cls)
            if is_root:
                gid = p.get("GlobalId")
                if not isinstance(gid, str) or not GUID_RE.match(gid):
                    self.add("guid_missing_or_malformed", sid, f"GlobalId {gid!r} ist keine gültige 22-stellige Kennung")
                else:
                    guids[gid].append(sid)
            for name, info in attrs.items():
                if info.derived:
                    continue
                has_prop = name in p
                has_edge = (sid, name) in ix.out
                may_be_empty = info.depth >= 1 and info.bounds[0][0] == 0     # leere Liste wird nicht abgebildet
                if not info.optional and name not in CUT and not may_be_empty and not has_prop and not has_edge:
                    self.add("required_attribute", sid, f"Pflichtattribut {cls}.{name} fehlt", subject=name)
                if has_prop:
                    self._check_value(sid, cls, p, name, info)
            for key in p:
                if key in META_KEYS or key in attrs:
                    continue
                if key.endswith("_type") and key[:-5] in attrs:
                    continue
                self.add("unknown_property", sid, f"{cls} hat kein Attribut {key}", subject=key)
        for gid, ids in guids.items():
            if len(ids) > 1:
                for sid in ids:
                    self.add("duplicate_guid", sid, f"GlobalId {gid} kommt {len(ids)}-mal vor", subject=gid)
        self._placement_cycles()

    def _check_value(self, sid, cls, p, name, info: AttrInfo) -> None:
        v = p[name]
        if info.kind == "other":
            return
        if info.kind in ("entity",) or (info.kind == "select" and not info.typedefs):
            self.add("prop_on_reference_attribute", sid, f"{cls}.{name} muss eine Referenz sein, hat aber den Wert {v!r}",
                     subject=name)
            return
        if info.depth == 0:
            if isinstance(v, list):
                self.add("value_type", sid, f"{cls}.{name} ist ein Einzelwert, hat aber eine Liste", subject=name)
                return
            vals = [v]
        elif info.depth == 1:
            if not isinstance(v, list):
                self.add("value_type", sid, f"{cls}.{name} ist eine Liste, hat aber {v!r}", subject=name)
                return
            vals = v
            self._bounds(sid, cls, name, info, 0, len(v))
        else:
            vals = []                              # verschachtelt: als JSON abgelegt, nicht geprüft
        if info.kind == "enum":
            for x in vals:
                if x not in info.items:
                    self.add("enum_value", sid, f"{cls}.{name}: {x!r} ist kein Wert von {sorted(info.items)[:5]}...",
                             subject=name)
        elif info.kind == "simple":
            for x in vals:
                if not _type_ok(x, info.pytypes):
                    self.add("value_type", sid, f"{cls}.{name}: {x!r} ({type(x).__name__}) passt nicht zum Schematyp",
                             subject=name)
        elif info.kind == "select":
            tname = p.get(name + "_type")
            if info.depth == 0:
                if tname not in info.typedefs:
                    self.add("value_type_missing", sid, f"{cls}.{name}: Typangabe {tname!r} fehlt oder ist nicht zulässig",
                             subject=name)
                elif not _type_ok(v, info.typedefs[tname]):
                    self.add("value_type", sid, f"{cls}.{name}: {v!r} passt nicht zu {tname}", subject=name)

    def _placement_cycles(self) -> None:
        ix = self.ix
        for pid in ix.by_class.get("IfcLocalPlacement", []):
            seen, cur = {pid}, ix.first(pid, "PlacementRelTo")
            while cur is not None and cur in ix.nodes:
                if cur in seen:
                    self.add("placement_cycle", pid, "PlacementRelTo-Kette ist ein Zyklus")
                    break
                seen.add(cur)
                cur = ix.first(cur, "PlacementRelTo")

    # ---- C ------------------------------------------------------------------------------------
    def group_c(self) -> None:
        ix = self.ix

        def rels(dst: int, attr: str, rel_class: str) -> list[int]:
            return [s for s in ix.inn.get((dst, attr), []) if ix.isa(s, rel_class)]

        for o in ix.of_type("IfcOpeningElement"):
            hosts = rels(o, "RelatedOpeningElement", "IfcRelVoidsElement")
            if not hosts:
                self.add("opening_no_host", o, "Öffnung ist in keiner Wand (kein IfcRelVoidsElement)")
            elif len(hosts) > 1:
                self.add("opening_multiple_hosts", o, f"Öffnung gehört zu {len(hosts)} IfcRelVoidsElement")
            fills = rels(o, "RelatingOpeningElement", "IfcRelFillsElement")
            if len(fills) > 1:
                self.add("opening_multiple_fillings", o, f"Öffnung hat {len(fills)} Füllungen")
        for e in ix.of_type("IfcElement"):
            p = ix.nodes[e]
            f = rels(e, "RelatedBuildingElement", "IfcRelFillsElement")
            if len(f) > 1:
                self.add("filling_multiple_openings", e, f"Bauteil steckt in {len(f)} Öffnungen")
            if (ix.isa(e, "IfcDoor") or ix.isa(e, "IfcWindow")) and not f:
                self.add("filling_without_opening", e, "Tür/Fenster steckt in keiner Öffnung")
            if (e, "ObjectPlacement") not in ix.out:
                self.add("element_without_placement", e, "Bauteil hat keine ObjectPlacement")
            if ix.isa(e, "IfcFeatureElement"):
                continue                                  # Öffnungen liegen über ihre Wand, nicht im Geschoss
            cont = rels(e, "RelatedElements", "IfcRelContainedInSpatialStructure")
            if len(cont) > 1:
                self.add("multiple_containers", e, f"Bauteil ist in {len(cont)} Containern")
            elif not cont:
                parent = rels(e, "RelatedObjects", "IfcRelAggregates") + rels(e, "RelatedObjects", "IfcRelNests")
                if not parent:
                    self.add("element_not_contained", e, "Bauteil hat weder Container noch übergeordnetes Bauteil")
        for o in ix.of_type("IfcObject"):
            t = rels(o, "RelatedObjects", "IfcRelDefinesByType")
            if len(t) > 1:
                self.add("multiple_types", o, f"Objekt hat {len(t)} Typobjekte")
        for ps in ix.of_type("IfcPropertySetDefinition"):
            used = (rels(ps, "RelatingPropertyDefinition", "IfcRelDefinesByProperties")
                    or ix.inn.get((ps, "HasPropertySets")))
            if not used:
                self.add("orphan_pset", ps, "Property-Set/Mengen ohne Zuordnung")
        for pr in ix.of_type("IfcProperty") + ix.of_type("IfcPhysicalQuantity"):
            if not ix.in_any.get(pr):
                self.add("orphan_property", pr, "Property/Menge ohne übergeordnetes Set")
        for pl in ix.by_class.get("IfcLocalPlacement", []):
            if not ix.in_any.get(pl):
                self.add("orphan_placement", pl, "Platzierung ohne Verweis")


# =========================================================================== Cypher-Entsprechungen

# code -> Abfrage (Parameter $f = source_file), Rückgabe: step_id der betroffenen Knoten.
CYPHER: dict[str, str] = {
    "opening_no_host":
        "MATCH (o:IfcOpeningElement {source_file:$f}) "
        "WHERE NOT EXISTS { MATCH (:IfcRelVoidsElement)-[:RelatedOpeningElement]->(o) } RETURN o.step_id AS id",
    "opening_multiple_hosts":
        "MATCH (o:IfcOpeningElement {source_file:$f}) "
        "WHERE size([(x:IfcRelVoidsElement)-[:RelatedOpeningElement]->(o) | x]) > 1 RETURN o.step_id AS id",
    "opening_multiple_fillings":
        "MATCH (o:IfcOpeningElement {source_file:$f}) "
        "WHERE size([(x:IfcRelFillsElement)-[:RelatingOpeningElement]->(o) | x]) > 1 RETURN o.step_id AS id",
    "filling_multiple_openings":
        "MATCH (e:IfcElement {source_file:$f}) "
        "WHERE size([(x:IfcRelFillsElement)-[:RelatedBuildingElement]->(e) | x]) > 1 RETURN e.step_id AS id",
    "filling_without_opening":
        "MATCH (e:IfcElement {source_file:$f}) WHERE (e:IfcDoor OR e:IfcWindow) "
        "AND NOT EXISTS { MATCH (:IfcRelFillsElement)-[:RelatedBuildingElement]->(e) } RETURN e.step_id AS id",
    "element_without_placement":
        "MATCH (e:IfcElement {source_file:$f}) WHERE NOT (e)-[:ObjectPlacement]->() RETURN e.step_id AS id",
    "multiple_containers":
        "MATCH (e:IfcElement {source_file:$f}) WHERE NOT e:IfcFeatureElement "
        "AND size([(x:IfcRelContainedInSpatialStructure)-[:RelatedElements]->(e) | x]) > 1 RETURN e.step_id AS id",
    "element_not_contained":
        "MATCH (e:IfcElement {source_file:$f}) WHERE NOT e:IfcFeatureElement "
        "AND NOT EXISTS { MATCH (:IfcRelContainedInSpatialStructure)-[:RelatedElements]->(e) } "
        "AND NOT EXISTS { MATCH (:IfcRelAggregates)-[:RelatedObjects]->(e) } "
        "AND NOT EXISTS { MATCH (:IfcRelNests)-[:RelatedObjects]->(e) } RETURN e.step_id AS id",
    "multiple_types":
        "MATCH (o:IfcObject {source_file:$f}) "
        "WHERE size([(x:IfcRelDefinesByType)-[:RelatedObjects]->(o) | x]) > 1 RETURN o.step_id AS id",
    "orphan_pset":
        "MATCH (p:IfcPropertySetDefinition {source_file:$f}) "
        "WHERE NOT EXISTS { MATCH (:IfcRelDefinesByProperties)-[:RelatingPropertyDefinition]->(p) } "
        "AND NOT EXISTS { MATCH ()-[:HasPropertySets]->(p) } RETURN p.step_id AS id",
    "orphan_property":
        "MATCH (p {source_file:$f}) WHERE (p:IfcProperty OR p:IfcPhysicalQuantity) "
        "AND NOT EXISTS { MATCH ()-[x]->(p) WHERE x.inverse IS NULL } RETURN p.step_id AS id",
    "orphan_placement":
        "MATCH (p:IfcLocalPlacement {source_file:$f}) "
        "WHERE NOT EXISTS { MATCH ()-[x]->(p) WHERE x.inverse IS NULL } RETURN p.step_id AS id",
    "duplicate_guid":
        "MATCH (n:IfcRoot {source_file:$f}) WITH n.GlobalId AS gid, collect(n.step_id) AS ids "
        "WHERE size(ids) > 1 UNWIND ids AS id RETURN id",
}


# =========================================================================== Gruppe D

ANY = "*"


def _rel_list(chain: frozenset, attr: str) -> bool:
    return "IfcRelationship" in chain and attr.startswith("Related")


# Fußabdruck je Operation:
#   added/removed: erlaubte Klassen (ANY = beliebig) für neue/entfernte Knoten
#   props: Klasse -> erlaubte geänderte Properties (an bestehenden Knoten)
#   refs: Funktion (Klassenkette, Attribut) -> bool für geänderte Referenzattribute bestehender Knoten
GEOM_NODES = {"IfcCartesianPoint", "IfcDirection", "IfcAxis2Placement3D", "IfcLocalPlacement"}
FOOTPRINT: dict[str, dict] = {
    "delete_element": dict(added=set(), removed=ANY, props={}, refs=_rel_list),
    "move_element": dict(added={"IfcCartesianPoint"}, removed=set(), props={"IfcCartesianPoint": {"Coordinates"}},
                         refs=lambda ch, a: "IfcAxis2Placement3D" in ch and a == "Location"),
    "reassign_container": dict(
        added={"IfcRelContainedInSpatialStructure", "IfcCartesianPoint"}, removed={"IfcRelContainedInSpatialStructure"},
        props={"IfcCartesianPoint": {"Coordinates"}},
        refs=lambda ch, a: (("IfcRelContainedInSpatialStructure" in ch and a in ("RelatedElements", "RelatingStructure"))
                            or ("IfcLocalPlacement" in ch and a == "PlacementRelTo")
                            or ("IfcAxis2Placement3D" in ch and a == "Location"))),
    "clone_element": dict(added=ANY, removed=set(), props={}, refs=_rel_list),
    "set_property": dict(added=set(), removed=set(), props={"IfcPropertySingleValue": {"NominalValue", "NominalValue_type"}},
                         refs=lambda ch, a: False),
    "set_attribute": dict(added=set(), removed=set(), props=ANY, refs=lambda ch, a: False),
    "assign_type": dict(added={"IfcRelDefinesByType"}, removed={"IfcRelDefinesByType"}, props={},
                        refs=lambda ch, a: "IfcRelDefinesByType" in ch and a == "RelatedObjects"),
    "add_element": dict(added=GEOM_NODES | {"IfcWall", "IfcColumn", "IfcSlab"}, removed=set(), props={},
                        refs=lambda ch, a: ("IfcRelationship" in ch and a in ("RelatedElements", "RelatedObjects"))),
    "add_hosted_element": dict(
        added=GEOM_NODES | {"IfcWindow", "IfcDoor", "IfcOpeningElement", "IfcRelVoidsElement", "IfcRelFillsElement"},
        removed=set(), props={},
        refs=lambda ch, a: ("IfcRelationship" in ch and a in ("RelatedElements", "RelatedObjects"))),
}
# Zusatzklassen, die add_element/add_hosted_element wahlweise anlegen dürfen
ALLOWED_EXTRA_ADDED = {"add_element": set(), "add_hosted_element": set()}


class DiffChecker:
    def __init__(self, si: SchemaInfo, base: r.Snapshot, target: r.Snapshot, op: dict):
        self.si, self.base, self.target, self.op = si, base, target, op
        self.bx, self.tx = Index(base, si), Index(target, si)
        self.findings: list[Finding] = []

    def add(self, code: str, sid: int | None, message: str, subject: str = "", **detail) -> None:
        group, sev, _ = RULES[code]
        p = (self.tx.nodes.get(sid) or self.bx.nodes.get(sid)) if sid is not None else None
        self.findings.append(Finding(group, code, sev, sid, p.get("GlobalId") if p else None,
                                     p.get("ifc_class") if p else None, message, subject, detail))

    def guid_node(self, ix: Index, guid: str | None) -> int | None:
        if guid is None:
            return None
        for sid, p in ix.nodes.items():
            if p.get("GlobalId") == guid:
                return sid
        return None

    def run(self) -> tuple[list[Finding], r.Diff | None]:
        name = self.op.get("op")
        fp = FOOTPRINT.get(name)
        if fp is None:
            self.add("unknown_operation", None, f"Operation {name!r} hat keinen Fußabdruck")
            return self.findings, None
        try:
            d = r.diff_snapshots(self.base, self.target)
        except ValueError as e:
            self.add("class_changed", None, str(e))
            return self.findings, None
        self._footprint(fp, d)
        getattr(self, "_op_" + name)(d)
        return self.findings, d

    # -- allgemeiner Fußabdruck ------------------------------------------------------------------
    def _footprint(self, fp: dict, d: r.Diff) -> None:
        si = self.si
        for sid in d.added_nodes:
            c = self.tx.cls(sid)
            if fp["added"] != ANY and c not in fp["added"]:
                self.add("unexpected_added", sid, f"{c} darf von {self.op['op']} nicht angelegt werden")
        for sid in d.removed_nodes:
            c = self.bx.cls(sid)
            if fp["removed"] != ANY and c not in fp["removed"]:
                self.add("unexpected_removed", sid, f"{c} darf von {self.op['op']} nicht entfernt werden")
            if self.op["op"] == "delete_element" and si.chain(c) & {"IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey"}:
                self.add("unexpected_removed", sid, f"{c} (Raumstruktur) darf nie entfernt werden", subject="spatial")
        for sid, ch in d.changed_nodes.items():
            c = self.bx.cls(sid)
            allowed = fp["props"] if fp["props"] == ANY else fp["props"].get(c, set())
            keys = set(ch["set"]) | set(ch["unset"])
            bad = set() if allowed == ANY else keys - set(allowed)
            for k in sorted(bad):
                self.add("unexpected_prop_change", sid, f"{c}.{k} darf von {self.op['op']} nicht geändert werden",
                         subject=k)
        for (src, attr), _ in d.changed_attrs.items():
            if src in self.bx.nodes and src in self.tx.nodes:       # neue Knoten tragen ihre Kanten selbst
                chain = si.chain(self.bx.cls(src))
                if not fp["refs"](chain, attr):
                    self.add("unexpected_ref_change", src, f"{self.bx.cls(src)}.{attr} darf von {self.op['op']} nicht geändert werden",
                             subject=attr)

    # -- Operationen -----------------------------------------------------------------------------
    def _target(self, d: r.Diff, key: str = "guid") -> int | None:
        guid = self.op.get(key)
        sid = self.guid_node(self.bx, guid)
        if sid is None:
            self.add("target_missing", None, f"{key}={guid!r} existiert im Ausgangsmodell nicht")
        return sid

    def _op_delete_element(self, d: r.Diff) -> None:
        sid = self._target(d)
        if sid is not None and sid not in d.removed_nodes:
            self.add("expected_change_missing", sid, "Das zu löschende Bauteil ist noch im Graph")

    def _op_move_element(self, d: r.Diff) -> None:
        sid = self._target(d)
        if sid is None:
            return
        if sid in d.removed_nodes:
            self.add("unexpected_removed", sid, "Das verschobene Bauteil darf nicht entfernt werden")
        if not any(self.bx.cls(s) == "IfcCartesianPoint" for s in d.changed_nodes):
            if not any(self.tx.cls(s) == "IfcCartesianPoint" for s in d.added_nodes):
                self.add("expected_change_missing", sid, "Keine Koordinate hat sich geändert")

    def _op_reassign_container(self, d: r.Diff) -> None:
        sid = self._target(d)
        stor = self.guid_node(self.tx, self.op.get("target_storey_guid"))
        if sid is None:
            return
        if stor is None:
            self.add("target_missing", None, "Zielgeschoss fehlt")
            return
        ok = any(self.tx.isa(rel, "IfcRelContainedInSpatialStructure") and self.tx.first(rel, "RelatingStructure") == stor
                 for rel in self.tx.inn.get((sid, "RelatedElements"), []))
        if not ok:
            self.add("expected_change_missing", sid, "Das Bauteil ist nicht im Zielgeschoss enthalten")

    def _op_clone_element(self, d: r.Diff) -> None:
        sid = self._target(d, "source_guid")
        if sid is None:
            return
        roots = [s for s in d.added_nodes if self.tx.isa(s, "IfcElement") and not self.tx.isa(s, "IfcFeatureElement")]
        if len(roots) != 1:
            self.add("expected_change_missing", sid, f"Erwartet genau ein neues Bauteil, gefunden {len(roots)}")
        elif self.tx.cls(roots[0]) != self.bx.cls(sid):
            self.add("unexpected_added", roots[0], f"Kopie hat Klasse {self.tx.cls(roots[0])} statt {self.bx.cls(sid)}")
        if sid in d.changed_nodes or sid in d.removed_nodes:
            self.add("unexpected_prop_change", sid, "Die Vorlage darf nicht verändert werden", subject="source")

    def _op_set_property(self, d: r.Diff) -> None:
        sid = self._target(d)
        if sid is None:
            return
        if len(d.changed_nodes) != 1:
            self.add("expected_change_missing", sid, f"Erwartet genau eine geänderte Property, gefunden {len(d.changed_nodes)}")
            return
        (pid,) = d.changed_nodes
        ok = False
        for rel in self.tx.inn.get((sid, "RelatedObjects"), []):
            if not self.tx.isa(rel, "IfcRelDefinesByProperties"):
                continue
            ps = self.tx.first(rel, "RelatingPropertyDefinition")
            if ps is not None and self.tx.nodes[ps].get("Name") == self.op.get("pset_name") \
                    and pid in self.tx.out.get((ps, "HasProperties"), []) \
                    and self.tx.nodes[pid].get("Name") == self.op.get("property_name"):
                ok = True
        if not ok:
            self.add("unexpected_prop_change", pid, "Die geänderte Property gehört nicht zu pset_name/property_name des Bauteils",
                     subject="target")

    def _op_set_attribute(self, d: r.Diff) -> None:
        sid = self._target(d)
        if sid is None:
            return
        if set(d.changed_nodes) != {sid}:
            self.add("expected_change_missing", sid, f"Erwartet genau das Bauteil als geänderten Knoten, gefunden {sorted(d.changed_nodes)[:5]}")
            return
        ch = d.changed_nodes[sid]
        keys = (set(ch["set"]) | set(ch["unset"])) - {self.op.get("attribute") + "_type"}
        if keys != {self.op.get("attribute")}:
            self.add("unexpected_prop_change", sid, f"Geändert wurde {sorted(keys)} statt {self.op.get('attribute')!r}",
                     subject="attribute")

    def _op_assign_type(self, d: r.Diff) -> None:
        sid = self._target(d)
        typ = self.guid_node(self.tx, self.op.get("type_guid"))
        if sid is None:
            return
        if typ is None:
            self.add("target_missing", None, "Typobjekt fehlt")
            return
        ok = any(self.tx.isa(rel, "IfcRelDefinesByType") and self.tx.first(rel, "RelatingType") == typ
                 for rel in self.tx.inn.get((sid, "RelatedObjects"), []))
        if not ok:
            self.add("expected_change_missing", sid, "Das Bauteil hängt nicht am angegebenen Typobjekt")

    def _new_element(self, d: r.Diff) -> int | None:
        want = self.op.get("ifc_class")
        news = [s for s in d.added_nodes if self.tx.cls(s) == want]
        if len(news) != 1:
            self.add("expected_change_missing", None, f"Erwartet genau ein neues {want}, gefunden {len(news)}")
            return None
        return news[0]

    def _op_add_element(self, d: r.Diff) -> None:
        e = self._new_element(d)
        stor = self.guid_node(self.tx, self.op.get("storey_guid"))
        if e is None:
            return
        if stor is None:
            self.add("target_missing", None, "Zielgeschoss fehlt")
            return
        ok = any(self.tx.isa(rel, "IfcRelContainedInSpatialStructure") and self.tx.first(rel, "RelatingStructure") == stor
                 for rel in self.tx.inn.get((e, "RelatedElements"), []))
        if not ok:
            self.add("expected_change_missing", e, "Das neue Bauteil ist nicht im Zielgeschoss enthalten")

    def _op_add_hosted_element(self, d: r.Diff) -> None:
        e = self._new_element(d)
        host = self.guid_node(self.tx, self.op.get("host_guid"))
        if e is None:
            return
        if host is None:
            self.add("target_missing", None, "Wirtswand fehlt")
            return
        openings = [s for s in d.added_nodes if self.tx.cls(s) == "IfcOpeningElement"]
        if len(openings) != 1:
            self.add("expected_change_missing", e, f"Erwartet genau eine neue Öffnung, gefunden {len(openings)}")
            return
        o = openings[0]
        voids = [x for x in self.tx.inn.get((o, "RelatedOpeningElement"), []) if self.tx.isa(x, "IfcRelVoidsElement")]
        if not voids or self.tx.first(voids[0], "RelatingBuildingElement") != host:
            self.add("expected_change_missing", o, "Die Öffnung gehört nicht zur angegebenen Wand")
        fills = [x for x in self.tx.inn.get((e, "RelatedBuildingElement"), []) if self.tx.isa(x, "IfcRelFillsElement")]
        if not fills or self.tx.first(fills[0], "RelatingOpeningElement") != o:
            self.add("expected_change_missing", e, "Das Bauteil füllt die neue Öffnung nicht")


# =========================================================================== Validator

class StructuralValidator:
    def __init__(self, schema, baseline: r.Snapshot):
        self.si = SchemaInfo(schema)
        self.baseline = baseline
        self.base_findings = Checker(self.si, baseline).run()
        self.base_keys = {f.key for f in self.base_findings}

    def validate(self, snap: r.Snapshot, op: dict | None = None, labels: dict | None = None) -> Report:
        t0 = time.time()
        findings = Checker(self.si, snap, labels).run()
        diff_summary = None
        if op is not None:
            dfs, d = DiffChecker(self.si, self.baseline, snap, op).run()
            findings += dfs
            if d is not None:
                diff_summary = d.summary()
        for f in findings:
            f.existing = f.group != "D" and f.key in self.base_keys
        rep = Report(findings, len(snap.nodes), len(snap.edges), time.time() - t0, diff_summary)
        order = {ERROR: 0, WARNING: 1}
        rep.findings.sort(key=lambda f: (f.existing, order[f.severity], f.group, f.code, f.step_id or 0))
        return rep


def schema_for(model: ifcopenshell.file):
    name = getattr(model, "schema_identifier", None) or model.schema
    return w.schema_by_name(name)


# =========================================================================== Neo4j / CLI

def labels_from_neo4j(uri, user, password, database, source) -> dict[int, list[str]]:
    from neo4j import GraphDatabase
    out = {}
    with GraphDatabase.driver(uri, auth=(user, password), notifications_min_severity="OFF") as drv:
        with drv.session(database=database) as s:
            for rec in s.run("MATCH (n:IfcEntity {source_file:$f}) RETURN n.step_id AS id, labels(n) AS l", f=source):
                out[rec["id"]] = rec["l"]
    return out


def cypher_check(uri, user, password, database, source, report: Report) -> list[str]:
    """Führt die Cypher-Entsprechungen aus und vergleicht mit den Python-Befunden (alle, auch bestehende)."""
    from neo4j import GraphDatabase
    py: dict[str, set] = defaultdict(set)
    for f in report.findings:
        if f.step_id is not None:
            py[f.code].add(f.step_id)
    lines = []
    with GraphDatabase.driver(uri, auth=(user, password), notifications_min_severity="OFF") as drv:
        with drv.session(database=database) as s:
            for code, q in CYPHER.items():
                cy = {rec["id"] for rec in s.run(q, f=source)}
                same = cy == py.get(code, set())
                lines.append(f"  {'ok  ' if same else 'DIFF'} {code}: Cypher {len(cy)} | Python {len(py.get(code, set()))}")
    return lines


def print_rules() -> None:
    for code, (grp, sev, text) in RULES.items():
        print(f"{grp}  {sev:7s} {code:28s} {text}" + ("   [Cypher]" if code in CYPHER else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Structural Validator: Graph gegen IFC-Schema und Ausgangsmodell prüfen")
    ap.add_argument("ifc", nargs="?", help="Ausgangs-IFC (Baseline); der Dateiname ist der source_file im Graph")
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    ap.add_argument("--database", default=None)
    ap.add_argument("--op", help="Operation als JSON für die Differenzprüfung (Gruppe D)")
    ap.add_argument("--json", help="Bericht als JSON-Datei schreiben")
    ap.add_argument("--cypher-check", action="store_true", help="Cypher-Entsprechungen ausführen und vergleichen")
    ap.add_argument("--list-rules", action="store_true")
    ap.add_argument("--all", action="store_true", help="auch bereits vorhandene Befunde auflisten")
    args = ap.parse_args(argv)
    if args.list_rules:
        print_rules()
        return 0
    if not args.ifc:
        ap.error("IFC-Datei fehlt")

    model = ifcopenshell.open(args.ifc)
    source = os.path.basename(args.ifc)
    base = r.snapshot_from_ifc(model, source)
    v = StructuralValidator(schema_for(model), base)
    print(f"Ausgangsmodell: {len(base.nodes)} Knoten, {len(base.edges)} Kanten, "
          f"{len(v.base_findings)} Befunde (gelten als bestehend)")
    snap = r.snapshot_from_neo4j(args.uri, args.user, args.password, args.database, source)
    labels = labels_from_neo4j(args.uri, args.user, args.password, args.database, source)
    op = json.loads(args.op) if args.op else None
    rep = v.validate(snap, op, labels)
    print(rep.summary())
    if rep.diff_summary:
        print("Differenz zum Ausgangsmodell:", rep.diff_summary)
    for f in rep.new_errors + rep.new_warnings + (rep.existing if args.all else []):
        tag = "bestehend" if f.existing else f.severity
        print(f"  [{f.group}/{f.code}] {tag}: {f.ifc_class} {f.guid or f.step_id}: {f.message}")
    if not args.all and rep.existing:
        print(f"  ({len(rep.existing)} bestehende Befunde, mit --all anzeigen)")
    if args.cypher_check:
        print("Cypher-Vergleich:")
        print("\n".join(cypher_check(args.uri, args.user, args.password, args.database, source, rep)))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep.to_json(), fh, ensure_ascii=False, indent=2)
        print("Bericht:", args.json)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())

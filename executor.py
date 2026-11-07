#!/usr/bin/env python3
"""
executor.py — Executor (Graph): Operationen delete_element, move_element, reassign_container, clone_element, set_property, set_attribute, assign_type, add_element, add_hosted_element (Kap. 4.5, 4.6)

Der Executor setzt eine strukturierte Operation deterministisch auf dem Graphen um. Er prüft nicht,
ob das Ergebnis gültig ist (das tun der Structural Validator und der Semantic–Geometric Validator),
sondern setzt die Kaskadenregeln um und lehnt Operationen ab, die laut Katalog nicht erlaubt sind
(Scope-Grenzen, strict-Modus). Eine Ablehnung (ExecutorError) ist die Fehlermeldung, die an den
Generator zurückgeht. Alles läuft in einer Transaktion, bei einem Fehler bleibt der Graph unverändert.

Aufbau: Executor (Regeln) arbeitet auf einem Backend. Neo4jBackend setzt die Regeln in Cypher um,
MemoryBackend arbeitet auf einem Snapshot (für Tests ohne Datenbank).

KI-Hinweis: mit Unterstützung von Claude (Anthropic) erstellt, Dokumentation in Anhang F.

move_element: Translation in Weltkoordinaten (Meter). Der Executor rechnet sie in das lokale Koordinatensystem
des Bezugs-Placements (PlacementRelTo) um und ändert nur den Ort (Location) des Placements des Elements.
Was per Placement-Kette am Element hängt, folgt von selbst. Bei Füllungen wird die Öffnung mitgenommen.

reassign_container: hängt ein Bauteil (samt Öffnungen, Füllungen und Teilen) in ein anderes Geschoss. Die
Containment-Beziehung wechselt, das Placement wird auf das des Zielgeschosses umgehängt. Standard: die lokale Lage
bleibt, das Bauteil wandert in der Höhe mit. Mit preserve_world_position bleibt die Weltlage erhalten.

clone_element: legt eine Kopie des Bauteils mit Versatz an (Klasse, Attribute ohne Tag, Typ, Material, Container, Psets
als eigene Kopie, eigenes Placement, neue GlobalId). Der Graph enthält keine Geometrie. Der Klon trägt deshalb die
Property geometry_from (step_id der Vorlage), und die Reconstruction kopiert beim Schreiben der Datei die Geometrie.
Bei Füllungen (Fenster, Türen) wird die Öffnung mitgeklont und derselbe Host verwendet.

set_property: ändert den Wert einer vorhandenen IfcPropertySingleValue (Pset + Property per Name), Typ des Werts muss zum
bisherigen passen, Einheiten des Projekts, kein Anlegen/Löschen. set_attribute: nur Name, Description, ObjectType, Tag,
LongName, PredefinedType (gegen Enum des Schemas geprüft). assign_type: bestehendes Typobjekt, nur die Beziehung wechselt.

add_element: legt Wand (start, end, thickness, height), Stütze (position, width, depth, height) oder Platte (outline,
thickness) in einem Geschoss an. Meter, Weltkoordinaten, Fußpunkt auf Geschosshöhe (Platte: Oberkante), optional z_offset und
type_guid. Der Körper entsteht in der Reconstruction aus der Property geometry_spec.

add_hosted_element: legt ein Fenster oder eine Tür in einer Wand an (offset = Abstand der Mitte vom Wandanfang, width, height,
sill_height). Die Wand bleibt unverändert: Öffnung (IfcOpeningElement, eigener Quader) mit IfcRelVoidsElement, Füllung mit
IfcRelFillsElement, Placement-Kette Wand -> Öffnung -> Füllung, Eintrag im Geschoss der Wand. Wanddicke aus der
IfcMaterialLayerSetUsage, Wandlänge (nur Näherung) aus BaseQuantities.Length.

Nutzung (Neo4j):
  python executor.py AC20-FZK-Haus.ifc delete --guid <GlobalId> --uri bolt://localhost:7687 --user neo4j --dry-run
  python executor.py AC20-FZK-Haus.ifc delete --guid <GlobalId> --uri bolt://localhost:7687 --user neo4j
  python executor.py AC20-FZK-Haus.ifc move --guid <GlobalId> --translate 0.5 0 0 --uri bolt://localhost:7687 --user neo4j
  python executor.py AC20-FZK-Haus.ifc move --guid <GlobalId> --target 10 2 0 --uri bolt://localhost:7687 --user neo4j
  python executor.py AC20-FZK-Haus.ifc clone --guid <GlobalId> --translate 1 0 0 [--name Neu] --uri bolt://localhost:7687 --user neo4j
  python executor.py AC20-FZK-Haus.ifc reassign --guid <GlobalId> --storey <GeschossGlobalId> [--keep-world-position] ...
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

# Scope-Grenze laut Operationskatalog: räumliche Struktur und Typobjekte werden abgelehnt
OUT_OF_SCOPE = {"IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcTypeObject", "IfcProject"}

# Knoten, die ein gelöschtes Element "besitzt". Sie werden mitgelöscht, sobald nichts anderes mehr
# auf sie zeigt (Referenzzählung). Material, Typobjekte, OwnerHistory usw. stehen nicht darauf
# und bleiben deshalb immer erhalten.
SWEEPABLE = {
    "IfcPropertySetDefinition",   # IfcPropertySet, IfcElementQuantity
    "IfcProperty",
    "IfcPhysicalQuantity",
    "IfcObjectPlacement",         # IfcLocalPlacement
    "IfcPlacement",               # IfcAxis2Placement3D usw.
    "IfcCartesianPoint",
    "IfcDirection",
    "IfcMaterialLayerSetUsage",
    "IfcMaterialProfileSetUsage",
}


# Neue Knoten bekommen Nummern ab dieser Grenze. So kollidieren sie nie mit den IDs der Original-IFC,
# auch wenn vorher Knoten gelöscht wurden (sonst würde die Differenz eine gelöschte ID als "geändert" lesen).
NEW_ID_BASE = 1_000_000_000

META_KEYS = {"step_id", "source_file", "ifc_class", "geometry_spec"}

# Länge einer Modelleinheit in Metern: SI-Präfix -> Faktor
SI_PREFIX = {None: 1.0, "EXA": 1e18, "PETA": 1e15, "TERA": 1e12, "GIGA": 1e9, "MEGA": 1e6, "KILO": 1e3,
             "HECTO": 1e2, "DECA": 1e1, "DECI": 1e-1, "CENTI": 1e-2, "MILLI": 1e-3, "MICRO": 1e-6,
             "NANO": 1e-9, "PICO": 1e-12, "FEMTO": 1e-15, "ATTO": 1e-18}


class Edge(NamedTuple):
    src: int
    type: str
    idx: int | None
    idx2: int | None
    dst: int

    @property
    def key(self) -> tuple:
        return (self.src, self.type, self.idx, self.idx2)


def _order(e: Edge) -> tuple:
    return (e.src, e.type, -1 if e.idx is None else e.idx, -1 if e.idx2 is None else e.idx2, e.dst)


class ExecutorError(Exception):
    """Operation abgelehnt oder nicht ausführbar. Geht als Fehlermeldung an den Generator."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass
class ExecResult:
    operation: str = ""
    deleted: list[tuple[int, str, str]] = field(default_factory=list)   # (step_id, Klasse, Grund)
    reindexed: list[tuple[int, str]] = field(default_factory=list)      # (Rel-Knoten, Attribut)
    moved: list[tuple[int, str, tuple, tuple]] = field(default_factory=list)  # (Bauteil, Klasse, Location alt, neu), lokal
    created: list[tuple[int, str, str]] = field(default_factory=list)   # (step_id, Klasse, Grund)
    stale: list[tuple[int, str]] = field(default_factory=list)          # Beziehungen, die jetzt veraltet sein können
    reassigned: list[tuple[int, str, int, int]] = field(default_factory=list)  # (Bauteil, Klasse, altes, neues Geschoss)
    new_elements: list[tuple[str, str, int]] = field(default_factory=list)      # (GlobalId, Klasse, step_id) neuer Bauteile
    skipped: list[tuple[int, str]] = field(default_factory=list)        # Beziehungen, die nicht kopiert wurden
    changed: list[tuple[int, str, str, object, object]] = field(default_factory=list)  # (Knoten, Klasse, was, alt, neu)

    def summary(self) -> str:
        head = [f"{self.operation}:"]
        if self.deleted or self.reindexed:
            head.append(f"{len(self.deleted)} Knoten gelöscht, {len(self.reindexed)} Listen neu nummeriert")
        if self.moved:
            head.append(f"{len(self.moved)} Placement(s) verschoben")
        if self.reassigned:
            head.append(f"{len(self.reassigned)} Bauteil(e) umgehängt")
        if self.changed:
            head.append(f"{len(self.changed)} Wert(e) geändert")
        if self.created:
            head.append(f"{len(self.created)} Knoten neu angelegt")
        lines = [" ".join(head)]
        lines += [f"  - {sid} {cls}  ({why})" for sid, cls, why in self.deleted]
        lines += [f"  - verschoben: {sid} {cls}  Location lokal {old} -> {new}" for sid, cls, old, new in self.moved]
        lines += [f"  - umgehängt: {sid} {cls}  Geschoss {old} -> {new}" for sid, cls, old, new in self.reassigned]
        lines += [f"  - geändert: {sid} {cls}  {what}: {old!r} -> {new!r}" for sid, cls, what, old, new in self.changed]
        lines += [f"  - neu: {sid} {cls}  ({why})" for sid, cls, why in self.created]
        lines += [f"  - neues Bauteil: {g_} {cls} (Knoten {sid})" for g_, cls, sid in self.new_elements]
        if self.skipped:
            kinds = sorted({c for _, c in self.skipped})
            lines.append(f"  Hinweis: {len(self.skipped)} Beziehung(en) wurden nicht auf den Klon übertragen "
                         f"({', '.join(kinds)})")
        if self.stale:
            lines.append(f"  Hinweis: {len(self.stale)} Beziehung(en) (SpaceBoundary / ConnectsPathElements) "
                         f"werden nicht neu berechnet und können veraltet sein")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Vektorrechnung (3x3-Drehung als drei Spaltenvektoren)
# ---------------------------------------------------------------------------------------------

def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(a):
    n = math.sqrt(_dot(a, a))
    if n < 1e-12:
        raise ExecutorError("unsupported_placement", "Richtungsvektor der Länge 0 in einem Placement")
    return (a[0] / n, a[1] / n, a[2] / n)


def _apply(r, v):
    """R · v"""
    return tuple(r[0][i] * v[0] + r[1][i] * v[1] + r[2][i] * v[2] for i in range(3))


def _apply_t(r, v):
    """R^T · v (Weltvektor in lokale Koordinaten)"""
    return (_dot(r[0], v), _dot(r[1], v), _dot(r[2], v))


def _compose(ra, rb):
    return tuple(_apply(ra, c) for c in rb)


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------------------------

class MemoryBackend:
    """Arbeitet direkt auf einem Snapshot (neo4j2ifc.Snapshot). Für Tests ohne Datenbank."""

    def __init__(self, snapshot, label_fn: Callable[[str], set[str]]):
        self.nodes = snapshot.nodes
        self.edges = snapshot.edges
        self._label_fn = label_fn
        self._in: dict[int, set] = defaultdict(set)
        self._out: dict[int, set] = defaultdict(set)
        for key, dst in self.edges.items():
            self._in[dst].add(key)
            self._out[key[0]].add(key)
        self._next = max(max(self.nodes, default=0) + 1, NEW_ID_BASE)

    def find_by_guid(self, guid: str) -> int | None:
        hits = [sid for sid, p in self.nodes.items() if p.get("GlobalId") == guid]
        if len(hits) > 1:
            raise ExecutorError("ambiguous_guid", f"GlobalId {guid} kommt {len(hits)}-mal vor")
        return hits[0] if hits else None

    def nodes_with_label(self, label: str) -> list[int]:
        return [sid for sid in self.nodes if label in self.labels(sid)]

    def set_props(self, sid: int, props: dict) -> None:
        self.nodes[sid].update(props)

    def create_node(self, ifc_class: str, props: dict, labels: set[str]) -> int:
        sid = self._next
        self._next += 1
        self.nodes[sid] = {"step_id": sid, "ifc_class": ifc_class, **props}
        return sid

    def add_edge(self, src: int, etype: str, dst: int, idx: int | None = None, idx2: int | None = None) -> None:
        key = (src, etype, idx, idx2)
        self.edges[key] = dst
        self._in[dst].add(key)
        self._out[src].add(key)

    def remove_edge(self, src: int, etype: str, idx: int | None = None, idx2: int | None = None) -> None:
        key = (src, etype, idx, idx2)
        if key in self.edges:
            self._drop(key)

    def props(self, sid: int) -> dict:
        return self.nodes[sid]

    def labels(self, sid: int) -> set[str]:
        return self._label_fn(self.nodes[sid]["ifc_class"])

    def in_edges(self, sid: int) -> list[Edge]:
        return sorted((Edge(*k, self.edges[k]) for k in self._in[sid]), key=_order)

    def out_edges(self, sid: int) -> list[Edge]:
        return sorted((Edge(*k, self.edges[k]) for k in self._out[sid]), key=_order)

    def remove_node(self, sid: int) -> None:
        for key in list(self._out[sid]) + list(self._in[sid]):
            if key in self.edges:
                self._drop(key)
        del self.nodes[sid]

    def reindex(self, src: int, etype: str) -> None:
        keys = sorted((k for k in self._out[src] if k[1] == etype and k[2] is not None and k[3] is None),
                      key=lambda k: k[2])
        dsts = [self.edges[k] for k in keys]
        for k in keys:
            self._drop(k)
        for i, d in enumerate(dsts):
            nk = (src, etype, i, None)
            self.edges[nk] = d
            self._in[d].add(nk)
            self._out[src].add(nk)

    def _drop(self, key: tuple) -> None:
        dst = self.edges.pop(key)
        self._in[dst].discard(key)
        self._out[key[0]].discard(key)


class Neo4jBackend:
    """Setzt die Zugriffe in Cypher um. `tx` ist eine offene Neo4j-Transaktion (Commit macht der Aufrufer)."""

    def __init__(self, tx, source: str):
        self.tx = tx
        self.f = source
        self._nodes: dict[int, tuple[dict, set]] = {}
        self._in: dict[int, list[Edge]] = {}
        self._out: dict[int, list[Edge]] = {}
        self._next: int | None = None

    def _q(self, query: str, **params):
        return list(self.tx.run(query, f=self.f, **params))

    def nodes_with_label(self, label: str) -> list[int]:
        if not label.isidentifier():
            raise ExecutorError("bad_attribute", f"Ungültiges Label {label!r}")
        rows = self._q(f"MATCH (n:`{label}` {{source_file: $f}}) RETURN n.step_id AS node")
        return [r["node"] for r in rows]

    def set_props(self, sid: int, props: dict) -> None:
        self._q("MATCH (n:IfcEntity {source_file: $f, step_id: $s}) SET n += $p", s=sid, p=props)
        self._nodes.pop(sid, None)

    def create_node(self, ifc_class: str, props: dict, labels: set[str]) -> int:
        if self._next is None:
            rows = self._q("MATCH (n:IfcEntity {source_file: $f}) RETURN max(n.step_id) AS m")
            top = rows[0]["m"] if rows and rows[0]["m"] is not None else 0
            self._next = max(top + 1, NEW_ID_BASE)
        sid, self._next = self._next, self._next + 1
        names = sorted(set(labels) | {"IfcEntity"})
        if not all(n.isidentifier() for n in names):
            raise ExecutorError("bad_attribute", f"Ungültige Labels {names}")
        lab = ":".join(f"`{n}`" for n in names)
        self._q(f"CREATE (n:{lab} {{source_file: $f, step_id: $s, ifc_class: $c}}) SET n += $p",
                s=sid, c=ifc_class, p=props)
        return sid

    def add_edge(self, src: int, etype: str, dst: int, idx: int | None = None, idx2: int | None = None) -> None:
        if not etype.isidentifier():
            raise ExecutorError("bad_attribute", f"Ungültiger Kantentyp {etype!r}")
        p = {k: v for k, v in (("idx", idx), ("idx2", idx2)) if v is not None}
        self._q(f"MATCH (a:IfcEntity {{source_file: $f, step_id: $a}}), (b:IfcEntity {{source_file: $f, step_id: $b}}) "
                f"CREATE (a)-[r:`{etype}`]->(b) SET r += $p", a=src, b=dst, p=p)
        self._in.pop(dst, None)
        self._out.pop(src, None)

    def remove_edge(self, src: int, etype: str, idx: int | None = None, idx2: int | None = None) -> None:
        if not etype.isidentifier():
            raise ExecutorError("bad_attribute", f"Ungültiger Kantentyp {etype!r}")
        old = [e.dst for e in self.out_edges(src) if (e.type, e.idx, e.idx2) == (etype, idx, idx2)]
        self._q(f"MATCH (a:IfcEntity {{source_file: $f, step_id: $a}})-[r:`{etype}`]->() "
                f"WHERE coalesce(r.idx, -1) = $i1 AND coalesce(r.idx2, -1) = $i2 DELETE r",
                a=src, i1=-1 if idx is None else idx, i2=-1 if idx2 is None else idx2)
        self._out.pop(src, None)
        for d in old:
            self._in.pop(d, None)

    def find_by_guid(self, guid: str) -> int | None:
        rows = self._q("MATCH (n:IfcRoot {source_file: $f, GlobalId: $g}) RETURN n.step_id AS s", g=guid)
        if len(rows) > 1:
            raise ExecutorError("ambiguous_guid", f"GlobalId {guid} kommt {len(rows)}-mal vor")
        return rows[0]["s"] if rows else None

    def _node(self, sid: int) -> tuple[dict, set]:
        if sid not in self._nodes:
            rows = self._q("MATCH (n:IfcEntity {source_file: $f, step_id: $s}) "
                           "RETURN properties(n) AS p, labels(n) AS l", s=sid)
            if not rows:
                raise ExecutorError("missing_node", f"Knoten {sid} nicht im Graphen")
            self._nodes[sid] = (dict(rows[0]["p"]), set(rows[0]["l"]))
        return self._nodes[sid]

    def props(self, sid: int) -> dict:
        return self._node(sid)[0]

    def labels(self, sid: int) -> set[str]:
        return self._node(sid)[1]

    def in_edges(self, sid: int) -> list[Edge]:
        if sid not in self._in:
            rows = self._q(
                "MATCH (a:IfcEntity {source_file: $f})-[r]->(b:IfcEntity {source_file: $f, step_id: $s}) "
                "WHERE r.inverse IS NULL "
                "RETURN a.step_id AS src, type(r) AS t, r.idx AS i1, r.idx2 AS i2", s=sid)
            self._in[sid] = sorted((Edge(r["src"], r["t"], r["i1"], r["i2"], sid) for r in rows), key=_order)
        return self._in[sid]

    def out_edges(self, sid: int) -> list[Edge]:
        if sid not in self._out:
            rows = self._q(
                "MATCH (a:IfcEntity {source_file: $f, step_id: $s})-[r]->(b:IfcEntity {source_file: $f}) "
                "WHERE r.inverse IS NULL "
                "RETURN b.step_id AS dst, type(r) AS t, r.idx AS i1, r.idx2 AS i2", s=sid)
            self._out[sid] = sorted((Edge(sid, r["t"], r["i1"], r["i2"], r["dst"]) for r in rows), key=_order)
        return self._out[sid]

    def remove_node(self, sid: int) -> None:
        self._q("MATCH (n:IfcEntity {source_file: $f, step_id: $s}) DETACH DELETE n", s=sid)
        self._nodes.pop(sid, None)
        self._in.clear()
        self._out.clear()

    def reindex(self, src: int, etype: str) -> None:
        if not etype.isidentifier():
            raise ExecutorError("bad_attribute", f"Ungültiger Kantentyp {etype!r}")
        self._q(f"MATCH (a:IfcEntity {{source_file: $f, step_id: $s}})-[r:`{etype}`]->() "
                f"WHERE r.idx IS NOT NULL AND r.idx2 IS NULL "
                f"WITH r ORDER BY r.idx WITH collect(r) AS rs "
                f"UNWIND range(0, size(rs) - 1) AS i "
                f"WITH rs[i] AS rel, i SET rel.idx = i", s=src)


# ---------------------------------------------------------------------------------------------
# Executor: Regeln
# ---------------------------------------------------------------------------------------------

class Executor:
    def __init__(self, backend, schema=None):
        self.b = backend
        self.schema = schema          # IfcOpenShell-Schema (für set_attribute, add_element), sonst None
        self._label_cache: dict = {}

    def apply(self, op: dict) -> ExecResult:
        name = op.get("op")
        if name == "delete_element":
            if "guid" not in op:
                raise ExecutorError("bad_parameter", "delete_element braucht 'guid'")
            return self.delete_element(op["guid"], op.get("cascade", "dependents"),
                                       op.get("remove_host_opening", True))
        if name == "move_element":
            if "guid" not in op:
                raise ExecutorError("bad_parameter", "move_element braucht 'guid'")
            return self.move_element(op["guid"], op.get("translation"), op.get("target_location"),
                                     op.get("with_dependents", True))
        if name == "clone_element":
            for k in ("source_guid", "translation"):
                if k not in op:
                    raise ExecutorError("bad_parameter", f"clone_element braucht '{k}'")
            return self.clone_element(op["source_guid"], op["translation"], op.get("name"))
        if name == "reassign_container":
            for k in ("guid", "target_storey_guid"):
                if k not in op:
                    raise ExecutorError("bad_parameter", f"reassign_container braucht '{k}'")
            return self.reassign_container(op["guid"], op["target_storey_guid"],
                                           op.get("preserve_world_position", False))
        if name == "set_property":
            for k in ("guid", "pset_name", "property_name", "value"):
                if k not in op:
                    raise ExecutorError("bad_parameter", f"set_property braucht '{k}'")
            return self.set_property(op["guid"], op["pset_name"], op["property_name"], op["value"])
        if name == "set_attribute":
            for k in ("guid", "attribute", "value"):
                if k not in op:
                    raise ExecutorError("bad_parameter", f"set_attribute braucht '{k}'")
            return self.set_attribute(op["guid"], op["attribute"], op["value"])
        if name == "assign_type":
            for k in ("guid", "type_guid"):
                if k not in op:
                    raise ExecutorError("bad_parameter", f"assign_type braucht '{k}'")
            return self.assign_type(op["guid"], op["type_guid"])
        if name == "add_element":
            for k in ("ifc_class", "storey_guid"):
                if k not in op:
                    raise ExecutorError("bad_parameter", f"add_element braucht '{k}'")
            known = {"op", "ifc_class", "storey_guid", "type_guid", "z_offset"}
            params = {k: v for k, v in op.items() if k not in known}
            return self.add_element(op["ifc_class"], op["storey_guid"], params, op.get("type_guid"),
                                    op.get("z_offset", 0.0))
        if name == "add_hosted_element":
            for k in ("ifc_class", "host_guid"):
                if k not in op:
                    raise ExecutorError("bad_parameter", f"add_hosted_element braucht '{k}'")
            known = {"op", "ifc_class", "host_guid", "type_guid"}
            params = {k: v for k, v in op.items() if k not in known}
            return self.add_hosted_element(op["ifc_class"], op["host_guid"], params, op.get("type_guid"))
        raise ExecutorError("unknown_op", f"Operation {name!r} ist (noch) nicht implementiert")

    # -- Hilfen ------------------------------------------------------------------------------

    def _cls(self, sid: int) -> str:
        return self.b.props(sid)["ifc_class"]

    def _targets(self, sid: int, etype: str) -> list[int]:
        return [e.dst for e in self.b.out_edges(sid) if e.type == etype]

    def _single(self, sid: int, etype: str) -> int | None:
        t = self._targets(sid, etype)
        return t[0] if t else None

    def _check_scope(self, sid: int, verb: str = "gelöscht") -> None:
        labels = self.b.labels(sid)
        blocked = labels & OUT_OF_SCOPE
        if blocked:
            raise ExecutorError("out_of_scope", f"{self._cls(sid)} ({sid}) darf nicht {verb} werden: "
                                f"{', '.join(sorted(blocked))} liegt außerhalb des Umfangs (Scope-Grenze)")
        if "IfcProduct" not in labels:
            raise ExecutorError("out_of_scope", f"{self._cls(sid)} ({sid}) ist kein Bauteil (IfcProduct)")

    # -- delete_element ----------------------------------------------------------------------

    def delete_element(self, guid: str, cascade: str = "dependents",
                       remove_host_opening: bool = True) -> ExecResult:
        if cascade not in ("dependents", "strict"):
            raise ExecutorError("bad_parameter", f"cascade muss 'dependents' oder 'strict' sein, nicht {cascade!r}")
        b = self.b
        root = b.find_by_guid(guid)
        if root is None:
            raise ExecutorError("not_found", f"Kein Element mit GlobalId {guid}")
        self._check_scope(root)

        deleted: dict[int, str] = {root: "Ziel der Operation"}
        products: list[int] = []                       # gelöschte Bauteile (ohne Rel-Objekte)
        removed: set[tuple] = set()                    # Listenkanten, die auf gelöschte Bauteile zeigen
        rel_lists: set[tuple[int, str]] = set()        # (Rel-Knoten, Listenattribut) mit entfernten Elementen
        queue: deque[int] = deque([root])

        def enqueue(sid: int, why: str) -> None:
            if sid not in deleted:
                self._check_scope(sid)
                deleted[sid] = why
                queue.append(sid)

        def strict_stop(x: int, what: str) -> None:
            if cascade == "strict":
                raise ExecutorError("dependents_exist", f"{self._cls(x)} ({x}) hat abhängige Elemente ({what}); "
                                    f"mit cascade='strict' wird die Operation abgelehnt")

        while queue:
            x = queue.popleft()
            products.append(x)
            for e in b.in_edges(x):
                if e.src in deleted:
                    continue                           # Kante verschwindet mit dem Knoten
                if "IfcRelationship" not in b.labels(e.src):
                    raise ExecutorError("unexpected_reference",
                                        f"{self._cls(e.src)} ({e.src}) verweist über '{e.type}' auf {self._cls(x)} "
                                        f"({x}) und ist keine Beziehung; dafür gibt es keine Regel")
                rc = self._cls(e.src)
                why = f"{rc} an {self._cls(x)} {x}"

                if rc == "IfcRelVoidsElement":
                    if e.type == "RelatingBuildingElement":              # x ist Host
                        for o in self._targets(e.src, "RelatedOpeningElement"):
                            strict_stop(x, f"Öffnung {o}")
                            enqueue(o, f"Öffnung von {self._cls(x)} {x}")
                    deleted[e.src] = why                                 # x ist Öffnung: Host bleibt
                elif rc == "IfcRelFillsElement":
                    if e.type == "RelatingOpeningElement":               # x ist Öffnung, Füllung folgt
                        for f_ in self._targets(e.src, "RelatedBuildingElement"):
                            strict_stop(x, f"Füllung {f_}")
                            enqueue(f_, f"Füllung der Öffnung {x}")
                    elif x == root and remove_host_opening:              # x ist Füllung
                        for o in self._targets(e.src, "RelatingOpeningElement"):
                            enqueue(o, f"Öffnung der Füllung {x}")
                    deleted[e.src] = why
                elif rc == "IfcRelAggregates" and e.type == "RelatingObject":
                    parts = self._targets(e.src, "RelatedObjects")
                    for p in parts:
                        strict_stop(x, f"Teil {p}")
                        enqueue(p, f"Teil von {self._cls(x)} {x}")
                    deleted[e.src] = why
                elif rc in ("IfcRelSpaceBoundary", "IfcRelConnectsPathElements") or e.idx is None:
                    deleted[e.src] = why                                 # Beziehung ohne x sinnlos
                else:                                                    # Listenattribut: x entfernen
                    removed.add(e.key)
                    rel_lists.add((e.src, e.type))

        # Rel-Objekte, deren Liste jetzt leer ist, löschen
        for rel, etype in sorted(rel_lists):
            if rel in deleted:
                continue
            rest = [o for o in b.out_edges(rel) if o.type == etype and o.key not in removed]
            if not rest:
                deleted[rel] = f"{self._cls(rel)}: {etype} leer"

        # Verwaiste Besitz-Knoten (Psets, Properties, Quantities, Placements, Punkte ...) mitlöschen
        cand: set[int] = set()
        for d in list(deleted):
            cand.update(o.dst for o in b.out_edges(d) if o.dst not in deleted)
        progress = True
        while progress:
            progress = False
            for c in sorted(cand):
                if c in deleted or not (b.labels(c) & SWEEPABLE):
                    continue
                if any(e.src not in deleted for e in b.in_edges(c)):
                    continue
                deleted[c] = f"verwaist nach Löschen ({self._cls(c)})"
                cand.update(o.dst for o in b.out_edges(c) if o.dst not in deleted)
                progress = True

        # Andere Placements, die relativ zu einem gelöschten Placement liegen: noch nicht unterstützt
        for x in products:
            for pl in self._targets(x, "ObjectPlacement"):
                deps = [e.src for e in b.in_edges(pl) if e.type == "PlacementRelTo" and e.src not in deleted]
                if deps:
                    raise ExecutorError(
                        "placement_dependents",
                        f"Placement {pl} von {self._cls(x)} {x} ist Bezug für {len(deps)} weitere Placements "
                        f"(z. B. {deps[0]}). Umhängen mit Koordinatenumrechnung ist noch nicht implementiert")

        # Ausführen
        res = ExecResult(operation=f"delete_element {guid}")
        for sid in sorted(deleted, key=lambda s: (s not in products, s)):
            res.deleted.append((sid, self._cls(sid), deleted[sid]))
        for sid in deleted:
            b.remove_node(sid)
        for rel, etype in sorted(rel_lists):
            if rel not in deleted:
                b.reindex(rel, etype)
                res.reindexed.append((rel, etype))
        return res

    # -- move_element ------------------------------------------------------------------------

    def _length_scale(self) -> float:
        """Meter pro Modelleinheit (aus der Längeneinheit des Projekts)."""
        projects = self.b.nodes_with_label("IfcProject")
        if len(projects) != 1:
            raise ExecutorError("unsupported_unit", f"{len(projects)} IfcProject-Knoten, erwartet genau einer")
        for ua in self._targets(projects[0], "UnitsInContext"):
            for u in self._targets(ua, "Units"):
                p = self.b.props(u)
                if p.get("UnitType") != "LENGTHUNIT":
                    continue
                if p["ifc_class"] != "IfcSIUnit" or p.get("Name") != "METRE" or p.get("Prefix") not in SI_PREFIX:
                    raise ExecutorError("unsupported_unit", f"Längeneinheit {p['ifc_class']} "
                                        f"{p.get('Prefix') or ''}{p.get('Name') or ''} wird nicht unterstützt "
                                        f"(nur SI-Meter mit Präfix)")
                return SI_PREFIX[p.get("Prefix")]
        raise ExecutorError("unsupported_unit", "Das Projekt hat keine Längeneinheit")

    def _vec3(self, sid: int | None, prop: str, what: str) -> tuple:
        vals = None if sid is None else self.b.props(sid).get(prop)
        if not isinstance(vals, (list, tuple)) or len(vals) != 3:
            raise ExecutorError("unsupported_placement", f"{what} ist nicht dreidimensional oder fehlt")
        return tuple(float(v) for v in vals)

    def _local_frame(self, pl: int) -> tuple[tuple, tuple]:
        """IfcLocalPlacement -> (Drehung, Ursprung) relativ zum Bezugs-Placement."""
        if self._cls(pl) != "IfcLocalPlacement":
            raise ExecutorError("unsupported_placement", f"{self._cls(pl)} ({pl}) wird nicht unterstützt "
                                f"(nur IfcLocalPlacement)")
        rel = self._single(pl, "RelativePlacement")
        if rel is None or self._cls(rel) != "IfcAxis2Placement3D":
            raise ExecutorError("unsupported_placement", f"Placement {pl} hat kein IfcAxis2Placement3D")
        loc = self._vec3(self._single(rel, "Location"), "Coordinates", f"Location von {rel}")
        a, r_ = self._single(rel, "Axis"), self._single(rel, "RefDirection")
        z = _unit(self._vec3(a, "DirectionRatios", f"Axis von {rel}")) if a is not None else (0.0, 0.0, 1.0)
        ref = self._vec3(r_, "DirectionRatios", f"RefDirection von {rel}") if r_ is not None else (1.0, 0.0, 0.0)
        d = _dot(ref, z)
        xr = _unit((ref[0] - d * z[0], ref[1] - d * z[1], ref[2] - d * z[2]))
        return (xr, _cross(z, xr), z), loc

    def _world_frame(self, pl: int, depth: int = 0) -> tuple[tuple, tuple]:
        if depth > 200:
            raise ExecutorError("unsupported_placement", f"Placement-Kette von {pl} ist zu lang oder zyklisch")
        r, o = self._local_frame(pl)
        parent = self._single(pl, "PlacementRelTo")
        if parent is None:
            return r, o
        rp, op = self._world_frame(parent, depth + 1)
        shift = _apply(rp, o)
        return _compose(rp, r), (op[0] + shift[0], op[1] + shift[1], op[2] + shift[2])

    def _ancestors(self, pl: int) -> list[int]:
        out: list[int] = []
        cur = self._single(pl, "PlacementRelTo")
        while cur is not None and len(out) <= 200:
            out.append(cur)
            cur = self._single(cur, "PlacementRelTo")
        return out

    def _move_members(self, root: int, with_dependents: bool) -> list[int]:
        """Das Element und, bei with_dependents, Öffnungen, Füllungen und Teile (wie bei delete_element)."""
        members: dict[int, None] = {root: None}
        queue: deque[int] = deque([root])

        def add(sid: int) -> None:
            if sid not in members:
                members[sid] = None
                queue.append(sid)

        while queue:
            x = queue.popleft()
            for e in self.b.in_edges(x):
                rc = self._cls(e.src)
                if rc == "IfcRelVoidsElement" and e.type == "RelatingBuildingElement" and with_dependents:
                    for o in self._targets(e.src, "RelatedOpeningElement"):
                        add(o)
                elif rc == "IfcRelFillsElement" and with_dependents:
                    if e.type == "RelatingOpeningElement":               # x ist Öffnung: Füllung folgt
                        for f_ in self._targets(e.src, "RelatedBuildingElement"):
                            add(f_)
                    elif x == root:                                      # x ist Füllung: Öffnung mitnehmen
                        for o in self._targets(e.src, "RelatingOpeningElement"):
                            add(o)
                elif rc == "IfcRelAggregates" and e.type == "RelatingObject" and with_dependents:
                    for p in self._targets(e.src, "RelatedObjects"):
                        add(p)
        return list(members)

    def move_element(self, guid: str, translation=None, target_location=None,
                     with_dependents: bool = True) -> ExecResult:
        if (translation is None) == (target_location is None):
            raise ExecutorError("bad_parameter", "Genau eines von 'translation' und 'target_location' angeben")
        param = "translation" if translation is not None else "target_location"
        vec = translation if translation is not None else target_location
        if (not isinstance(vec, (list, tuple)) or len(vec) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vec)):
            raise ExecutorError("bad_parameter", f"'{param}' muss aus drei endlichen Zahlen [x, y, z] in Metern bestehen")
        if not isinstance(with_dependents, bool):
            raise ExecutorError("bad_parameter", "'with_dependents' muss true oder false sein")

        b = self.b
        root = b.find_by_guid(guid)
        if root is None:
            raise ExecutorError("not_found", f"Kein Element mit GlobalId {guid}")
        self._check_scope(root, "verschoben")

        members = self._move_members(root, with_dependents)
        placements: dict[int, int] = {}
        for m in members:
            self._check_scope(m, "verschoben")
            pl = self._single(m, "ObjectPlacement")
            if pl is None:
                raise ExecutorError("no_placement", f"{self._cls(m)} ({m}) hat kein Placement")
            if len([e for e in b.in_edges(pl) if e.type == "ObjectPlacement"]) > 1:
                raise ExecutorError("shared_placement", f"Placement {pl} von {self._cls(m)} {m} gehört zu mehreren "
                                    f"Bauteilen; Verschieben würde alle mitbewegen")
            placements[m] = pl

        scale = self._length_scale()
        if translation is not None:
            d_world = tuple(float(v) / scale for v in vec)
        else:
            _, origin = self._world_frame(placements[root])
            d_world = tuple(float(v) / scale - o for v, o in zip(vec, origin))
        if max(abs(c) for c in d_world) * scale < 1e-9:
            raise ExecutorError("no_change", "Die Verschiebung ist 0 (Element liegt schon an der Zielposition)")

        if not with_dependents:
            deps = [e.src for e in b.in_edges(placements[root]) if e.type == "PlacementRelTo"]
            if deps:
                raise ExecutorError(
                    "placement_dependents",
                    f"{len(deps)} weitere Placements hängen am Placement von {self._cls(root)} {root} "
                    f"(z. B. {deps[0]}) und würden mitwandern. Gegenbewegung ist nicht implementiert; "
                    f"mit with_dependents=true verschieben")

        # Nur die obersten Placements bewegen. Wer per Placement-Kette an einem davon hängt, folgt von selbst.
        pl_set = set(placements.values())
        tops = sorted((m, pl) for m, pl in placements.items()
                      if not any(a in pl_set for a in self._ancestors(pl)))

        plan = []
        for m, pl in tops:
            parent = self._single(pl, "PlacementRelTo")
            rp = self._world_frame(parent)[0] if parent is not None else IDENTITY
            d_local = _apply_t(rp, d_world)
            rel = self._single(pl, "RelativePlacement")
            pt = self._single(rel, "Location")
            if any(e.src != pl for e in b.in_edges(rel)):
                raise ExecutorError("shared_resource", f"IfcAxis2Placement3D {rel} wird von mehreren Placements "
                                    f"genutzt; Verschieben würde alle mitbewegen")
            old = tuple(float(v) for v in b.props(pt)["Coordinates"])
            new = tuple(round(o + dl, 9) + 0.0 for o, dl in zip(old, d_local))
            plan.append((m, rel, pt, old, new, any(e.src != rel for e in b.in_edges(pt))))

        res = ExecResult(operation=f"move_element {guid}")
        for m, rel, pt, old, new, shared in plan:
            self._write_location(rel, pt, new, shared, res)
            res.moved.append((m, self._cls(m), old, new))
        self._mark_stale(members, res)
        return res

    def _write_location(self, rel: int, pt: int, new: tuple, shared: bool, res: ExecResult) -> None:
        """Neue Koordinaten in den Location-Punkt schreiben. Ist der Punkt geteilt, wird er kopiert."""
        b = self.b
        if shared:
            props = {k: v for k, v in b.props(pt).items() if k not in META_KEYS}
            props["Coordinates"] = list(new)
            new_pt = b.create_node(self._cls(pt), props, b.labels(pt))
            b.remove_edge(rel, "Location")
            b.add_edge(rel, "Location", new_pt)
            res.created.append((new_pt, self._cls(pt), f"Kopie des geteilten Punkts {pt}"))
        else:
            b.set_props(pt, {"Coordinates": list(new)})

    def _mark_stale(self, members: list[int], res: ExecResult) -> None:
        """Beziehungen, die nach einer Lageänderung nicht neu berechnet werden, als veraltet melden."""
        seen: set[int] = set()
        for m in members:
            for e in self.b.in_edges(m):
                rc = self._cls(e.src)
                if rc in ("IfcRelSpaceBoundary", "IfcRelConnectsPathElements") and e.src not in seen:
                    seen.add(e.src)
                    res.stale.append((e.src, rc))

    # -- reassign_container ------------------------------------------------------------------

    # (Beziehungsklasse, Kante zum Geschoss, Listenattribut)
    CONTAINER_KINDS = (("IfcRelContainedInSpatialStructure", "RelatingStructure", "RelatedElements"),
                       ("IfcRelAggregates", "RelatingObject", "RelatedObjects"))

    def _container_edges(self, sid: int) -> list[tuple[Edge, tuple, int]]:
        """Eingehende Listenkanten, über die ein Bauteil in einem Geschoss liegt: (Kante, Art, Geschoss)."""
        out = []
        for e in self.b.in_edges(sid):
            for kind in self.CONTAINER_KINDS:
                if self._cls(e.src) == kind[0] and e.type == kind[2]:
                    st = self._single(e.src, kind[1])
                    if st is not None and "IfcBuildingStorey" in self.b.labels(st):
                        out.append((e, kind, st))
        return out

    def _target_relation(self, target: int, kind: tuple, like_rel: int, res: ExecResult) -> int:
        """Beziehung, in die Bauteile des Zielgeschosses eingetragen werden. Fehlt sie, wird sie angelegt."""
        rels = sorted(e.src for e in self.b.in_edges(target) if e.type == kind[1] and self._cls(e.src) == kind[0])
        if rels:
            return rels[0]
        existing = self.b.nodes_with_label(kind[0])
        if not existing:
            raise ExecutorError("no_target_relation", f"Zielgeschoss {target} hat keine {kind[0]} und es gibt "
                                f"keine Vorlage zum Anlegen")
        import ifcopenshell.guid
        new = self.b.create_node(kind[0], {"GlobalId": ifcopenshell.guid.new()}, self.b.labels(existing[0]))
        self.b.add_edge(new, kind[1], target)
        for oh in self._targets(like_rel, "OwnerHistory"):
            self.b.add_edge(new, "OwnerHistory", oh)
        res.created.append((new, kind[0], f"neue Beziehung für Geschoss {target}"))
        return new

    def reassign_container(self, guid: str, target_storey_guid: str,
                           preserve_world_position: bool = False) -> ExecResult:
        if not isinstance(preserve_world_position, bool):
            raise ExecutorError("bad_parameter", "'preserve_world_position' muss true oder false sein")
        b = self.b
        root = b.find_by_guid(guid)
        if root is None:
            raise ExecutorError("not_found", f"Kein Element mit GlobalId {guid}")
        self._check_scope(root, "umgehängt")
        target = b.find_by_guid(target_storey_guid)
        if target is None:
            raise ExecutorError("not_found", f"Kein Geschoss mit GlobalId {target_storey_guid}")
        if "IfcBuildingStorey" not in b.labels(target):
            raise ExecutorError("bad_parameter", f"{self._cls(target)} ({target}) ist kein IfcBuildingStorey")

        for e in b.in_edges(root):
            rc = self._cls(e.src)
            if (rc == "IfcRelFillsElement" and e.type == "RelatedBuildingElement") or \
               (rc == "IfcRelVoidsElement" and e.type == "RelatedOpeningElement"):
                raise ExecutorError("hosted_element", f"{self._cls(root)} ({root}) sitzt in einem Host (Wand). "
                                    f"Den Host umhängen, die Füllung folgt")
        root_edges = self._container_edges(root)
        if len(root_edges) != 1:
            raise ExecutorError(
                "no_container",
                f"{self._cls(root)} ({root}) liegt in {len(root_edges)} Geschossen. Füllungen, Öffnungen und "
                f"Teile gehören zu ihrem Host bzw. Ganzen: dieses statt des Teils umhängen")
        old = root_edges[0][2]
        if old == target:
            raise ExecutorError("no_change", f"{self._cls(root)} liegt schon im Zielgeschoss")

        members = self._move_members(root, True)
        moves = []                                           # (Bauteil, Kante, Art)
        for m in members:
            self._check_scope(m, "umgehängt")
            es = self._container_edges(m)
            if not es:
                continue                                      # z. B. Öffnungen: nur über den Host eingebunden
            if len(es) != 1 or es[0][2] != old:
                raise ExecutorError("unexpected_container", f"{self._cls(m)} ({m}) liegt nicht (nur) im Geschoss "
                                    f"{old} wie das Zielelement")
            moves.append((m, es[0][0], es[0][1]))

        old_pl, new_pl = self._single(old, "ObjectPlacement"), self._single(target, "ObjectPlacement")
        if old_pl is None or new_pl is None:
            raise ExecutorError("no_placement", "Quell- oder Zielgeschoss hat kein Placement")

        placements: dict[int, int] = {}
        for m in members:
            pl = self._single(m, "ObjectPlacement")
            if pl is None:
                raise ExecutorError("no_placement", f"{self._cls(m)} ({m}) hat kein Placement")
            if len([e for e in b.in_edges(pl) if e.type == "ObjectPlacement"]) > 1:
                raise ExecutorError("shared_placement", f"Placement {pl} von {self._cls(m)} {m} gehört zu mehreren "
                                    f"Bauteilen")
            placements[m] = pl
        pl_set = set(placements.values())
        tops = sorted((m, pl) for m, pl in placements.items() if not any(a in pl_set for a in self._ancestors(pl)))
        for m, pl in tops:
            if self._single(pl, "PlacementRelTo") != old_pl:
                raise ExecutorError(
                    "unsupported_placement",
                    f"Das Placement von {self._cls(m)} ({m}) ist nicht direkt relativ zum Placement des Geschosses "
                    f"{old}; Umhängen ist dafür nicht umgesetzt")

        plan = []                                            # (Bauteil, Placement, rel, pt, alt, neu, geteilt)
        if preserve_world_position:
            r_old, o_old = self._world_frame(old_pl)
            r_new, o_new = self._world_frame(new_pl)
            if any(abs(a - c) > 1e-9 for va, vc in zip(r_old, r_new) for a, c in zip(va, vc)):
                raise ExecutorError("unsupported_rotation", "Die beiden Geschosse haben unterschiedliche Drehung; "
                                    "die Weltlage zu halten würde die Orientierung ändern (nicht umgesetzt)")
            for m, pl in tops:
                rel = self._single(pl, "RelativePlacement")
                pt = self._single(rel, "Location")
                if any(e.src != pl for e in b.in_edges(rel)):
                    raise ExecutorError("shared_resource", f"IfcAxis2Placement3D {rel} wird von mehreren Placements "
                                        f"genutzt")
                loc = tuple(float(v) for v in b.props(pt)["Coordinates"])
                shift = _apply(r_old, loc)
                world = (o_old[0] + shift[0], o_old[1] + shift[1], o_old[2] + shift[2])
                new = _apply_t(r_new, (world[0] - o_new[0], world[1] - o_new[1], world[2] - o_new[2]))
                new = tuple(round(v, 9) + 0.0 for v in new)
                plan.append((m, pl, rel, pt, loc, new, any(e.src != rel for e in b.in_edges(pt))))

        res = ExecResult(operation=f"reassign_container {guid}")
        # Placement: auf das Placement des Zielgeschosses umhängen
        for m, pl in tops:
            b.remove_edge(pl, "PlacementRelTo")
            b.add_edge(pl, "PlacementRelTo", new_pl)
        for m, pl, rel, pt, loc, new, shared in plan:
            if new != tuple(round(v, 9) + 0.0 for v in loc):
                self._write_location(rel, pt, new, shared, res)
                res.moved.append((m, self._cls(m), loc, new))

        # Containment: aus der alten Beziehung austragen, in die des Zielgeschosses eintragen
        touched: set[tuple[int, str]] = set()
        new_rels: dict[tuple, int] = {}
        for m, e, kind in moves:
            b.remove_edge(e.src, e.type, e.idx, e.idx2)
            touched.add((e.src, e.type))
            if kind not in new_rels:
                new_rels[kind] = self._target_relation(target, kind, e.src, res)
            rel = new_rels[kind]
            n_ = len([o for o in b.out_edges(rel) if o.type == kind[2]])
            b.add_edge(rel, kind[2], m, n_)
            res.reassigned.append((m, self._cls(m), old, target))
        for rel, etype in sorted(touched):
            rest = [o for o in b.out_edges(rel) if o.type == etype]
            if not rest:
                res.deleted.append((rel, self._cls(rel), f"{etype} leer"))
                b.remove_node(rel)
            else:
                b.reindex(rel, etype)
                res.reindexed.append((rel, etype))
        self._mark_stale(members, res)
        return res

    # -- clone_element -----------------------------------------------------------------------

    # Knoten, die beim Klonen als eigene Kopie entstehen (alles andere bleibt geteilt)
    COPY_OWNED = {"IfcPropertySetDefinition", "IfcProperty", "IfcPhysicalQuantity"}
    USAGE_OWNED = {"IfcMaterialLayerSetUsage", "IfcMaterialProfileSetUsage"}
    APPEND_RELS = {("IfcRelContainedInSpatialStructure", "RelatedElements"),
                   ("IfcRelDefinesByType", "RelatedObjects"),
                   ("IfcRelAggregates", "RelatedObjects"),
                   ("IfcRelAssociatesMaterial", "RelatedObjects")}

    @staticmethod
    def _new_guid() -> str:
        import ifcopenshell.guid
        return ifcopenshell.guid.new()

    def _clone_node(self, sid: int, res: ExecResult, why: str, drop=(), override=None, extra=None) -> int:
        """Knoten mit allen Properties kopieren (neue GlobalId, falls vorhanden). Kanten werden nicht kopiert."""
        props = {k: v for k, v in self.b.props(sid).items() if k not in META_KEYS and k not in drop}
        if "GlobalId" in props:
            props["GlobalId"] = self._new_guid()
        props.update(override or {})
        props.update(extra or {})
        new = self.b.create_node(self._cls(sid), props, self.b.labels(sid))
        res.created.append((new, self._cls(sid), why))
        return new

    def _copy_edges(self, old: int, new: int, skip=(), mapping=None) -> None:
        for e in self.b.out_edges(old):
            if e.type not in skip:
                dst = (mapping or {}).get(e.dst, e.dst)
                self.b.add_edge(new, e.type, dst, e.idx, e.idx2)

    def _copy_owned(self, sid: int, res: ExecResult, why: str, owned: set[str]) -> int:
        """Pset samt Properties bzw. Mengen als eigene Kopie anlegen. Einheiten, Enumerationen usw. bleiben geteilt."""
        new = self._clone_node(sid, res, why)
        for e in self.b.out_edges(sid):
            dst = self._copy_owned(e.dst, res, why, owned) if self.b.labels(e.dst) & owned else e.dst
            self.b.add_edge(new, e.type, dst, e.idx, e.idx2)
        return new

    def _copy_placement(self, pl: int, mapping: dict, d_local, res: ExecResult) -> int:
        """IfcLocalPlacement mit eigener Achse, eigenen Punkten und Richtungen. Bezug (PlacementRelTo) bleibt
        derselbe, außer er gehört selbst zu den geklonten Bauteilen."""
        b = self.b
        new_pl = b.create_node(self._cls(pl), {}, b.labels(pl))
        res.created.append((new_pl, self._cls(pl), "Placement des Klons"))
        parent = self._single(pl, "PlacementRelTo")
        if parent is not None:
            b.add_edge(new_pl, "PlacementRelTo", mapping.get(parent, parent))
        rel = self._single(pl, "RelativePlacement")
        new_rel = b.create_node(self._cls(rel), {k: v for k, v in b.props(rel).items() if k not in META_KEYS},
                                b.labels(rel))
        res.created.append((new_rel, self._cls(rel), "Achsenplacement des Klons"))
        b.add_edge(new_pl, "RelativePlacement", new_rel)
        for e in b.out_edges(rel):
            props = {k: v for k, v in b.props(e.dst).items() if k not in META_KEYS}
            if e.type == "Location" and d_local is not None:
                props["Coordinates"] = [round(float(c) + d, 9) + 0.0 for c, d in zip(props["Coordinates"], d_local)]
            node = b.create_node(self._cls(e.dst), props, b.labels(e.dst))
            res.created.append((node, self._cls(e.dst), f"{e.type} des Klons"))
            b.add_edge(new_rel, e.type, node, e.idx, e.idx2)
        return new_pl

    def clone_element(self, source_guid: str, translation, name: str | None = None) -> ExecResult:
        vec = translation
        if (not isinstance(vec, (list, tuple)) or len(vec) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vec)):
            raise ExecutorError("bad_parameter", "'translation' muss aus drei endlichen Zahlen [dx, dy, dz] in Metern bestehen")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ExecutorError("bad_parameter", "'name' muss ein nicht leerer Text sein")
        b = self.b
        root = b.find_by_guid(source_guid)
        if root is None:
            raise ExecutorError("not_found", f"Kein Element mit GlobalId {source_guid}")
        self._check_scope(root, "geklont")

        opening = fills_rel = None
        for e in b.in_edges(root):
            rc = self._cls(e.src)
            if rc == "IfcRelVoidsElement" and e.type == "RelatedOpeningElement":
                raise ExecutorError("hosted_element", f"{self._cls(root)} ({root}) ist eine Öffnung. Die Füllung "
                                    f"(Fenster, Tür) klonen, die Öffnung wird mitgeklont")
            if rc == "IfcRelAggregates" and e.type == "RelatingObject":
                raise ExecutorError("has_parts", f"{self._cls(root)} ({root}) besteht aus Teilen; Klonen von "
                                    f"zusammengesetzten Bauteilen ist nicht umgesetzt")
            if rc == "IfcRelFillsElement" and e.type == "RelatedBuildingElement":
                fills_rel = e.src
                ops = self._targets(e.src, "RelatingOpeningElement")
                opening = ops[0] if ops else None
        members = [root] + ([opening] if opening is not None else [])

        placements: dict[int, int] = {}
        for m in members:
            pl = self._single(m, "ObjectPlacement")
            if pl is None:
                raise ExecutorError("no_placement", f"{self._cls(m)} ({m}) hat kein Placement")
            placements[m] = pl
        scale = self._length_scale()
        d_world = tuple(float(v) / scale for v in vec)
        if max(abs(c) for c in d_world) * scale < 1e-9:
            raise ExecutorError("bad_parameter", "Der Versatz ist 0: die Kopie läge genau auf dem Original")

        pl_set = set(placements.values())
        tops = {m for m, pl in placements.items() if not any(a in pl_set for a in self._ancestors(pl))}
        d_local: dict[int, tuple] = {}
        for m in tops:
            parent = self._single(placements[m], "PlacementRelTo")
            rp = self._world_frame(parent)[0] if parent is not None else IDENTITY
            d_local[m] = _apply_t(rp, d_world)
        # Vorab prüfen, damit bei einem Fehler nichts angelegt ist
        for m in members:
            self._world_frame(placements[m])
        host = None
        if opening is not None:
            rv = [e.src for e in b.in_edges(opening) if self._cls(e.src) == "IfcRelVoidsElement"]
            hosts = self._targets(rv[0], "RelatingBuildingElement") if rv else []
            if not hosts:
                raise ExecutorError("unexpected_structure", f"Öffnung {opening} hat keinen Host")
            host, voids_rel = hosts[0], rv[0]

        res = ExecResult(operation=f"clone_element {source_guid}")
        mapping: dict[int, int] = {}
        order = sorted(members, key=lambda m: len([a for a in self._ancestors(placements[m]) if a in pl_set]))
        for m in order:
            mapping[placements[m]] = self._copy_placement(placements[m], mapping, d_local.get(m), res)

        clones: dict[int, int] = {}
        for m in members:
            why = "Klon" if m == root else f"Öffnung des Klons"
            clones[m] = self._clone_node(m, res, why, drop=("Tag",), override={"Name": name} if name else None,
                                         extra={"geometry_from": m})
            for e in b.out_edges(m):
                if e.type == "ObjectPlacement":
                    b.add_edge(clones[m], e.type, mapping[e.dst])
                else:
                    b.add_edge(clones[m], e.type, e.dst, e.idx, e.idx2)
        guid_of = lambda sid: b.props(sid).get("GlobalId")
        res.new_elements.append((b.props(clones[root]).get("GlobalId"), self._cls(root), clones[root]))

        skipped: dict[tuple[int, str], None] = {}
        for m in members:
            for e in b.in_edges(m):
                rc = self._cls(e.src)
                if (rc, e.type) in self.APPEND_RELS:
                    if rc == "IfcRelAssociatesMaterial" and any(
                            self.b.labels(t) & self.USAGE_OWNED for t in self._targets(e.src, "RelatingMaterial")):
                        rel = self._clone_node(e.src, res, "Material-Beziehung des Klons")
                        self._copy_edges(e.src, rel, skip=("RelatedObjects", "RelatingMaterial"))
                        usage = self._targets(e.src, "RelatingMaterial")[0]
                        b.add_edge(rel, "RelatingMaterial", self._copy_owned(usage, res, "Materialnutzung des Klons",
                                                                             self.USAGE_OWNED))
                        b.add_edge(rel, "RelatedObjects", clones[m], 0)
                    else:
                        n_ = len([o for o in b.out_edges(e.src) if o.type == e.type])
                        b.add_edge(e.src, e.type, clones[m], n_)
                elif rc == "IfcRelDefinesByProperties" and e.type == "RelatedObjects":
                    rel = self._clone_node(e.src, res, "Pset-Beziehung des Klons")
                    self._copy_edges(e.src, rel, skip=("RelatedObjects", "RelatingPropertyDefinition"))
                    for d in self._targets(e.src, "RelatingPropertyDefinition"):
                        b.add_edge(rel, "RelatingPropertyDefinition",
                                   self._copy_owned(d, res, "Pset-Kopie des Klons", self.COPY_OWNED))
                    b.add_edge(rel, "RelatedObjects", clones[m], 0)
                elif rc == "IfcRelFillsElement" and e.src == fills_rel:
                    continue                                                  # wird unten neu aufgebaut
                elif rc == "IfcRelVoidsElement" and opening is not None and m == opening:
                    continue
                else:
                    skipped[(e.src, rc)] = None
        for sid, rc in skipped:
            res.skipped.append((sid, rc))

        if opening is not None:
            new_v = self._clone_node(voids_rel, res, "Void-Beziehung des Klons")
            self._copy_edges(voids_rel, new_v, skip=("RelatingBuildingElement", "RelatedOpeningElement"))
            b.add_edge(new_v, "RelatingBuildingElement", host)
            b.add_edge(new_v, "RelatedOpeningElement", clones[opening])
            new_f = self._clone_node(fills_rel, res, "Fill-Beziehung des Klons")
            self._copy_edges(fills_rel, new_f, skip=("RelatingOpeningElement", "RelatedBuildingElement"))
            b.add_edge(new_f, "RelatingOpeningElement", clones[opening])
            b.add_edge(new_f, "RelatedBuildingElement", clones[root])
        return res

    # -- Phase 2: set_property, set_attribute, assign_type -----------------------------------

    # Attribute, die set_attribute setzen darf (Entscheidung 02.11.2026). Nicht erlaubt: GlobalId, Referenzen,
    # Placement, Geometrie, OwnerHistory.
    ATTR_WHITELIST = ("Name", "Description", "ObjectType", "Tag", "LongName", "PredefinedType")

    def _schema_attributes(self, ifc_class: str) -> dict:
        if self.schema is None:
            raise ExecutorError("no_schema", "Dem Executor wurde kein IFC-Schema übergeben (Schema der Datei nötig)")
        try:
            decl = self.schema.declaration_by_name(ifc_class)
        except Exception:
            raise ExecutorError("unknown_class", f"Klasse {ifc_class} ist im Schema nicht bekannt")
        return {a.name(): a for a in decl.all_attributes()}

    def set_property(self, guid: str, pset_name: str, property_name: str, value) -> ExecResult:
        """Ändert den Wert einer vorhandenen IfcPropertySingleValue in einem Pset des Bauteils.
        Der Wert gilt in den Einheiten des Projekts (keine Umrechnung). Neu anlegen und Löschen sind nicht Teil davon."""
        for k, v in (("pset_name", pset_name), ("property_name", property_name)):
            if not isinstance(v, str) or not v.strip():
                raise ExecutorError("bad_parameter", f"'{k}' muss ein nicht leerer Text sein")
        b = self.b
        root = b.find_by_guid(guid)
        if root is None:
            raise ExecutorError("not_found", f"Kein Element mit GlobalId {guid}")
        self._check_scope(root, "geändert")

        psets = []                                        # (Beziehung, Pset)
        for e in b.in_edges(root):
            if e.type == "RelatedObjects" and self._cls(e.src) == "IfcRelDefinesByProperties":
                for d in self._targets(e.src, "RelatingPropertyDefinition"):
                    if b.props(d).get("Name") == pset_name:
                        psets.append((e.src, d))
        if not psets:
            known = sorted({b.props(d).get("Name") for e in b.in_edges(root)
                            if e.type == "RelatedObjects" and self._cls(e.src) == "IfcRelDefinesByProperties"
                            for d in self._targets(e.src, "RelatingPropertyDefinition")} - {None})
            raise ExecutorError("pset_not_found", f"{self._cls(root)} ({root}) hat kein Pset {pset_name!r}. "
                                f"Vorhanden: {', '.join(known) or 'keine'}")
        if len(psets) > 1:
            raise ExecutorError("ambiguous_pset", f"{len(psets)} Psets mit dem Namen {pset_name!r} am Bauteil")
        rel, pset = psets[0]
        if "IfcPropertySet" not in b.labels(pset):
            raise ExecutorError("unsupported_property", f"{pset_name!r} ist ein {self._cls(pset)}. Mengen "
                                f"(ElementQuantity) sind abgeleitete Werte und werden nicht gesetzt")
        if len(self._targets(rel, "RelatedObjects")) > 1 or \
                len([e for e in b.in_edges(pset) if e.type == "RelatingPropertyDefinition"]) > 1:
            raise ExecutorError("shared_resource", f"Pset {pset_name!r} gehört zu mehreren Bauteilen; Ändern würde "
                                f"auch die anderen ändern")
        props = [p for p in self._targets(pset, "HasProperties") if b.props(p).get("Name") == property_name]
        if not props:
            known = sorted(str(b.props(p).get("Name")) for p in self._targets(pset, "HasProperties"))
            raise ExecutorError("property_not_found", f"Pset {pset_name!r} hat keine Property {property_name!r}. "
                                f"Vorhanden: {', '.join(known) or 'keine'}")
        if len(props) > 1:
            raise ExecutorError("ambiguous_property", f"{len(props)} Properties {property_name!r} im Pset {pset_name!r}")
        prop = props[0]
        if self._cls(prop) != "IfcPropertySingleValue":
            raise ExecutorError("unsupported_property", f"{property_name!r} ist eine {self._cls(prop)}; nur "
                                f"IfcPropertySingleValue wird unterstützt")
        if len([e for e in b.in_edges(prop) if e.type == "HasProperties"]) > 1:
            raise ExecutorError("shared_resource", f"Property {property_name!r} gehört zu mehreren Psets")
        old = b.props(prop).get("NominalValue")
        tname = b.props(prop).get("NominalValue_type")
        if old is None or isinstance(old, list):
            raise ExecutorError("unsupported_property", f"{property_name!r} hat keinen einfachen Wert (leer oder Liste)")
        if isinstance(old, bool):
            ok, new = isinstance(value, bool), value
        elif isinstance(old, int):
            ok, new = isinstance(value, int) and not isinstance(value, bool), value
        elif isinstance(old, float):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            new = float(value) if ok else value
        else:
            ok, new = isinstance(value, str), value
        if not ok:
            want = {bool: "true/false", int: "eine ganze Zahl", float: "eine endliche Zahl", str: "ein Text"}[type(old)]
            raise ExecutorError("bad_value_type", f"{property_name!r} ist {tname or type(old).__name__}; "
                                f"erwartet wird {want}, bekommen: {value!r}")
        if new == old:
            raise ExecutorError("no_change", f"{property_name!r} hat schon den Wert {old!r}")
        b.set_props(prop, {"NominalValue": new})
        res = ExecResult(operation=f"set_property {guid}")
        res.changed.append((prop, self._cls(prop), f"{pset_name}.{property_name}", old, new))
        return res

    def set_attribute(self, guid: str, attribute: str, value) -> ExecResult:
        """Setzt ein einfaches Attribut aus der Whitelist (ATTR_WHITELIST) auf einen neuen Wert."""
        b = self.b
        if attribute not in self.ATTR_WHITELIST:
            raise ExecutorError("attribute_not_allowed", f"Attribut {attribute!r} darf nicht gesetzt werden. "
                                f"Erlaubt: {', '.join(self.ATTR_WHITELIST)}")
        if not isinstance(value, str) or not value.strip():
            raise ExecutorError("bad_parameter", "'value' muss ein nicht leerer Text sein")
        root = b.find_by_guid(guid)
        if root is None:
            raise ExecutorError("not_found", f"Kein Element mit GlobalId {guid}")
        self._check_scope(root, "geändert")
        cls = self._cls(root)
        attrs = self._schema_attributes(cls)
        if attribute not in attrs:
            raise ExecutorError("attribute_not_on_class", f"{cls} hat kein Attribut {attribute!r}")
        if attribute == "PredefinedType":
            decl = attrs[attribute].type_of_attribute()
            try:
                items = list(decl.declared_type().enumeration_items())
            except Exception:
                raise ExecutorError("unsupported_attribute", f"PredefinedType von {cls} ist keine Aufzählung")
            if value not in items:
                raise ExecutorError("bad_enum_value", f"{value!r} gehört nicht zu {decl.declared_type().name()}. "
                                    f"Erlaubt: {', '.join(items)}")
            if value == "USERDEFINED" and not b.props(root).get("ObjectType"):
                raise ExecutorError("needs_object_type", "PredefinedType USERDEFINED verlangt ein gesetztes "
                                    "ObjectType. Erst ObjectType setzen")
        old = b.props(root).get(attribute)
        if old == value:
            raise ExecutorError("no_change", f"{attribute} hat schon den Wert {old!r}")
        b.set_props(root, {attribute: value})
        res = ExecResult(operation=f"set_attribute {guid}")
        res.changed.append((root, cls, attribute, old, value))
        return res

    def assign_type(self, guid: str, type_guid: str) -> ExecResult:
        """Hängt ein Bauteil an ein anderes, bereits vorhandenes Typobjekt (IfcRelDefinesByType). Geometrie, Material und
        Instanzdaten bleiben unverändert."""
        b = self.b
        el = b.find_by_guid(guid)
        if el is None:
            raise ExecutorError("not_found", f"Kein Element mit GlobalId {guid}")
        self._check_scope(el, "geändert")
        ty = b.find_by_guid(type_guid)
        if ty is None:
            raise ExecutorError("not_found", f"Kein Typobjekt mit GlobalId {type_guid}")
        self._check_type_fits(b.labels(el), self._cls(el), ty)
        cur = [e for e in b.in_edges(el) if e.type == "RelatedObjects" and self._cls(e.src) == "IfcRelDefinesByType"]
        if len(cur) > 1:
            raise ExecutorError("unexpected_structure", f"{self._cls(el)} ({el}) hat {len(cur)} Typ-Beziehungen")
        old_ty = self._single(cur[0].src, "RelatingType") if cur else None
        if old_ty == ty:
            raise ExecutorError("no_change", f"{self._cls(el)} ({el}) hat den Typ {ty} schon")
        res = ExecResult(operation=f"assign_type {guid}")
        self._link_type(el, ty, cur, res)
        if cur:
            old = cur[0]
            b.remove_edge(old.src, "RelatedObjects", old.idx, old.idx2)
            if self._targets(old.src, "RelatedObjects"):
                b.reindex(old.src, "RelatedObjects")
                res.reindexed.append((old.src, "RelatedObjects"))
            else:
                b.remove_node(old.src)
                res.deleted.append((old.src, "IfcRelDefinesByType", "Beziehung leer geworden"))
        res.changed.append((el, self._cls(el), "Typ", b.props(old_ty).get("Name") if old_ty is not None else None,
                            b.props(ty).get("Name")))
        return res

    # -- add_element -------------------------------------------------------------------------

    # erlaubte Klassen und Parameter (alle Längen in Metern, Koordinaten in Weltkoordinaten)
    ADD_PARAMS = {"IfcWall": ("start", "end", "thickness", "height"),
                  "IfcColumn": ("position", "width", "depth", "height"),
                  "IfcSlab": ("outline", "thickness")}
    MIN_SIZE_M = 0.005                                   # kleinstes zulässiges Maß: 5 mm

    def _labels_for(self, ifc_class: str) -> set[str]:
        if self.schema is None:
            raise ExecutorError("no_schema", "Dem Executor wurde kein IFC-Schema übergeben (Schema der Datei nötig)")
        from ifc2neo4j import label_chain
        return set(label_chain(self.schema, ifc_class, self._label_cache)) | {"IfcEntity"}

    @staticmethod
    def _positive(params: dict, key: str, minimum: float) -> float:
        v = params[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ExecutorError("bad_parameter", f"'{key}' muss eine endliche Zahl in Metern sein")
        if v < minimum:
            raise ExecutorError("bad_parameter", f"'{key}' muss mindestens {minimum} m sein, bekommen: {v}")
        return float(v)

    @staticmethod
    def _xy(v, key: str) -> tuple[float, float]:
        if (not isinstance(v, (list, tuple)) or len(v) != 2
                or any(isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) for c in v)):
            raise ExecutorError("bad_parameter", f"'{key}' muss [x, y] in Metern (Weltkoordinaten) sein")
        return float(v[0]), float(v[1])

    @staticmethod
    def _segments_cross(p1, p2, p3, p4) -> bool:
        def orient(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        d1, d2, d3, d4 = orient(p3, p4, p1), orient(p3, p4, p2), orient(p1, p2, p3), orient(p1, p2, p4)
        return d1 * d2 < 0 and d3 * d4 < 0

    def _check_type_fits(self, labels: set[str], cls: str, ty: int) -> None:
        if "IfcTypeObject" not in self.b.labels(ty):
            raise ExecutorError("not_a_type", f"{self._cls(ty)} ({ty}) ist kein Typobjekt (IfcTypeObject)")
        tcls = self._cls(ty)
        if not tcls.endswith("Type") or tcls[:-4] not in labels:
            raise ExecutorError("type_mismatch", f"{tcls} passt nicht zu {cls}: der Typ muss {tcls[:-4]!r} "
                                f"oder eine Unterklasse davon typisieren")

    def add_element(self, ifc_class: str, storey_guid: str, params: dict, type_guid: str | None = None,
                    z_offset: float = 0.0) -> ExecResult:
        """Legt ein neues Bauteil (Wand, Stütze oder Platte) in einem Geschoss an. Der Graph enthält keine Geometrie.
        Der neue Knoten trägt deshalb die Property geometry_spec (JSON, Maße in Modelleinheiten), und die
        Reconstruction baut daraus einen Extrusionskörper. Name, Material und Psets kommen nicht mit
        (nachträglich über set_attribute / set_property möglich)."""
        import json
        b = self.b
        if ifc_class not in self.ADD_PARAMS:
            raise ExecutorError("class_not_supported", f"add_element legt nur {', '.join(sorted(self.ADD_PARAMS))} "
                                f"an, nicht {ifc_class!r}")
        allowed = set(self.ADD_PARAMS[ifc_class])
        unknown = set(params) - allowed
        missing = allowed - set(params)
        if unknown or missing:
            raise ExecutorError("bad_parameter", f"{ifc_class} braucht genau {', '.join(self.ADD_PARAMS[ifc_class])}"
                                + (f"; unbekannt: {', '.join(sorted(unknown))}" if unknown else "")
                                + (f"; fehlt: {', '.join(sorted(missing))}" if missing else ""))
        if isinstance(z_offset, bool) or not isinstance(z_offset, (int, float)) or not math.isfinite(z_offset):
            raise ExecutorError("bad_parameter", "'z_offset' muss eine endliche Zahl in Metern sein")
        mn = self.MIN_SIZE_M

        # Maße prüfen
        if ifc_class == "IfcWall":
            s, e = self._xy(params["start"], "start"), self._xy(params["end"], "end")
            length = math.hypot(e[0] - s[0], e[1] - s[1])
            if length < 0.05:
                raise ExecutorError("bad_parameter", f"Die Wand ist zu kurz (Länge {length:.3f} m, mindestens 0,05 m)")
            thick, height = self._positive(params, "thickness", mn), self._positive(params, "height", mn)
        elif ifc_class == "IfcColumn":
            pos = self._xy(params["position"], "position")
            wid, dep, height = (self._positive(params, k, mn) for k in ("width", "depth", "height"))
        else:
            pts = params["outline"]
            if not isinstance(pts, (list, tuple)) or len(pts) < 3:
                raise ExecutorError("bad_parameter", "'outline' braucht mindestens drei Punkte [[x, y], ...]")
            poly = [self._xy(p, "outline") for p in pts]
            if poly[0] == poly[-1]:
                poly = poly[:-1]
            if len(poly) < 3 or len(set(poly)) != len(poly):
                raise ExecutorError("bad_parameter", "'outline' braucht mindestens drei verschiedene Punkte")
            area2 = sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                        for i in range(len(poly)))
            if abs(area2) / 2 < 1e-4:
                raise ExecutorError("bad_parameter", "'outline' hat keine Fläche (Punkte liegen auf einer Linie)")
            nseg = len(poly)
            for i in range(nseg):
                for j in range(i + 1, nseg):
                    if j == i + 1 or (i == 0 and j == nseg - 1):
                        continue
                    if self._segments_cross(poly[i], poly[(i + 1) % nseg], poly[j], poly[(j + 1) % nseg]):
                        raise ExecutorError("bad_parameter", "'outline' überschneidet sich selbst")
            thick = self._positive(params, "thickness", mn)

        storey = b.find_by_guid(storey_guid)
        if storey is None:
            raise ExecutorError("not_found", f"Kein Geschoss mit GlobalId {storey_guid}")
        if "IfcBuildingStorey" not in b.labels(storey):
            raise ExecutorError("not_a_storey", f"{self._cls(storey)} ({storey}) ist kein Geschoss")
        st_pl = self._single(storey, "ObjectPlacement")
        if st_pl is None:
            raise ExecutorError("no_placement", f"Geschoss {storey} hat kein Placement")
        scale = self._length_scale()
        r_st, o_st = self._world_frame(st_pl)
        labels = self._labels_for(ifc_class)
        ty = None
        if type_guid is not None:
            ty = b.find_by_guid(type_guid)
            if ty is None:
                raise ExecutorError("not_found", f"Kein Typobjekt mit GlobalId {type_guid}")
            self._check_type_fits(labels, ifc_class, ty)
        kind = self.CONTAINER_KINDS[0]
        existing = [e.src for e in b.in_edges(storey)
                    if e.type == kind[1] and self._cls(e.src) == kind[0]]
        if not existing and not b.nodes_with_label(kind[0]):
            raise ExecutorError("no_target_relation", f"Geschoss {storey} hat keine {kind[0]} und es gibt keine "
                                f"Vorlage zum Anlegen")

        # Lage: Ursprung W0 und Achsen des Bauteils in Weltkoordinaten, Profil im Bauteilsystem (Modelleinheiten)
        u = lambda m: m / scale
        z_world = o_st[2] + u(z_offset)
        x_el = (1.0, 0.0, 0.0)
        if ifc_class == "IfcWall":
            w0 = (u(s[0]), u(s[1]), z_world)
            x_el = (u(e[0] - s[0]) / u(length), u(e[1] - s[1]) / u(length), 0.0)
            spec = {"profile": "rectangle", "x_dim": u(length), "y_dim": u(thick), "center": [u(length) / 2, 0.0],
                    "depth": u(height), "direction": [0.0, 0.0, 1.0]}
        elif ifc_class == "IfcColumn":
            w0 = (u(pos[0]), u(pos[1]), z_world)
            spec = {"profile": "rectangle", "x_dim": u(wid), "y_dim": u(dep), "center": [0.0, 0.0],
                    "depth": u(height), "direction": [0.0, 0.0, 1.0]}
        else:
            w0 = (o_st[0], o_st[1], z_world)
            spec = {"profile": "polygon", "points": [[u(p[0]) - w0[0], u(p[1]) - w0[1]] for p in poly],
                    "depth": u(thick), "direction": [0.0, 0.0, -1.0]}
        loc = _apply_t(r_st, tuple(w0[i] - o_st[i] for i in range(3)))
        axis = _apply_t(r_st, (0.0, 0.0, 1.0))
        ref = _apply_t(r_st, x_el)
        rnd = lambda v: [round(float(c), 9) + 0.0 for c in v]

        # Anlegen
        res = ExecResult(operation=f"add_element {ifc_class}")
        mk = lambda cls, props, why: (lambda n: (res.created.append((n, cls, why)), n)[1])(
            b.create_node(cls, props, self._labels_for(cls)))
        pt = mk("IfcCartesianPoint", {"Coordinates": rnd(loc)}, "Location des neuen Bauteils")
        d_ax = mk("IfcDirection", {"DirectionRatios": rnd(axis)}, "Axis des neuen Bauteils")
        d_ref = mk("IfcDirection", {"DirectionRatios": rnd(ref)}, "RefDirection des neuen Bauteils")
        a2 = mk("IfcAxis2Placement3D", {}, "Achsenplacement des neuen Bauteils")
        b.add_edge(a2, "Location", pt)
        b.add_edge(a2, "Axis", d_ax)
        b.add_edge(a2, "RefDirection", d_ref)
        pl = mk("IfcLocalPlacement", {}, "Placement des neuen Bauteils")
        b.add_edge(pl, "PlacementRelTo", st_pl)
        b.add_edge(pl, "RelativePlacement", a2)
        el = b.create_node(ifc_class, {"GlobalId": self._new_guid(), "geometry_spec": json.dumps(spec)}, labels)
        res.created.append((el, ifc_class, "neues Bauteil"))
        b.add_edge(el, "ObjectPlacement", pl)
        for oh in self._targets(storey, "OwnerHistory"):
            b.add_edge(el, "OwnerHistory", oh)
        rel = self._target_relation(storey, kind, existing[0] if existing else b.nodes_with_label(kind[0])[0], res)
        b.add_edge(rel, kind[2], el, len(self._targets(rel, kind[2])))
        if ty is not None:
            self._link_type(el, ty, None, res)
        res.new_elements.append((b.props(el)["GlobalId"], ifc_class, el))
        return res

    def _link_type(self, el: int, ty: int, cur, res: ExecResult) -> None:
        """el in die IfcRelDefinesByType des Typs eintragen (Beziehung anlegen, falls der Typ keine hat)."""
        b = self.b
        target = sorted(e.src for e in b.in_edges(ty) if e.type == "RelatingType" and self._cls(e.src) == "IfcRelDefinesByType")
        if target:
            target = target[0]
        else:
            like = cur[0].src if cur else (b.nodes_with_label("IfcRelDefinesByType") or [None])[0]
            if like is None:
                raise ExecutorError("no_target_relation", f"Typ {ty} hat keine IfcRelDefinesByType und es gibt keine "
                                    f"Vorlage zum Anlegen")
            target = b.create_node("IfcRelDefinesByType", {"GlobalId": self._new_guid()}, b.labels(like))
            b.add_edge(target, "RelatingType", ty)
            for oh in self._targets(like, "OwnerHistory"):
                b.add_edge(target, "OwnerHistory", oh)
            res.created.append((target, "IfcRelDefinesByType", f"neue Beziehung für Typ {ty}"))
        b.add_edge(target, "RelatedObjects", el, len(self._targets(target, "RelatedObjects")))

    # -- add_hosted_element ------------------------------------------------------------------

    ADD_HOSTED = {"IfcWindow": 0.06, "IfcDoor": 0.04}      # Klasse -> Dicke des einfachen Quaders (m)
    QTO_NAMES = ("BaseQuantities", "Qto_WallBaseQuantities")
    LENGTH_TOLERANCE_M = 0.5

    def _wall_dimensions(self, host: int) -> tuple[float, tuple[float, float]]:
        """(Länge laut Mengen, (y_min, y_max) des Wandkörpers im Wandsystem) in Modelleinheiten.
        Die Dicke kommt aus der IfcMaterialLayerSetUsage, die Länge aus den Mengen (BaseQuantities.Length).
        Die Länge ist nur eine Näherung (bei Innenwänden kann die Geometrie länger sein)."""
        b = self.b
        length = None
        for e in b.in_edges(host):
            if e.type == "RelatedObjects" and self._cls(e.src) == "IfcRelDefinesByProperties":
                for d in self._targets(e.src, "RelatingPropertyDefinition"):
                    if "IfcElementQuantity" in b.labels(d) and b.props(d).get("Name") in self.QTO_NAMES:
                        for q in self._targets(d, "Quantities"):
                            if self._cls(q) == "IfcQuantityLength" and b.props(q).get("Name") == "Length":
                                length = b.props(q).get("LengthValue")
        usage = None
        for e in b.in_edges(host):
            if e.type == "RelatedObjects" and self._cls(e.src) == "IfcRelAssociatesMaterial":
                for mt in self._targets(e.src, "RelatingMaterial"):
                    if self._cls(mt) == "IfcMaterialLayerSetUsage":
                        usage = mt
        if length is None or usage is None:
            raise ExecutorError("missing_wall_dimensions", f"Wand {host}: Länge (BaseQuantities.Length) oder Dicke "
                                f"(IfcMaterialLayerSetUsage) fehlt, Position kann nicht geprüft werden")
        up = b.props(usage)
        if up.get("LayerSetDirection") != "AXIS2":
            raise ExecutorError("unsupported_wall", f"Schichtrichtung {up.get('LayerSetDirection')} der Wand {host} "
                                f"wird nicht unterstützt (nur AXIS2)")
        sets = self._targets(usage, "ForLayerSet")
        thick = sum(b.props(l).get("LayerThickness", 0.0) for s_ in sets for l in self._targets(s_, "MaterialLayers"))
        if thick <= 0:
            raise ExecutorError("missing_wall_dimensions", f"Wand {host}: Schichtdicke ist 0")
        off = up.get("OffsetFromReferenceLine") or 0.0
        if up.get("DirectionSense") == "NEGATIVE":
            y0, y1 = -off - thick, -off
        else:
            y0, y1 = off, off + thick
        return float(length), (y0, y1)

    def _new_placement(self, rel_to: int, loc, axis, ref, res: ExecResult, what: str) -> int:
        b = self.b
        rnd = lambda v: [round(float(c), 9) + 0.0 for c in v]
        mk = lambda cls, props, why: (lambda n: (res.created.append((n, cls, why)), n)[1])(
            b.create_node(cls, props, self._labels_for(cls)))
        pt = mk("IfcCartesianPoint", {"Coordinates": rnd(loc)}, f"Location ({what})")
        d_ax = mk("IfcDirection", {"DirectionRatios": rnd(axis)}, f"Axis ({what})")
        d_ref = mk("IfcDirection", {"DirectionRatios": rnd(ref)}, f"RefDirection ({what})")
        a2 = mk("IfcAxis2Placement3D", {}, f"Achsenplacement ({what})")
        b.add_edge(a2, "Location", pt)
        b.add_edge(a2, "Axis", d_ax)
        b.add_edge(a2, "RefDirection", d_ref)
        pl = mk("IfcLocalPlacement", {}, f"Placement ({what})")
        b.add_edge(pl, "PlacementRelTo", rel_to)
        b.add_edge(pl, "RelativePlacement", a2)
        return pl

    def add_hosted_element(self, ifc_class: str, host_guid: str, params: dict, type_guid: str | None = None) -> ExecResult:
        """Legt ein Fenster oder eine Tür in einer bestehenden Wand an: Öffnung (IfcOpeningElement) mit IfcRelVoidsElement,
        Füllung mit IfcRelFillsElement, Placement-Kette Wand -> Öffnung -> Füllung, Eintrag im Geschoss der Wand. Die Wandgeometrie
        bleibt unverändert, die Öffnung schneidet sich beim Anzeigen heraus. Maße in Metern. offset = Abstand der Mitte vom
        Ursprung der Wand-Placement entlang der Wandachse, sill_height = Höhe der Unterkante über dem Wandfuß."""
        import json
        b = self.b
        if ifc_class not in self.ADD_HOSTED:
            raise ExecutorError("class_not_supported", f"add_hosted_element legt nur {', '.join(sorted(self.ADD_HOSTED))} "
                                f"an, nicht {ifc_class!r}")
        required = {"offset", "width", "height"} | ({"sill_height"} if ifc_class == "IfcWindow" else set())
        allowed = required | {"sill_height"}
        unknown, missing = set(params) - allowed, required - set(params)
        if unknown or missing:
            raise ExecutorError("bad_parameter", f"{ifc_class} braucht {', '.join(sorted(required))} (optional "
                                f"sill_height bei Türen)" + (f"; unbekannt: {', '.join(sorted(unknown))}" if unknown else "")
                                + (f"; fehlt: {', '.join(sorted(missing))}" if missing else ""))
        width, height = self._positive(params, "width", 0.05), self._positive(params, "height", 0.05)
        sill = params.get("sill_height", 0.0)
        if isinstance(sill, bool) or not isinstance(sill, (int, float)) or not math.isfinite(sill) or sill < 0:
            raise ExecutorError("bad_parameter", "'sill_height' muss eine endliche Zahl >= 0 in Metern sein")
        off = params["offset"]
        if isinstance(off, bool) or not isinstance(off, (int, float)) or not math.isfinite(off):
            raise ExecutorError("bad_parameter", "'offset' muss eine endliche Zahl in Metern sein")

        host = b.find_by_guid(host_guid)
        if host is None:
            raise ExecutorError("not_found", f"Kein Element mit GlobalId {host_guid}")
        self._check_scope(host, "geändert")
        if "IfcWall" not in b.labels(host):
            raise ExecutorError("host_not_supported", f"{self._cls(host)} ({host}) ist keine Wand; Fenster und Türen "
                                f"werden nur in Wänden (IfcWall) angelegt")
        wall_pl = self._single(host, "ObjectPlacement")
        if wall_pl is None or self._cls(wall_pl) != "IfcLocalPlacement":
            raise ExecutorError("unsupported_placement", f"Wand {host} hat kein IfcLocalPlacement")
        containers = self._container_edges(host)
        if len(containers) != 1:
            raise ExecutorError("no_container" if not containers else "unexpected_container",
                                f"Wand {host} liegt in {len(containers)} Geschossen (erwartet 1)")
        scale = self._length_scale()
        length_u, (y0, y1) = self._wall_dimensions(host)
        length_m = length_u * scale
        tol = self.LENGTH_TOLERANCE_M
        if off - width / 2 < -tol or off + width / 2 > length_m + tol:
            raise ExecutorError("outside_wall", f"Das Bauteil (Mitte {off} m, Breite {width} m) liegt außerhalb der Wand "
                                f"(Länge laut Mengen {length_m:.2f} m, Abstand vom Wandanfang)")
        labels = self._labels_for(ifc_class)
        ty = None
        if type_guid is not None:
            ty = b.find_by_guid(type_guid)
            if ty is None:
                raise ExecutorError("not_found", f"Kein Typobjekt mit GlobalId {type_guid}")
            self._check_type_fits(labels, ifc_class, ty)
        u = lambda m: m / scale
        thick_u = y1 - y0
        leaf = min(u(self.ADD_HOSTED[ifc_class]), thick_u)

        res = ExecResult(operation=f"add_hosted_element {ifc_class}")
        z, x1 = (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
        op_pl = self._new_placement(wall_pl, (u(off), (y0 + y1) / 2, u(sill)), z, x1, res, "Öffnung")
        el_pl = self._new_placement(op_pl, (0.0, 0.0, 0.0), z, x1, res, ifc_class)
        owner = self._targets(host, "OwnerHistory")
        opening = b.create_node("IfcOpeningElement", {"GlobalId": self._new_guid(), "geometry_spec": json.dumps(
            {"profile": "rectangle", "x_dim": u(width), "y_dim": thick_u, "center": [0.0, 0.0],
             "depth": u(height), "direction": [0.0, 0.0, 1.0]})}, self._labels_for("IfcOpeningElement"))
        res.created.append((opening, "IfcOpeningElement", "Öffnung"))
        b.add_edge(opening, "ObjectPlacement", op_pl)
        el = b.create_node(ifc_class, {"GlobalId": self._new_guid(), "OverallHeight": u(height), "OverallWidth": u(width),
                                       "geometry_spec": json.dumps(
                                           {"profile": "rectangle", "x_dim": u(width), "y_dim": leaf, "center": [0.0, 0.0],
                                            "depth": u(height), "direction": [0.0, 0.0, 1.0]})}, labels)
        res.created.append((el, ifc_class, "neues Bauteil"))
        b.add_edge(el, "ObjectPlacement", el_pl)
        for n_ in (opening, el):
            for oh in owner:
                b.add_edge(n_, "OwnerHistory", oh)
        mk_rel = lambda cls, a, ta, c, tc, why: (lambda r: (res.created.append((r, cls, why)),
                                                            b.add_edge(r, ta, a), b.add_edge(r, tc, c),
                                                            [b.add_edge(r, "OwnerHistory", oh) for oh in owner], r)[-1])(
            b.create_node(cls, {"GlobalId": self._new_guid()}, self._labels_for(cls)))
        mk_rel("IfcRelVoidsElement", host, "RelatingBuildingElement", opening, "RelatedOpeningElement", "Void-Beziehung")
        mk_rel("IfcRelFillsElement", opening, "RelatingOpeningElement", el, "RelatedBuildingElement", "Fill-Beziehung")
        edge, kind, _storey = containers[0]
        b.add_edge(edge.src, kind[2], el, len(self._targets(edge.src, kind[2])))
        if ty is not None:
            self._link_type(el, ty, None, res)
        res.new_elements.append((b.props(el)["GlobalId"], ifc_class, el))
        return res


# ---------------------------------------------------------------------------------------------
# Kommandozeile (Neo4j)
# ---------------------------------------------------------------------------------------------

def schema_of_file(path: str):
    """IFC-Schema aus dem Dateikopf (FILE_SCHEMA), None wenn die Datei nicht lesbar ist."""
    import re
    try:
        with open(path, "r", errors="ignore") as fh:
            head = fh.read(20000)
        m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", head, re.I)
        if not m:
            return None
        import ifcopenshell.ifcopenshell_wrapper as w
        return w.schema_by_name(m.group(1))
    except Exception:
        return None


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--uri", default="bolt://localhost:7687")
    common.add_argument("--user", default="neo4j")
    common.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    common.add_argument("--database", default=None)
    common.add_argument("--dry-run", action="store_true", help="Plan zeigen, Transaktion zurückrollen")

    ap = argparse.ArgumentParser(description="Executor (Graph): Operation auf dem Neo4j-Graphen ausführen")
    ap.add_argument("ifc", help="IFC-Datei, deren Knoten bearbeitet werden (nur der Dateiname zählt: source_file)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("delete", parents=[common], help="delete_element")
    d.add_argument("--guid", required=True)
    d.add_argument("--cascade", choices=["dependents", "strict"], default="dependents")
    d.add_argument("--keep-opening", action="store_true", help="remove_host_opening=false")
    mv = sub.add_parser("move", parents=[common], help="move_element (Meter, Weltkoordinaten)")
    mv.add_argument("--guid", required=True)
    grp = mv.add_mutually_exclusive_group(required=True)
    grp.add_argument("--translate", nargs=3, type=float, metavar=("DX", "DY", "DZ"), help="relativ")
    grp.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"), help="absolut")
    mv.add_argument("--no-dependents", action="store_true", help="with_dependents=false")
    cl = sub.add_parser("clone", parents=[common], help="clone_element (Versatz in Metern, Weltkoordinaten)")
    cl.add_argument("--guid", required=True, help="GlobalId der Vorlage")
    cl.add_argument("--translate", nargs=3, type=float, required=True, metavar=("DX", "DY", "DZ"))
    cl.add_argument("--name", default=None)
    rc = sub.add_parser("reassign", parents=[common], help="reassign_container")
    rc.add_argument("--guid", required=True)
    rc.add_argument("--storey", required=True, help="GlobalId des Zielgeschosses")
    rc.add_argument("--keep-world-position", action="store_true", help="preserve_world_position=true")
    sp = sub.add_parser("setprop", parents=[common], help="set_property (Wert einer vorhandenen Property ändern)")
    sp.add_argument("--guid", required=True)
    sp.add_argument("--pset", required=True)
    sp.add_argument("--property", required=True)
    sp.add_argument("--value", required=True, help="true/false, Zahl oder Text (mit --as-text immer Text)")
    sp.add_argument("--as-text", action="store_true", help="Wert nicht als Zahl/Boolean deuten")
    sa = sub.add_parser("setattr", parents=[common], help="set_attribute (Name, Description, ObjectType, Tag, LongName, PredefinedType)")
    sa.add_argument("--guid", required=True)
    sa.add_argument("--attribute", required=True)
    sa.add_argument("--value", required=True)
    at = sub.add_parser("assigntype", parents=[common], help="assign_type (bestehendes Typobjekt)")
    at.add_argument("--guid", required=True)
    at.add_argument("--type", required=True, help="GlobalId des Typobjekts")
    ad = sub.add_parser("add", parents=[common], help="add_element (Wand, Stütze, Platte; Meter, Weltkoordinaten)")
    ad.add_argument("--class", dest="ifc_class", required=True, choices=["IfcWall", "IfcColumn", "IfcSlab"])
    ad.add_argument("--storey", required=True, help="GlobalId des Geschosses")
    ad.add_argument("--type", default=None, help="GlobalId eines bestehenden Typobjekts (optional)")
    ad.add_argument("--z-offset", type=float, default=0.0, help="Höhe über Geschoss in m")
    ad.add_argument("--start", nargs=2, type=float, metavar=("X", "Y"))
    ad.add_argument("--end", nargs=2, type=float, metavar=("X", "Y"))
    ad.add_argument("--position", nargs=2, type=float, metavar=("X", "Y"))
    ad.add_argument("--outline", nargs="+", metavar="X,Y", help='Punkte als "x,y" (Platte)')
    ad.add_argument("--thickness", type=float)
    ad.add_argument("--height", type=float)
    ad.add_argument("--width", type=float)
    ad.add_argument("--depth", type=float)
    ah = sub.add_parser("addhosted", parents=[common], help="add_hosted_element (Fenster/Tür in einer Wand)")
    ah.add_argument("--class", dest="ifc_class", required=True, choices=["IfcWindow", "IfcDoor"])
    ah.add_argument("--host", required=True, help="GlobalId der Wand")
    ah.add_argument("--offset", type=float, required=True, help="Abstand der Mitte vom Wandanfang in m")
    ah.add_argument("--width", type=float, required=True)
    ah.add_argument("--height", type=float, required=True)
    ah.add_argument("--sill", type=float, default=None, help="Brüstungshöhe in m (Fenster Pflicht, Tür Standard 0)")
    ah.add_argument("--type", default=None, help="GlobalId eines bestehenden Typobjekts (optional)")
    args = ap.parse_args(argv)

    from neo4j import GraphDatabase
    source = os.path.basename(args.ifc)
    if args.cmd == "delete":
        op = {"op": "delete_element", "guid": args.guid, "cascade": args.cascade,
              "remove_host_opening": not args.keep_opening}
    elif args.cmd == "clone":
        op = {"op": "clone_element", "source_guid": args.guid, "translation": args.translate}
        if args.name:
            op["name"] = args.name
    elif args.cmd == "add":
        op = {"op": "add_element", "ifc_class": args.ifc_class, "storey_guid": args.storey, "z_offset": args.z_offset}
        if args.type:
            op["type_guid"] = args.type
        for key in ("start", "end", "position", "thickness", "height", "width", "depth"):
            val = getattr(args, key)
            if val is not None:
                op[key] = val
        if args.outline:
            op["outline"] = [[float(c) for c in pt.split(",")] for pt in args.outline]
    elif args.cmd == "addhosted":
        op = {"op": "add_hosted_element", "ifc_class": args.ifc_class, "host_guid": args.host, "offset": args.offset,
              "width": args.width, "height": args.height}
        if args.sill is not None:
            op["sill_height"] = args.sill
        if args.type:
            op["type_guid"] = args.type
    elif args.cmd == "setprop":
        val = args.value
        if not args.as_text:
            import json
            try:
                val = json.loads(args.value)
            except ValueError:
                pass
        op = {"op": "set_property", "guid": args.guid, "pset_name": args.pset, "property_name": args.property,
              "value": val}
    elif args.cmd == "setattr":
        op = {"op": "set_attribute", "guid": args.guid, "attribute": args.attribute, "value": args.value}
    elif args.cmd == "assigntype":
        op = {"op": "assign_type", "guid": args.guid, "type_guid": args.type}
    elif args.cmd == "reassign":
        op = {"op": "reassign_container", "guid": args.guid, "target_storey_guid": args.storey,
              "preserve_world_position": args.keep_world_position}
    else:
        op = {"op": "move_element", "guid": args.guid, "with_dependents": not args.no_dependents}
        if args.translate is not None:
            op["translation"] = args.translate
        else:
            op["target_location"] = args.target

    with GraphDatabase.driver(args.uri, auth=(args.user, args.password),
                               notifications_min_severity="OFF") as driver:
        driver.verify_connectivity()
        with driver.session(database=args.database) as session:
            tx = session.begin_transaction()
            try:
                res = Executor(Neo4jBackend(tx, source), schema_of_file(args.ifc)).apply(op)
            except ExecutorError as exc:
                tx.rollback()
                print(f"Abgelehnt ({exc.code}): {exc.message}")
                print("Graph unverändert (Transaktion zurückgerollt).")
                return 2
            except Exception:
                tx.rollback()
                raise
            print(res.summary())
            if args.dry_run:
                tx.rollback()
                print("Dry-Run: Transaktion zurückgerollt, Graph unverändert.")
            else:
                tx.commit()
                print("Transaktion abgeschlossen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
neo4j2ifc.py — Rückweg Graph -> IFC (Option A: Kopie der Original-IFC patchen).

Idee:
  Der Graph in Neo4j ist die bearbeitbare Darstellung. Die Original-IFC-Datei bleibt die
  Quelle für alles, was im Graph fehlt (Körpergeometrie, Header, nicht abgebildete Daten).
  Das Skript
    1. bildet die Original-IFC mit denselben Funktionen wie ifc2neo4j.py in einen Snapshot ab
       (Basis, "so wurde der Graph geladen"),
    2. liest den aktuellen Graph aus Neo4j in einen zweiten Snapshot (Ziel),
    3. berechnet die Differenz (Knoten, Properties, Kanten),
    4. wendet die Differenz mit IfcOpenShell auf eine Kopie der Original-IFC an,
    5. räumt Geometrie auf, die nach dem Löschen von Elementen niemand mehr referenziert,
    6. bildet die gepatchte Datei erneut ab und vergleicht sie mit dem Zielgraph. Das Ergebnis ist ein
       Messwert für die Round-Trip-Treue (Kapitel 7.1), keine Gültigkeitsprüfung: ob die Änderung
       strukturell, semantisch oder geometrisch in Ordnung ist, entscheiden allein die Validatoren.

Konventionen für Knoten, die der Executor neu anlegt:
  * step_id: beliebige, im Graph noch nicht vergebene Zahl (die Datei vergibt eigene IDs)
  * ifc_class: IFC-Klasse als Property
  * Listen-Attribute: eine Kante je Element mit idx (bei Listen von Listen auch idx2)
  * getypte Einfachwerte: Wert plus <Attribut>_type, wie im Konverter
  * Geometrie legt dieses Skript nicht an, außer ein neuer Knoten trägt die Property geometry_from
    (step_id eines vorhandenen Bauteils): dann wird dessen Geometrie samt Darstellungsstilen kopiert.
    Die Property steht nur im Graph und wird nicht in die Datei geschrieben.

KI-Hinweis (TUHH-Handreichung "KI-Tools in Studium und Lehre"): Dieses Skript wurde mit Unterstützung
von Claude (Anthropic) erstellt. Prompts und Ausgaben gehören in Anhang F (AI usage documentation).

Nutzung:
  pip install ifcopenshell neo4j
  python neo4j2ifc.py AC20-FZK-Haus.ifc --uri neo4j+s://<id>.databases.neo4j.io --diff-only
  python neo4j2ifc.py AC20-FZK-Haus.ifc --uri neo4j+s://<id>.databases.neo4j.io --out ergebnis.ifc
  (Passwort über --password oder die Umgebungsvariable NEO4J_PASSWORD)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import ifcopenshell
import ifcopenshell.guid
import ifcopenshell.util.attribute as ifc_attr
from ifcopenshell import entity_instance, ifcopenshell_wrapper

import ifc2neo4j as g   # gleiche Auswahl- und Mappingfunktionen wie beim Hinweg

META_KEYS = {"step_id", "source_file", "ifc_class", "geometry_from", "geometry_spec"}
AGGREGATES = {"list", "set", "bag", "array"}

EdgeKey = tuple  # (src, attribut, idx, idx2)


# --------------------------------------------------------------------------- Snapshot

@dataclass
class Snapshot:
    nodes: dict[int, dict] = field(default_factory=dict)    # step_id -> Properties
    edges: dict[EdgeKey, int] = field(default_factory=dict)  # (src, Attribut, idx, idx2) -> dst
    collisions: int = 0                                      # Kanten mit gleichem Schlüssel


def snapshot_from_ifc(model: ifcopenshell.file, source: str) -> Snapshot:
    """Bildet eine IFC-Datei so ab, wie ifc2neo4j.py sie (ohne Geometrie) in den Graph lädt."""
    snap = Snapshot()
    for inst in g.collect_instances(model, include_geometry=False):
        props, edges = g.map_instance(inst, source, False)
        props.pop("source_file", None)
        snap.nodes[inst.id()] = props
        for e in edges:
            key = (e["src"], e["type"], e["props"].get("idx"), e["props"].get("idx2"))
            if key in snap.edges:
                snap.collisions += 1
            snap.edges[key] = e["dst"]
    return snap


def snapshot_from_neo4j(uri: str, user: str, password: str, database: str | None,
                        source: str) -> Snapshot:
    """Liest alle Knoten und Kanten eines Modells (source_file) aus Neo4j.
    Inverse Kanten (inverse=true) werden ignoriert."""
    from neo4j import GraphDatabase
    snap = Snapshot()
    with GraphDatabase.driver(uri, auth=(user, password), notifications_min_severity="OFF") as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            for rec in session.run(
                "MATCH (n:IfcEntity {source_file: $f}) RETURN properties(n) AS p", f=source
            ):
                p = dict(rec["p"])
                p.pop("source_file", None)
                snap.nodes[p["step_id"]] = p
            for rec in session.run(
                "MATCH (a:IfcEntity {source_file: $f})-[r]->(b:IfcEntity {source_file: $f}) "
                "WHERE r.inverse IS NULL "
                "RETURN a.step_id AS src, b.step_id AS dst, type(r) AS t, "
                "r.idx AS i1, r.idx2 AS i2",
                f=source,
            ):
                key = (rec["src"], rec["t"], rec["i1"], rec["i2"])
                if key in snap.edges:
                    snap.collisions += 1
                snap.edges[key] = rec["dst"]
    return snap


# --------------------------------------------------------------------------- Differenz

def ref_lists(snap: Snapshot) -> dict[tuple, tuple]:
    """Kanten je (Knoten, Attribut) als geordnete Liste der Ziele. Die Reihenfolge kommt aus idx/idx2,
    nicht aus deren Zahlenwerten: Lücken nach dem Löschen eines Listenelements zählen nicht als Änderung.
    Ergebnis: (src, Attribut) -> (indexiert?, Ziele) mit Ziele = (d, d, ...) oder ((d, d), (d, ...))."""
    groups: dict[tuple, list] = defaultdict(list)
    for (src, name, i1, i2), dst in snap.edges.items():
        groups[(src, name)].append((-1 if i1 is None else i1, -1 if i2 is None else i2, dst))
    out = {}
    for key, items in groups.items():
        items.sort()
        indexed = any(i1 != -1 for i1, _, _ in items)
        if any(i2 != -1 for _, i2, _ in items):                     # Liste von Listen
            rows: dict[int, list] = defaultdict(list)
            for i1, _, dst in items:
                rows[i1].append(dst)
            out[key] = (indexed, tuple(tuple(rows[k]) for k in sorted(rows)))
        else:
            out[key] = (indexed, tuple(dst for _, _, dst in items))
    return out


@dataclass
class Diff:
    added_nodes: list[int] = field(default_factory=list)
    removed_nodes: list[int] = field(default_factory=list)
    changed_nodes: dict[int, dict] = field(default_factory=dict)   # sid -> {"set": {...}, "unset": [...]}
    changed_attrs: dict[tuple, dict] = field(default_factory=dict)  # (src, Attribut) -> {"before", "after"}

    def is_empty(self) -> bool:
        return not (self.added_nodes or self.removed_nodes or self.changed_nodes or self.changed_attrs)

    def summary(self) -> str:
        return (f"Knoten +{len(self.added_nodes)} -{len(self.removed_nodes)} "
                f"geändert {len(self.changed_nodes)} | Referenzattribute geändert {len(self.changed_attrs)}")

    def to_json(self) -> dict:
        return {
            "added_nodes": self.added_nodes,
            "removed_nodes": self.removed_nodes,
            "changed_nodes": {str(k): v for k, v in self.changed_nodes.items()},
            "changed_attrs": [{"src": s, "attr": a, **v} for (s, a), v in self.changed_attrs.items()],
        }


def diff_snapshots(base: Snapshot, target: Snapshot) -> Diff:
    d = Diff()
    d.added_nodes = sorted(set(target.nodes) - set(base.nodes))
    d.removed_nodes = sorted(set(base.nodes) - set(target.nodes))
    removed = set(d.removed_nodes)

    for sid in sorted(set(base.nodes) & set(target.nodes)):
        a, b = base.nodes[sid], target.nodes[sid]
        if a == b:
            continue
        if a.get("ifc_class") != b.get("ifc_class"):
            raise ValueError(f"Knoten {sid}: Klassenwechsel {a.get('ifc_class')} -> "
                             f"{b.get('ifc_class')} wird nicht unterstützt")
        d.changed_nodes[sid] = {
            "set": {k: v for k, v in b.items() if k not in a or a[k] != v},
            "unset": [k for k in a if k not in b],
        }

    before, after = ref_lists(base), ref_lists(target)
    for key in sorted(set(before) | set(after), key=lambda k: (k[0], k[1])):
        if key[0] in removed:
            continue                      # Kanten gelöschter Knoten verschwinden mit dem Knoten
        if before.get(key) != after.get(key):
            d.changed_attrs[key] = {"before": before.get(key, (False, ()))[1],
                                    "after": after.get(key, (False, ()))[1]}
    return d


# --------------------------------------------------------------------------- Patch

@dataclass
class PatchReport:
    warnings: list[str] = field(default_factory=list)
    generated_guids: list[int] = field(default_factory=list)
    removed_geometry: int = 0
    added_ids: dict[int, int] = field(default_factory=dict)   # Graph-step_id -> Datei-ID (nur neue Knoten)


_agg_cache: dict[tuple[str, str], bool] = {}


def is_aggregate_attr(schema, ifc_class: str, name: str) -> bool | None:
    """True, wenn das Attribut laut EXPRESS-Schema eine Liste/Menge ist. None, wenn unbekannt."""
    key = (ifc_class, name)
    if key not in _agg_cache:
        result = None
        try:
            for a in schema.declaration_by_name(ifc_class).as_entity().all_attributes():
                if a.name() == name:
                    pt = ifc_attr.get_primitive_type(a)
                    result = isinstance(pt, tuple) and pt[0] in AGGREGATES
                    break
        except Exception:
            pass
        _agg_cache[key] = result
    return _agg_cache[key]


def _flatten(value):
    if isinstance(value, (tuple, list)):
        for v in value:
            yield from _flatten(v)
    else:
        yield value


def to_ifc_value(model, value: Any, type_name: str | None, template: Any, aggregate: bool | None):
    """Property aus dem Graph -> Attributwert für IfcOpenShell.
    template ist der bisherige Wert (None bei neuen Knoten) und liefert fehlende Typinfos."""
    if value is None:
        return None
    if type_name:                                     # getypter Einfachwert, z. B. IfcLabel
        return model.create_entity(type_name, value)

    is_list_attr = isinstance(template, (tuple, list)) or bool(aggregate)
    if isinstance(value, str) and is_list_attr:       # gemischte/verschachtelte Listen liegen als JSON
        value = json.loads(value)

    if isinstance(value, (list, tuple)):
        wrap = None                                   # Listen getypter Werte: Typ vom alten Wert übernehmen
        if isinstance(template, (tuple, list)):
            first = next(iter(_flatten(template)), None)
            if isinstance(first, entity_instance) and first.id() == 0:
                wrap = first.is_a()
        return tuple(_convert_item(model, v, wrap) for v in value)

    if isinstance(template, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _convert_item(model, v, wrap):
    if isinstance(v, (list, tuple)):
        return tuple(_convert_item(model, x, wrap) for x in v)
    return model.create_entity(wrap, v) if wrap else v


def geometry_closure(inst: entity_instance) -> dict[int, entity_instance]:
    """Alles, was von den Schnittattributen (Representation usw.) aus erreichbar ist."""
    seen: dict[int, entity_instance] = {}
    stack: list[entity_instance] = []
    for i in range(len(inst)):
        if inst.attribute_name(i) in g.GEOMETRY_CUT_ATTRIBUTES:
            try:
                value = inst[i]
            except Exception:
                continue
            stack.extend(ref for _, ref in g.walk_refs(value))
    while stack:
        e = stack.pop()
        if e.id() in seen:
            continue
        seen[e.id()] = e
        for j in range(len(e)):
            if e.attribute_type(j) == "DERIVED":
                continue
            try:
                value = e[j]
            except Exception:
                continue
            stack.extend(ref for _, ref in g.walk_refs(value))
    return seen


def apply_diff(model: ifcopenshell.file, base: Snapshot, target: Snapshot, diff: Diff) -> PatchReport:
    rep = PatchReport()
    schema_name = getattr(model, "schema_identifier", None) or model.schema
    schema = ifcopenshell_wrapper.schema_by_name(schema_name)
    removed = set(diff.removed_nodes)
    idmap: dict[int, entity_instance] = {}

    def inst_of(sid: int) -> entity_instance:
        if sid in removed:
            raise ValueError(f"Kante zeigt auf gelöschten Knoten {sid}")
        if sid not in idmap:
            idmap[sid] = model.by_id(sid)
        return idmap[sid]

    # 1. neue Knoten anlegen
    for sid in diff.added_nodes:
        idmap[sid] = model.create_entity(target.nodes[sid]["ifc_class"])

    # 2. Properties setzen (neue Knoten komplett, geänderte nur die Differenz)
    def set_props(sid: int, names: set[str], unset: list[str], is_new: bool):
        inst = inst_of(sid)
        props = target.nodes[sid]
        for name in sorted(names):
            template = None if is_new else getattr(inst, name, None)
            aggregate = is_aggregate_attr(schema, inst.is_a(), name)
            try:
                setattr(inst, name, to_ifc_value(model, props.get(name), props.get(name + "_type"),
                                                 template, aggregate))
            except Exception as exc:
                rep.warnings.append(f"Knoten {sid} ({inst.is_a()}).{name}: {exc}")
        for name in unset:
            if name in META_KEYS or name.endswith("_type"):
                continue
            try:
                setattr(inst, name, None)
            except Exception as exc:
                rep.warnings.append(f"Knoten {sid} ({inst.is_a()}).{name} leeren: {exc}")

    def attr_names(keys) -> set[str]:
        return {k[:-5] if k.endswith("_type") else k for k in keys if k not in META_KEYS}

    for sid in diff.added_nodes:
        set_props(sid, attr_names(target.nodes[sid]), [], True)
        inst = idmap[sid]
        if inst.is_a("IfcRoot") and not inst.GlobalId:
            inst.GlobalId = ifcopenshell.guid.new()
            rep.generated_guids.append(sid)
    for sid, ch in diff.changed_nodes.items():
        set_props(sid, attr_names(ch["set"]), ch["unset"], False)

    # 3. Referenzattribute neu aufbauen (ein Attribut komplett aus den Kanten des Zielgraphen)
    target_refs = ref_lists(target)
    for (src, t) in sorted(diff.changed_attrs, key=lambda k: (k[0], k[1])):
        inst = inst_of(src)
        indexed, refs = target_refs.get((src, t), (False, ()))
        try:
            current = None if src in diff.added_nodes else getattr(inst, t)
            if isinstance(current, (tuple, list)) and any(
                    not isinstance(x, entity_instance) or x.id() == 0 for x in _flatten(current)):
                rep.warnings.append(f"Knoten {src}.{t}: gemischte Liste aus Referenzen und Werten "
                                    f"wird nicht gepatcht")
                continue
            aggregate = is_aggregate_attr(schema, inst.is_a(), t)
            if aggregate is None:
                aggregate = isinstance(current, (tuple, list)) or indexed
            if not refs:
                value = () if aggregate else None
            elif any(isinstance(r, tuple) for r in refs):            # Liste von Listen
                value = tuple(tuple(inst_of(d) for d in row) for row in refs)
            elif aggregate:
                value = tuple(inst_of(d) for d in refs)
            else:
                value = inst_of(refs[0])
            setattr(inst, t, value)
        except Exception as exc:
            rep.warnings.append(f"Knoten {src}.{t}: {exc}")

    # 3b. Geometrie für Klone kopieren (Property geometry_from im Graph)
    for sid in diff.added_nodes:
        src_id = target.nodes[sid].get("geometry_from")
        if src_id is None:
            continue
        try:
            src = model.by_id(int(src_id))
            if getattr(src, "Representation", None) is not None:
                idmap[sid].Representation = copy_geometry(model, src.Representation)
        except Exception as exc:
            rep.warnings.append(f"Knoten {sid}: Geometrie von {src_id} nicht kopiert: {exc}")

    # 3c. Geometrie für neue Bauteile aus geometry_spec bauen (add_element)
    for sid in diff.added_nodes:
        spec = target.nodes[sid].get("geometry_spec")
        if spec is None:
            continue
        try:
            idmap[sid].Representation = build_extrusion(model, json.loads(spec))
        except Exception as exc:
            rep.warnings.append(f"Knoten {sid}: Geometrie aus geometry_spec nicht gebaut: {exc}")

    # 4. gelöschte Knoten entfernen, danach verwaiste Geometrie aufräumen
    candidates: dict[int, entity_instance] = {}
    for sid in diff.removed_nodes:
        inst = model.by_id(sid)
        candidates.update(geometry_closure(inst))
        model.remove(inst)
    gone: set[int] = set()
    changed = True
    while changed:
        changed = False
        for eid, e in list(candidates.items()):
            if eid in gone or eid in base.nodes:   # Knoten des Graphen sind Sache des Executors
                continue
            if len(model.get_inverse(e)) == 0:
                model.remove(e)
                gone.add(eid)
                changed = True
    rep.removed_geometry = len(gone)

    rep.added_ids = {sid: idmap[sid].id() for sid in diff.added_nodes}
    return rep


# --------------------------------------------------------------------------- Geometrie kopieren

# Diese Klassen bleiben bei der Kopie geteilt (Kontext, Typgeometrie, Darstellungsstile)
GEOMETRY_SHARED = ("IfcGeometricRepresentationContext", "IfcRepresentationMap", "IfcPresentationStyle",
                   "IfcPresentationStyleAssignment", "IfcMaterial")


def copy_geometry(model: ifcopenshell.file, rep: entity_instance) -> entity_instance:
    """Tiefe Kopie einer Produktdarstellung. Kontexte, Typgeometrie (IfcRepresentationMap) und Stile bleiben geteilt,
    Farbzuweisungen (IfcStyledItem) werden für die kopierten Elemente neu angelegt."""
    memo: dict[int, entity_instance] = {}

    def conv(v):
        if isinstance(v, entity_instance):
            if v.id() == 0:                                   # getypter Einfachwert
                return model.create_entity(v.is_a(), v.wrappedValue)
            return cp(v)
        if isinstance(v, (tuple, list)):
            return tuple(conv(x) for x in v)
        return v

    def cp(e: entity_instance) -> entity_instance:
        if e.id() in memo:
            return memo[e.id()]
        if any(e.is_a(c) for c in GEOMETRY_SHARED):
            return e
        new = model.create_entity(e.is_a())
        memo[e.id()] = new
        for i in range(len(e)):
            if e.attribute_type(i) == "DERIVED":
                continue
            try:
                v = e[i]
            except Exception:
                continue
            if v is not None:
                new[i] = conv(v)
        for inv in model.get_inverse(e):
            if inv.is_a("IfcStyledItem"):
                cp(inv)
        return new

    return cp(rep)


def body_context(model: ifcopenshell.file) -> entity_instance:
    """Kontext für die Körpergeometrie: Unterkontext 'Body' (Model), sonst der Model-Kontext."""
    subs = [c for c in model.by_type("IfcGeometricRepresentationSubContext")
            if c.ContextIdentifier == "Body" and c.ContextType == "Model"]
    if subs:
        return subs[0]
    ctxs = [c for c in model.by_type("IfcGeometricRepresentationContext")
            if not c.is_a("IfcGeometricRepresentationSubContext") and c.ContextType == "Model"]
    if not ctxs:
        raise ValueError("kein Model-Kontext im IFC")
    return ctxs[0]


def build_extrusion(model: ifcopenshell.file, spec: dict) -> entity_instance:
    """IfcProductDefinitionShape mit einem Extrusionskörper (SweptSolid) aus geometry_spec. Maße in Modelleinheiten,
    Profil im Koordinatensystem des Bauteils."""
    if spec["profile"] == "rectangle":
        cx, cy = spec["center"]
        profile = model.create_entity(
            "IfcRectangleProfileDef", ProfileType="AREA",
            Position=model.create_entity("IfcAxis2Placement2D",
                                         Location=model.create_entity("IfcCartesianPoint", (float(cx), float(cy)))),
            XDim=float(spec["x_dim"]), YDim=float(spec["y_dim"]))
    elif spec["profile"] == "polygon":
        pts = [model.create_entity("IfcCartesianPoint", (float(x), float(y))) for x, y in spec["points"]]
        pts.append(pts[0])
        profile = model.create_entity("IfcArbitraryClosedProfileDef", ProfileType="AREA",
                                      OuterCurve=model.create_entity("IfcPolyline", Points=pts))
    else:
        raise ValueError(f"unbekanntes Profil {spec['profile']!r}")
    solid = model.create_entity(
        "IfcExtrudedAreaSolid", SweptArea=profile,
        Position=model.create_entity("IfcAxis2Placement3D",
                                     Location=model.create_entity("IfcCartesianPoint", (0.0, 0.0, 0.0))),
        ExtrudedDirection=model.create_entity("IfcDirection", tuple(float(c) for c in spec["direction"])),
        Depth=float(spec["depth"]))
    shape = model.create_entity("IfcShapeRepresentation", ContextOfItems=body_context(model),
                                RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=(solid,))
    return model.create_entity("IfcProductDefinitionShape", Representations=(shape,))


# --------------------------------------------------------------------------- Prüfung

def translate(target: Snapshot, added_ids: dict[int, int]) -> Snapshot:
    """Zielgraph in den ID-Raum der gepatchten Datei übersetzen (nur neue Knoten bekommen neue IDs)."""
    f = lambda s: added_ids.get(s, s)
    out = Snapshot()
    for sid, props in target.nodes.items():
        out.nodes[f(sid)] = {**{k: v for k, v in props.items() if k not in ("geometry_from", "geometry_spec")}, "step_id": f(sid)}
    for (s, t, i1, i2), d in target.edges.items():
        out.edges[(f(s), t, i1, i2)] = f(d)
    return out


def verify_roundtrip(patched: ifcopenshell.file, target: Snapshot, rep: PatchReport,
                     source: str = "") -> Diff:
    """Leeres Diff = die gepatchte Datei bildet sich exakt auf den Zielgraph ab."""
    return diff_snapshots(snapshot_from_ifc(patched, source), translate(target, rep.added_ids))


# --------------------------------------------------------------------------- Hauptprogramm

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Neo4j-Graph -> IFC (Original-IFC patchen)")
    ap.add_argument("ifc", help="Original-IFC (wird nicht verändert)")
    ap.add_argument("--out", help="Ausgabedatei (Standard: <name>_patched.ifc neben dem Original)")
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    ap.add_argument("--database", default=None)
    ap.add_argument("--diff-only", action="store_true", help="nur Differenz berechnen und ausgeben")
    ap.add_argument("--report", help="Differenz als JSON speichern")
    args = ap.parse_args(argv)

    t0 = time.time()
    source = os.path.basename(args.ifc)
    model = ifcopenshell.open(args.ifc)
    base = snapshot_from_ifc(model, source)
    print(f"[{source}] Basis: {len(base.nodes)} Knoten, {len(base.edges)} Kanten")

    target = snapshot_from_neo4j(args.uri, args.user, args.password, args.database, source)
    print(f"[{source}] Neo4j: {len(target.nodes)} Knoten, {len(target.edges)} Kanten")
    if not target.nodes:
        print("Der Graph enthält keine Knoten für diese Datei (source_file stimmt?).")
        return 1
    for name, s in (("Basis", base), ("Neo4j", target)):
        if s.collisions:
            print(f"WARNUNG: {s.collisions} Kanten mit gleichem Schlüssel in {name} (Verschachtelung > 2)")

    diff = diff_snapshots(base, target)
    print(f"Differenz: {diff.summary()}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(diff.to_json(), fh, indent=1)
    if args.diff_only:
        return 0

    rep = apply_diff(model, base, target, diff)
    for w in rep.warnings:
        print("WARNUNG:", w)
    if rep.generated_guids:
        print(f"{len(rep.generated_guids)} neue GlobalIds erzeugt (Executor sollte sie selbst setzen)")
    print(f"{rep.removed_geometry} verwaiste Geometrie-Instanzen entfernt")

    out = args.out or re.sub(r"\.ifc$", "", args.ifc, flags=re.I) + "_patched.ifc"
    model.write(out)
    print(f"Geschrieben: {out}")

    rest = verify_roundtrip(ifcopenshell.open(out), target, rep, source)
    if rest.is_empty():
        print("Round-Trip-Treue: Datei und Graph stimmen überein")
    else:
        print(f"Round-Trip-Treue: Datei und Graph weichen ab ({rest.summary()}). "
              f"Das ist ein Messwert, keine Bewertung der Änderung.")
    print(f"Fertig in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

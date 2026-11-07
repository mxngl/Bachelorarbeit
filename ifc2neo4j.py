#!/usr/bin/env python3
"""
ifc2neo4j.py — Konvertiert eine IFC-Datei (STEP Physical File) in einen
Labeled Property Graph in Neo4j.

Abbildungsregeln (nach Zhu et al. 2023, "IFC-graph", Automation in Construction 148):
  * jede IFC-Instanz            -> ein Knoten; Labels = IFC-Klasse + alle Oberklassen
                                   (+ gemeinsames Label :IfcEntity für Index und Lookup)
  * intrinsische Attribute      -> Knoteneigenschaften (Skalare, Listen von Skalaren)
  * extrinsische Attribute      -> gerichtete Kanten; Kantentyp = Attributname
                                   (Aggregate: eine Kante je Element, mit idx-Eigenschaft,
                                   damit die Reihenfolge für den Rückweg erhalten bleibt)
  * objektifizierte Beziehungen -> bleiben eigene Knoten (IfcRel*)
  * inverse Attribute           -> optional als zusätzliche Kanten (--inverse),
                                   markiert mit inverse=true
  * Geometrie                   -> standardmäßig ausgelassen: der Teilgraph hinter
                                   Representation/RepresentationMaps wird nicht verfolgt
                                   (vgl. LoD-Stufen bei Zhu et al. 2026). Die Platzierungs-
                                   kette (ObjectPlacement -> IfcLocalPlacement ->
                                   IfcAxis2Placement3D -> IfcCartesianPoint) bleibt erhalten,
                                   weil sie für das Neuplatzieren gebraucht wird.
                                   --geometry nimmt alle Instanzen mit.

Jeder Knoten trägt zusätzlich:
  step_id      Instanznummer aus der SPF-Datei (#42 -> 42)  — Schlüssel für Kanten und Rückweg
  source_file  Dateiname                                     — erlaubt mehrere Modelle in einer DB
  ifc_class    IFC-Klasse als Property (zusätzlich zum Label, bequem für RETURN)

Getypte Einfachwerte (IfcLabel, IfcReal, ...) z. B. in IfcPropertySingleValue.NominalValue
werden als Property gespeichert, der Typname daneben als <Attribut>_type — nötig,
um beim Rückschreiben den richtigen IFC-Typ wieder zu erzeugen.

Nutzung:
  pip install ifcopenshell neo4j
  python ifc2neo4j.py modell.ifc --dry-run                       # nur analysieren
  python ifc2neo4j.py modell.ifc --password geheim --wipe        # in Neo4j laden
  python ifc2neo4j.py modell.ifc --password geheim --inverse --geometry
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict, deque
from typing import Any, Iterable, Iterator

import ifcopenshell
from ifcopenshell import entity_instance

# Attribute, hinter denen der Geometrie-Teilgraph beginnt.
# Ohne --geometry werden diese Referenzen nicht verfolgt (und nicht als Kante angelegt).
GEOMETRY_CUT_ATTRIBUTES = {
    "Representation",       # IfcProduct -> IfcProductDefinitionShape -> ... Körpergeometrie
    "RepresentationMaps",   # IfcTypeProduct -> IfcRepresentationMap
    "ConnectionGeometry",   # IfcRelSpaceBoundary -> Flächengeometrie
    "AxisCurve",            # IfcGridAxis -> Kurve
}

COMMON_LABEL = "IfcEntity"


# --------------------------------------------------------------------------- Hilfsfunktionen

def is_reference(value: Any) -> bool:
    """True, wenn im Wert (auch verschachtelt) mindestens eine echte Instanzreferenz steckt.
    Getypte Einfachwerte (IfcLabel('x') o. ä.) haben id() == 0 und zählen nicht."""
    if isinstance(value, entity_instance):
        return value.id() != 0
    if isinstance(value, (tuple, list)):
        return any(is_reference(v) for v in value)
    return False


def walk_refs(value: Any, path: tuple = ()) -> Iterator[tuple[tuple, entity_instance]]:
    """Liefert (Indexpfad, Instanz) für jede echte Referenz im Wert."""
    if isinstance(value, entity_instance):
        if value.id() != 0:
            yield path, value
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from walk_refs(v, path + (i,))


def to_property(value: Any) -> Any:
    """Wandelt einen intrinsischen Attributwert in einen Neo4j-tauglichen Wert um.
    Neo4j erlaubt nur Skalare und homogene Listen von Skalaren; alles andere wird JSON."""
    if value is None:
        return None
    if isinstance(value, entity_instance):
        if value.id() == 0:                       # getypter Einfachwert
            return to_property(value.wrappedValue)
        return None                               # Referenz -> wird Kante, keine Property
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        items = [to_property(v) for v in value]
        items = [i for i in items if i is not None]
        if not items:
            return None
        if all(type(i) is bool for i in items):
            return items
        if all(type(i) is int for i in items):
            return items
        if all(isinstance(i, (int, float)) and not isinstance(i, bool) for i in items):
            return [float(i) for i in items]
        if all(isinstance(i, str) for i in items):
            return items
        return json.dumps(items)                  # verschachtelt oder gemischt
    return str(value)


def label_chain(schema, ifc_class: str, cache: dict) -> list[str]:
    """IFC-Klasse plus alle Oberklassen aus dem EXPRESS-Schema, z. B.
    IfcWall -> [IfcWall, IfcBuildingElement, IfcElement, IfcProduct, IfcObject,
    IfcObjectDefinition, IfcRoot]. Damit funktioniert MATCH (e:IfcElement) für alle Bauteile."""
    if ifc_class in cache:
        return cache[ifc_class]
    labels = [ifc_class]
    try:
        decl = schema.declaration_by_name(ifc_class)
        sup = decl.supertype()
        while sup is not None:
            labels.append(sup.name())
            sup = sup.supertype()
    except Exception:
        pass
    cache[ifc_class] = labels
    return labels


# --------------------------------------------------------------------------- Auswahl der Instanzen

def collect_instances(model: ifcopenshell.file, include_geometry: bool) -> list[entity_instance]:
    """Ohne Geometrie: Breitensuche von allen IfcRoot-Instanzen entlang der Vorwärtsreferenzen,
    ohne die Schnittattribute zu verfolgen. Damit bleiben Platzierung, Property-Sets,
    Materialien, Einheiten und OwnerHistory erhalten, die Körpergeometrie fällt weg."""
    if include_geometry:
        return list(model)

    seen: set[int] = set()
    queue: deque[entity_instance] = deque()
    for inst in model.by_type("IfcRoot"):
        seen.add(inst.id())
        queue.append(inst)

    while queue:
        inst = queue.popleft()
        for i in range(len(inst)):
            if inst.attribute_name(i) in GEOMETRY_CUT_ATTRIBUTES:
                continue
            if inst.attribute_type(i) == "DERIVED":
                continue
            try:
                value = inst[i]
            except Exception:
                continue
            for _, ref in walk_refs(value):
                if ref.id() not in seen:
                    seen.add(ref.id())
                    queue.append(ref)

    return [model.by_id(i) for i in sorted(seen)]


# --------------------------------------------------------------------------- Mapping Instanz -> Knoten + Kanten

def map_instance(inst: entity_instance, source: str, include_geometry: bool):
    """Gibt (props, edges) zurück. props = Knoteneigenschaften, edges = Liste von Kantenrecords."""
    props: dict[str, Any] = {
        "step_id": inst.id(),
        "source_file": source,
        "ifc_class": inst.is_a(),
    }
    edges: list[dict] = []

    for i in range(len(inst)):
        name = inst.attribute_name(i)
        if inst.attribute_type(i) == "DERIVED":
            continue
        try:
            value = inst[i]
        except Exception:
            continue
        if value is None:
            continue

        if is_reference(value):
            if not include_geometry and name in GEOMETRY_CUT_ATTRIBUTES:
                continue
            for path, ref in walk_refs(value):
                eprops: dict[str, Any] = {}
                if len(path) >= 1:
                    eprops["idx"] = path[0]
                if len(path) >= 2:
                    eprops["idx2"] = path[1]
                edges.append({
                    "file": source, "src": inst.id(), "dst": ref.id(),
                    "type": name, "props": eprops,
                })
        else:
            p = to_property(value)
            if p is None:
                continue
            props[name] = p
            if isinstance(value, entity_instance) and value.id() == 0:
                props[name + "_type"] = value.is_a()

    return props, edges


def inverse_edges(inst: entity_instance, schema, source: str) -> Iterator[dict]:
    """Optional: inverse Attribute (z. B. IfcWall.HasOpenings) als eigene Kanten.
    Nur Navigationshilfe; beim Rückschreiben ignorieren (inverse=true)."""
    try:
        decl = schema.declaration_by_name(inst.is_a())
        inverses = decl.all_inverse_attributes()
    except Exception:
        return
    for inv in inverses:
        name = inv.name()
        try:
            value = getattr(inst, name)
        except Exception:
            continue
        for path, ref in walk_refs(value):
            yield {
                "file": source, "src": inst.id(), "dst": ref.id(),
                "type": name, "props": {"inverse": True, **({"idx": path[0]} if path else {})},
            }


# --------------------------------------------------------------------------- Neo4j-Schreiber

class Neo4jWriter:
    def __init__(self, uri: str, user: str, password: str, database: str | None, batch_size: int):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self.database = database
        self.batch_size = batch_size

    def close(self):
        self.driver.close()

    def _run(self, query: str, **params):
        with self.driver.session(database=self.database) as session:
            return session.execute_write(lambda tx: tx.run(query, **params).consume())

    def wipe(self):
        """Löscht die Datenbank in Häppchen (versionsunabhängig, ohne CALL IN TRANSACTIONS)."""
        while True:
            with self.driver.session(database=self.database) as session:
                deleted = session.execute_write(
                    lambda tx: tx.run(
                        "MATCH (n) WITH n LIMIT $n DETACH DELETE n RETURN count(n) AS c",
                        n=20000,
                    ).single()["c"]
                )
            if deleted == 0:
                break

    def ensure_indexes(self):
        self._run(
            f"CREATE INDEX ifc_step_id IF NOT EXISTS "
            f"FOR (n:{COMMON_LABEL}) ON (n.source_file, n.step_id)"
        )
        self._run("CREATE INDEX ifc_global_id IF NOT EXISTS FOR (n:IfcRoot) ON (n.GlobalId)")

    def write_nodes(self, rows_by_labels: dict[tuple[str, ...], list[dict]]) -> int:
        """Labels lassen sich in Cypher nicht parametrisieren, daher ein UNWIND je Labelkombination."""
        written = 0
        for labels, rows in rows_by_labels.items():
            label_str = ":".join(f"`{l}`" for l in labels)
            query = f"UNWIND $rows AS row CREATE (n:{label_str}) SET n = row"
            for start in range(0, len(rows), self.batch_size):
                chunk = rows[start:start + self.batch_size]
                self._run(query, rows=chunk)
                written += len(chunk)
        return written

    def write_edges(self, rows_by_type: dict[str, list[dict]]) -> int:
        written = 0
        for rtype, rows in rows_by_type.items():
            query = (
                f"UNWIND $rows AS row "
                f"MATCH (a:{COMMON_LABEL} {{source_file: row.file, step_id: row.src}}) "
                f"MATCH (b:{COMMON_LABEL} {{source_file: row.file, step_id: row.dst}}) "
                f"CREATE (a)-[r:`{rtype}`]->(b) SET r = row.props"
            )
            for start in range(0, len(rows), self.batch_size):
                chunk = rows[start:start + self.batch_size]
                self._run(query, rows=chunk)
                written += len(chunk)
        return written


# --------------------------------------------------------------------------- Hauptprogramm

def convert(path: str, args) -> tuple[dict, dict]:
    t0 = time.time()
    model = ifcopenshell.open(path)
    schema_name = getattr(model, "schema_identifier", None) or model.schema
    schema = ifcopenshell.ifcopenshell_wrapper.schema_by_name(schema_name)
    source = os.path.basename(path)
    print(f"[{source}] Schema {schema_name}, {len(list(model))} Instanzen gesamt")

    instances = collect_instances(model, args.geometry)
    included_ids = {inst.id() for inst in instances}
    print(f"[{source}] {len(instances)} Instanzen ausgewählt "
          f"({'mit' if args.geometry else 'ohne'} Geometrie)")

    label_cache: dict = {}
    rows_by_labels: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    rows_by_type: dict[str, list[dict]] = defaultdict(list)
    dropped_edges = 0

    for n, inst in enumerate(instances, 1):
        props, edges = map_instance(inst, source, args.geometry)
        labels = tuple(label_chain(schema, inst.is_a(), label_cache) + [COMMON_LABEL])
        rows_by_labels[labels].append(props)
        if args.inverse:
            edges.extend(inverse_edges(inst, schema, source))
        for e in edges:
            if e["dst"] in included_ids:
                rows_by_type[e["type"]].append(e)
            else:
                dropped_edges += 1          # Ziel liegt im ausgelassenen Geometrie-Teilgraph
        if n % 20000 == 0:
            print(f"[{source}]   {n} Instanzen gemappt ...")

    n_nodes = sum(len(r) for r in rows_by_labels.values())
    n_edges = sum(len(r) for r in rows_by_type.values())
    print(f"[{source}] Mapping fertig: {n_nodes} Knoten, {n_edges} Kanten, "
          f"{len(rows_by_labels)} Labelkombinationen, {len(rows_by_type)} Kantentypen, "
          f"{dropped_edges} Kanten mit Ziel außerhalb der Auswahl verworfen, {time.time() - t0:.1f} s")
    return rows_by_labels, rows_by_type


def print_summary(rows_by_labels, rows_by_type, top: int = 15):
    classes = Counter()
    for labels, rows in rows_by_labels.items():
        classes[labels[0]] += len(rows)
    print("\nHäufigste Klassen:")
    for cls, c in classes.most_common(top):
        print(f"  {c:8d}  {cls}")
    print("\nHäufigste Kantentypen:")
    for rtype, rows in sorted(rows_by_type.items(), key=lambda kv: -len(kv[1]))[:top]:
        print(f"  {len(rows):8d}  {rtype}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="IFC -> Neo4j Labeled Property Graph")
    ap.add_argument("ifc", nargs="+", help="IFC-Datei(en)")
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    ap.add_argument("--database", default=None, help="Neo4j-Datenbankname (Standard: default)")
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--wipe", action="store_true", help="Datenbank vorher leeren")
    ap.add_argument("--geometry", action="store_true", help="Geometrie-Teilgraph mitnehmen")
    ap.add_argument("--inverse", action="store_true", help="inverse Attribute als Kanten anlegen")
    ap.add_argument("--dry-run", action="store_true", help="nur mappen und zählen, nichts schreiben")
    args = ap.parse_args(argv)

    results = [convert(p, args) for p in args.ifc]

    if args.dry_run:
        for rows_by_labels, rows_by_type in results:
            print_summary(rows_by_labels, rows_by_type)
        return 0

    writer = Neo4jWriter(args.uri, args.user, args.password, args.database, args.batch_size)
    try:
        if args.wipe:
            print("Leere Datenbank ...")
            writer.wipe()
        writer.ensure_indexes()
        for (rows_by_labels, rows_by_type) in results:
            t0 = time.time()
            n = writer.write_nodes(rows_by_labels)
            print(f"{n} Knoten geschrieben ({time.time() - t0:.1f} s)")
            t0 = time.time()
            e = writer.write_edges(rows_by_type)
            print(f"{e} Kanten geschrieben ({time.time() - t0:.1f} s)")
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
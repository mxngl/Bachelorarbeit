#!/usr/bin/env python3
"""
llm_stations.py — Locator und Generator als Sprachmodell-Stationen für orchestrator.py (OpenAI-kompatible API, z. B. vLLM).

  LlmLocator   (LocatorBase)    nur Lesezugriff auf den Graphen: list_storeys, find_elements, describe_element
  LlmGenerator (GeneratorBase)  genau eine Operation aus dem Vokabular des Executors, oder abort

Beide Stationen nutzen Constrained Decoding (JSON-Schema im response_format). Locator und Generator können
unterschiedliche Modelle verwenden (zwei LlmClient-Instanzen).

Der Locator liest über GraphView (Snapshot-basiert, wie die Validatoren). Er bekommt vom Orchestrator den Snapshot
vor der Aufgabe; Schreibzugriff gibt es nicht. Adressiert wird nur über GlobalId.

Ergebnis des Locators (`located`, geht unverändert an den Generator):
  {"status": "ok", "target_guids": [...], "targets": [<describe_element-Ergebnisse>], "reason": "..."}
  {"status": "abort", "abort": {"reason": "ambiguous|not_found|locator_failed", "detail": "...", "candidates": [...]}}

Bei Locator-Abbruch gibt der Generator sofort abort zurück. Für ein getrenntes Protokoll (Kap. 4.1) liegt
orchestrator_locator_abort.patch bei (4 Zeilen in Orchestrator.run).

KI-Hinweis (TUHH-Handreichung "KI-Tools in Studium und Lehre"): mit Unterstützung von Claude (Anthropic)
erstellt. Prompts und Ausgaben gehören in Anhang F (AI usage documentation).

Nutzung (Spark, vLLM läuft):
  pip install openai
  export LLM_BASE_URL=http://localhost:8000/v1
  python llm_stations.py test-models/AC20-FZK-Haus.ifc --tasks tasks.json --model <Name> --no-think --log runs.jsonl
  (tasks.json: [{"id": "t1", "task": "Delete the door ..."}, ...]; --neo4j nutzt die Datenbank statt des Speichers)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import executor as x
import ifc2neo4j as g
import neo4j2ifc as r
import orchestrator as o

GUID_PATTERN = r"^[0-9A-Za-z_$]{22}$"
MAX_RESULTS = 30          # find_elements: höchstens so viele Treffer zurückgeben
MAX_RESULT_CHARS = 6000   # Antwort einer Leseoperation an das Modell


# =========================================================================== LLM-Client

class LlmClient:
    """Dünner Client für eine OpenAI-kompatible API mit JSON-Schema-Ausgabe und Protokoll (JSONL)."""

    def __init__(self, base_url: str, api_key: str, model: str, no_think: bool = False,
                 log_path: str | None = None, temperature: float = 0.0, seed: int | None = 0):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.no_think, self.log_path = model, no_think, log_path
        self.temperature, self.seed = temperature, seed

    def chat_json(self, station: str, messages: list[dict], schema: dict, max_tokens: int = 1024) -> dict:
        extra = {"chat_template_kwargs": {"enable_thinking": False}} if self.no_think else {}
        kw = {"seed": self.seed} if self.seed is not None else {}
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": station, "schema": schema}},
            extra_body=extra or None, **kw)
        dt = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        u = resp.usage
        self._log({"station": station, "model": self.model, "latency_s": round(dt, 3),
                   "prompt_tokens": getattr(u, "prompt_tokens", None),
                   "completion_tokens": getattr(u, "completion_tokens", None),
                   "finish_reason": resp.choices[0].finish_reason, "messages": messages, "output": text})
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                raise
            return json.loads(m.group(0))

    def _log(self, rec: dict) -> None:
        if not self.log_path:
            return
        rec["ts"] = datetime.now().isoformat(timespec="seconds")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# =========================================================================== Lesezugriff (nur Snapshot)

class GraphView:
    """Leseoperationen des Locators auf einem neo4j2ifc.Snapshot. Ändert nichts."""

    def __init__(self, snap: r.Snapshot, schema):
        self.snap, self.schema, self.nodes = snap, schema, snap.nodes
        self._chain: dict = {}
        self.out_: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self.in_: dict[int, list] = defaultdict(list)
        for (src, attr, i1, i2), dst in snap.edges.items():
            self.out_[src][attr].append(((-1 if i1 is None else i1, -1 if i2 is None else i2), dst))
            self.in_[dst].append((src, attr))
        for attrs in self.out_.values():
            for lst in attrs.values():
                lst.sort()
        self.by_guid = {p["GlobalId"]: sid for sid, p in self.nodes.items() if p.get("GlobalId")}
        labels = lambda cls: set(g.label_chain(schema, cls, self._chain)) | {"IfcEntity"}
        self._ex = x.Executor(x.MemoryBackend(snap, labels), schema)

    # -- Grundfunktionen
    def cls(self, sid: int) -> str:
        return self.nodes[sid]["ifc_class"]

    def isa(self, sid: int, name: str) -> bool:
        return name in g.label_chain(self.schema, self.cls(sid), self._chain)

    def out(self, sid: int, attr: str) -> list[int]:
        return [d for _, d in self.out_[sid][attr]] if sid in self.out_ else []

    def rels_in(self, sid: int, attr: str, rel_class: str) -> list[int]:
        return [s for s, a in self.in_.get(sid, []) if a == attr and self.cls(s) == rel_class]

    def guid(self, sid: int) -> str | None:
        return self.nodes[sid].get("GlobalId")

    def is_element(self, sid: int) -> bool:
        return self.isa(sid, "IfcElement") and not self.isa(sid, "IfcOpeningElement")

    # -- Beziehungen
    def storey_of(self, sid: int) -> int | None:
        for rel in self.rels_in(sid, "RelatedElements", "IfcRelContainedInSpatialStructure"):
            t = self.out(rel, "RelatingStructure")
            if t:
                return t[0]
        return None

    def openings_of(self, wall: int) -> list[int]:
        return [d for rel in self.rels_in(wall, "RelatingBuildingElement", "IfcRelVoidsElement")
                for d in self.out(rel, "RelatedOpeningElement")]

    def fillings_of(self, opening: int) -> list[int]:
        return [d for rel in self.rels_in(opening, "RelatingOpeningElement", "IfcRelFillsElement")
                for d in self.out(rel, "RelatedBuildingElement")]

    def host_of(self, filling: int) -> int | None:
        for rel in self.rels_in(filling, "RelatedBuildingElement", "IfcRelFillsElement"):
            for op in self.out(rel, "RelatingOpeningElement"):
                for rel2 in self.rels_in(op, "RelatedOpeningElement", "IfcRelVoidsElement"):
                    t = self.out(rel2, "RelatingBuildingElement")
                    if t:
                        return t[0]
        return None

    def hosted_by(self, wall: int) -> list[int]:
        return [f for op in self.openings_of(wall) for f in self.fillings_of(op)]

    def type_of(self, sid: int) -> int | None:
        for rel in self.rels_in(sid, "RelatedObjects", "IfcRelDefinesByType"):
            t = self.out(rel, "RelatingType")
            if t:
                return t[0]
        return None

    # -- Ausgabeformate
    def brief(self, sid: int) -> dict:
        st = self.storey_of(sid)
        return {"guid": self.guid(sid), "class": self.cls(sid), "name": self.nodes[sid].get("Name"),
                "storey": self.nodes[st].get("Name") if st is not None else None}

    def _scale(self) -> float | None:
        try:
            return self._ex._length_scale()
        except Exception:
            return None

    def list_storeys(self, **_) -> list[dict]:
        sc = self._scale() or 1.0
        out = []
        for sid in self.nodes:
            if self.isa(sid, "IfcBuildingStorey"):
                el = self.nodes[sid].get("Elevation")
                out.append({"guid": self.guid(sid), "name": self.nodes[sid].get("Name"),
                            "elevation_m": round(el * sc, 3) if isinstance(el, (int, float)) else None})
        return sorted(out, key=lambda d: (d["elevation_m"] is None, d["elevation_m"]))

    def find_elements(self, **a) -> dict:
        unknown = set(a) - {"class", "storey_guid", "name_contains", "hosted_by"}
        if unknown:
            return {"error": f"unknown filter(s): {sorted(unknown)}"}
        cls = a.get("class")
        if cls:
            try:
                self.schema.declaration_by_name(cls)
            except Exception:
                return {"error": f"unknown IFC class {cls!r}"}
        storey = None
        if a.get("storey_guid"):
            storey = self.by_guid.get(a["storey_guid"])
            if storey is None:
                return {"error": f"unknown storey_guid {a['storey_guid']}"}
        host_set = None
        if a.get("hosted_by"):
            h = self.by_guid.get(a["hosted_by"])
            if h is None:
                return {"error": f"unknown hosted_by {a['hosted_by']}"}
            host_set = set(self.hosted_by(h))
        sub = (a.get("name_contains") or "").lower()
        hits = []
        for sid in self.nodes:
            if not (self.isa(sid, cls) if cls else self.is_element(sid)):
                continue
            if cls and self.isa(sid, "IfcOpeningElement") and cls != "IfcOpeningElement":
                continue
            if storey is not None and self.storey_of(sid) != storey:
                continue
            if host_set is not None and sid not in host_set:
                continue
            if sub and sub not in str(self.nodes[sid].get("Name") or "").lower():
                continue
            hits.append(sid)
        hits.sort(key=lambda s: (self.cls(s), str(self.nodes[s].get("Name") or ""), s))
        return {"count": len(hits), "truncated": len(hits) > MAX_RESULTS,
                "elements": [self.brief(s) for s in hits[:MAX_RESULTS]]}

    def describe_element(self, guid: str | None = None, **_) -> dict:
        sid = self.by_guid.get(guid)
        if sid is None:
            return {"error": f"unknown guid {guid}"}
        p = self.nodes[sid]
        d = self.brief(sid)
        d["attributes"] = {k: p[k] for k in ("Description", "ObjectType", "Tag", "PredefinedType", "LongName") if p.get(k) is not None}
        t = self.type_of(sid)
        if t is not None:
            d["type"] = {"guid": self.guid(t), "class": self.cls(t), "name": self.nodes[t].get("Name")}
        d["property_sets"] = self._psets(sid)
        if self.isa(sid, "IfcWall"):
            d["hosted_elements"] = [self.brief(f) for f in self.hosted_by(sid)]
        h = self.host_of(sid)
        if h is not None:
            d["host_wall"] = self.brief(h)
        d.update(self._geometry(sid))
        return d

    def _psets(self, sid: int, max_sets: int = 8, max_props: int = 12) -> list[dict]:
        out = []
        for rel in self.rels_in(sid, "RelatedObjects", "IfcRelDefinesByProperties"):
            for ps in self.out(rel, "RelatingPropertyDefinition"):
                if not self.isa(ps, "IfcPropertySet"):
                    continue
                props = {}
                for pr in self.out(ps, "HasProperties")[:max_props]:
                    pn = self.nodes[pr]
                    if "NominalValue" in pn:
                        props[pn.get("Name")] = pn["NominalValue"]
                out.append({"name": self.nodes[ps].get("Name"), "properties": props})
        return out[:max_sets]

    def _geometry(self, sid: int) -> dict:
        res: dict = {}
        try:
            sc = self._ex._length_scale()
            pl = self._ex._single(sid, "ObjectPlacement")
            rot, org = self._ex._world_frame(pl)
            res["world_location_m"] = [round(c * sc, 3) for c in org]
            if self.isa(sid, "IfcWall"):
                res["axis_direction"] = [round(c, 3) for c in rot[0]]
                try:
                    res["length_m_approx"] = round(self._ex._wall_dimensions(sid)[0] * sc, 3)
                except Exception:
                    pass
        except Exception:
            pass
        return res


# =========================================================================== Schemas

def _guid():
    return {"type": "string", "pattern": GUID_PATTERN}


def _num():
    return {"type": "number"}


def _vec(n):
    return {"type": "array", "items": _num(), "minItems": n, "maxItems": n}


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


LOCATOR_STEP_SCHEMA = {"anyOf": [
    _obj({"action": {"const": "read"}, "op": {"const": "list_storeys"}, "args": _obj({}, [])},
         ["action", "op", "args"]),
    _obj({"action": {"const": "read"}, "op": {"const": "find_elements"},
          "args": _obj({"class": {"type": "string"}, "storey_guid": _guid(), "name_contains": {"type": "string"},
                        "hosted_by": _guid()}, [])},
         ["action", "op", "args"]),
    _obj({"action": {"const": "read"}, "op": {"const": "describe_element"},
          "args": _obj({"guid": _guid()}, ["guid"])},
         ["action", "op", "args"]),
    _obj({"action": {"const": "final"}, "guids": {"type": "array", "items": _guid(), "minItems": 1},
          "reason": {"type": "string"}}, ["action", "guids"]),
    _obj({"action": {"const": "abort"}, "reason": {"enum": ["ambiguous", "not_found"]},
          "detail": {"type": "string"}, "candidates": {"type": "array", "items": _guid()}},
         ["action", "reason"]),
]}

_SCALAR = {"type": ["string", "number", "boolean"]}


def _op(name, props, required, extra=None):
    p = {"op": {"const": name}, **props}
    if extra:
        p.update(extra)
    return _obj(p, ["op", *required])


GENERATOR_SCHEMA = {"anyOf": [
    _op("delete_element", {"guid": _guid(), "cascade": {"enum": ["dependents", "strict"]},
                           "remove_host_opening": {"type": "boolean"}}, ["guid"]),
    _op("move_element", {"guid": _guid(), "translation": _vec(3), "with_dependents": {"type": "boolean"}},
        ["guid", "translation"]),
    _op("move_element", {"guid": _guid(), "target_location": _vec(3), "with_dependents": {"type": "boolean"}},
        ["guid", "target_location"]),
    _op("clone_element", {"source_guid": _guid(), "translation": _vec(3), "name": {"type": "string"}},
        ["source_guid", "translation"]),
    _op("reassign_container", {"guid": _guid(), "target_storey_guid": _guid(),
                               "preserve_world_position": {"type": "boolean"}}, ["guid", "target_storey_guid"]),
    _op("set_property", {"guid": _guid(), "pset_name": {"type": "string"}, "property_name": {"type": "string"},
                         "value": _SCALAR}, ["guid", "pset_name", "property_name", "value"]),
    _op("set_attribute", {"guid": _guid(), "attribute": {"enum": ["Name", "Description", "ObjectType", "Tag",
                                                                  "LongName", "PredefinedType"]},
                          "value": {"type": ["string", "null"]}}, ["guid", "attribute", "value"]),
    _op("assign_type", {"guid": _guid(), "type_guid": _guid()}, ["guid", "type_guid"]),
    _op("add_element", {"ifc_class": {"const": "IfcWall"}, "storey_guid": _guid(), "type_guid": _guid(),
                        "z_offset": _num(), "start": _vec(2), "end": _vec(2), "thickness": _num(), "height": _num()},
        ["ifc_class", "storey_guid", "start", "end", "thickness", "height"]),
    _op("add_element", {"ifc_class": {"const": "IfcColumn"}, "storey_guid": _guid(), "type_guid": _guid(),
                        "z_offset": _num(), "position": _vec(2), "width": _num(), "depth": _num(), "height": _num()},
        ["ifc_class", "storey_guid", "position", "width", "depth", "height"]),
    _op("add_element", {"ifc_class": {"const": "IfcSlab"}, "storey_guid": _guid(), "type_guid": _guid(),
                        "z_offset": _num(), "outline": {"type": "array", "items": _vec(2), "minItems": 3},
                        "thickness": _num()}, ["ifc_class", "storey_guid", "outline", "thickness"]),
    _op("add_hosted_element", {"ifc_class": {"const": "IfcDoor"}, "host_guid": _guid(), "type_guid": _guid(),
                               "offset": _num(), "width": _num(), "height": _num(), "sill_height": _num()},
        ["ifc_class", "host_guid", "offset", "width", "height"]),
    _op("add_hosted_element", {"ifc_class": {"const": "IfcWindow"}, "host_guid": _guid(), "type_guid": _guid(),
                               "offset": _num(), "width": _num(), "height": _num(), "sill_height": _num()},
        ["ifc_class", "host_guid", "offset", "width", "height", "sill_height"]),
    _obj({"op": {"const": "abort"},
          "reason": {"enum": ["ambiguous", "not_found", "insufficient_information", "out_of_scope"]},
          "detail": {"type": "string"}, "candidates": {"type": "array", "items": _guid()}}, ["op", "reason"]),
]}

LOCATOR_SYSTEM = """You are the Locator of a pipeline that edits IFC building models.
You only READ the model. Identify the element(s) the task refers to, plus every other element the edit will
need as a parameter: the host wall when a door or window is added, the target storey when an element is added
or moved to another storey, a type object when the task names one.
Each turn answer with exactly one JSON object:
- {"action":"read","op":"list_storeys","args":{}}
- {"action":"read","op":"find_elements","args":{"class":"IfcDoor","storey_guid":"...","name_contains":"...","hosted_by":"..."}}
  (all filters optional and combined; "class" includes subclasses, e.g. IfcWall also finds IfcWallStandardCase)
- {"action":"read","op":"describe_element","args":{"guid":"..."}}
- {"action":"final","guids":[...],"reason":"..."}  when you know all GlobalIds the edit needs
- {"action":"abort","reason":"ambiguous|not_found","detail":"...","candidates":[...]}  if the reference is
  ambiguous (several matches and the task does not decide) or no element matches. Do not guess.
Elements are addressed only by GlobalId (22 characters). Lengths are meters."""

GENERATOR_SYSTEM = """You are the Generator of a pipeline that edits IFC building models.
You get a task and the elements found by the Locator. Propose exactly ONE operation as JSON, or an abort.
A deterministic Executor applies it; validators check the result. Use only GlobalIds given to you.
All lengths are meters, coordinates are world coordinates.

Operations:
- delete_element {guid, cascade: dependents|strict, remove_host_opening: bool}
- move_element {guid, translation [dx,dy,dz] | target_location [x,y,z], with_dependents: bool}
- clone_element {source_guid, translation [dx,dy,dz], name}
- reassign_container {guid, target_storey_guid, preserve_world_position: bool}  (moves an element to another storey)
- set_property {guid, pset_name, property_name, value}  (existing property only)
- set_attribute {guid, attribute: Name|Description|ObjectType|Tag|LongName|PredefinedType, value}
- assign_type {guid, type_guid}
- add_element {ifc_class: IfcWall, storey_guid, start [x,y], end [x,y], thickness, height, z_offset?, type_guid?}
  | {ifc_class: IfcColumn, storey_guid, position [x,y], width, depth, height, ...}
  | {ifc_class: IfcSlab, storey_guid, outline [[x,y],...], thickness, ...}
- add_hosted_element {ifc_class: IfcDoor|IfcWindow, host_guid (a wall), offset (distance of the element's CENTER
  from the wall start along the wall axis), width, height, sill_height (required for windows), type_guid?}
- abort {reason: ambiguous|not_found|insufficient_information|out_of_scope, detail}
  Use abort if the task needs information you do not have (e.g. size or position missing) or something the
  vocabulary cannot do (changing materials, rotating elements, deleting storeys or buildings).
If you get an error message from an earlier attempt, fix exactly that problem."""


# =========================================================================== Stationen

class LlmLocator(o.LocatorBase):
    def __init__(self, llm: LlmClient, schema, max_steps: int = 8):
        self.llm, self.schema, self.max_steps = llm, schema, max_steps

    def locate(self, task: str, snapshot: r.Snapshot) -> dict:
        view = GraphView(snapshot, self.schema)
        msgs = [{"role": "system", "content": LOCATOR_SYSTEM}, {"role": "user", "content": f"Task: {task}"}]
        ops = {"list_storeys": view.list_storeys, "find_elements": view.find_elements,
               "describe_element": view.describe_element}
        for _ in range(self.max_steps):
            try:
                step = self.llm.chat_json("locator", msgs, LOCATOR_STEP_SCHEMA)
            except Exception as e:  # ungültige Ausgabe oder Serverfehler
                return {"status": "abort", "abort": {"reason": "locator_failed", "detail": f"LLM-Aufruf: {e}"}}
            msgs.append({"role": "assistant", "content": json.dumps(step, ensure_ascii=False)})
            act = step["action"]
            if act == "abort":
                return {"status": "abort", "abort": {k: v for k, v in step.items() if k != "action"}}
            if act == "final":
                unknown = [gd for gd in step["guids"] if gd not in view.by_guid]
                if unknown:
                    msgs.append({"role": "user", "content": f"Error: unknown GlobalId(s) {unknown}. Read the model again."})
                    continue
                return {"status": "ok", "target_guids": step["guids"], "reason": step.get("reason", ""),
                        "targets": [view.describe_element(gd) for gd in step["guids"]]}
            res = ops[step["op"]](**step.get("args", {}))
            text = json.dumps(res, ensure_ascii=False)
            if len(text) > MAX_RESULT_CHARS:
                text = text[:MAX_RESULT_CHARS] + " ...(truncated)"
            msgs.append({"role": "user", "content": "Result: " + text})
        return {"status": "abort", "abort": {"reason": "locator_failed", "detail": f"no result after {self.max_steps} steps"}}


class LlmGenerator(o.GeneratorBase):
    def __init__(self, llm: LlmClient, max_tokens: int = 1024):
        self.llm, self.max_tokens = llm, max_tokens
        self._ops: list[dict] = []

    def generate(self, task: str, located: dict, feedback: list[str]) -> dict:
        if not feedback:
            self._ops = []
        if located.get("status") == "abort":
            a = located.get("abort", {})
            return {"op": "abort", "reason": a.get("reason", "not_found"), "detail": a.get("detail", ""),
                    "from": "locator"}
        info = {"target_guids": located.get("target_guids"), "targets": located.get("targets")}
        msgs = [{"role": "system", "content": GENERATOR_SYSTEM},
                {"role": "user", "content": f"Task: {task}\nLocated elements: " + json.dumps(info, ensure_ascii=False)}]
        for prev, fb in zip(self._ops, feedback):
            msgs.append({"role": "assistant", "content": json.dumps(prev, ensure_ascii=False)})
            msgs.append({"role": "user", "content": fb + "\nPropose a corrected operation."})
        try:
            op = self.llm.chat_json("generator", msgs, GENERATOR_SCHEMA, self.max_tokens)
        except Exception as e:
            return {"op": "abort", "reason": "generator_error", "detail": str(e)}
        self._ops.append(op)
        return op


# =========================================================================== Lauf über eine Aufgabenliste

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Aufgabenliste mit LLM-Locator und LLM-Generator durch die Pipeline schicken")
    ap.add_argument("ifc")
    ap.add_argument("--tasks", required=True, help='JSON-Liste: [{"id": "t1", "task": "..."}]')
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "EMPTY"))
    ap.add_argument("--model", help="Modell für beide Stationen (Standard: erstes Modell des Servers)")
    ap.add_argument("--locator-model")
    ap.add_argument("--generator-model")
    ap.add_argument("--no-think", action="store_true", help="Thinking abschalten (Qwen3-Familie)")
    ap.add_argument("--rounds", type=int, default=3,
                    help="Höchstzahl der Versuche je Aufgabe (1 = Reparaturschleife aus)")
    ap.add_argument("--no-validators", action="store_true",
                    help="Ablation: Structural und Semantic–Geometric Validator überspringen (Executor und Reconstruction laufen weiter)")
    ap.add_argument("--log", default="runs.jsonl", help="JSONL: ein Eintrag je Aufgabe (TaskResult)")
    ap.add_argument("--llm-log", default=None, help="JSONL: jeder LLM-Aufruf (Standard: <log>.llm.jsonl)")
    ap.add_argument("--neo4j", action="store_true")
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    ap.add_argument("--database", default=None)
    a = ap.parse_args(argv)

    from openai import OpenAI
    served = [m.id for m in OpenAI(base_url=a.base_url, api_key=a.api_key).models.list().data]
    base = a.model or served[0]
    loc_m, gen_m = a.locator_model or base, a.generator_model or base
    print("Server-Modelle:", served, "| Locator:", loc_m, "| Generator:", gen_m)
    llm_log = a.llm_log or a.log + ".llm.jsonl"
    loc = LlmClient(a.base_url, a.api_key, loc_m, a.no_think, llm_log)
    gen = LlmClient(a.base_url, a.api_key, gen_m, a.no_think, llm_log)

    import ifcopenshell
    import structural_validator as sv
    model = ifcopenshell.open(a.ifc)
    schema = sv.schema_for(model)
    source = os.path.basename(a.ifc)
    store = (o.Neo4jStore(a.uri, a.user, a.password, a.database, source) if a.neo4j
             else o.MemoryStore(r.snapshot_from_ifc(model, source), schema))
    orch = o.Orchestrator(a.ifc, store, locator=LlmLocator(loc, schema), generator=LlmGenerator(gen),
                          max_rounds=a.rounds, keep=False, log_path=a.log, validate=not a.no_validators)
    tasks = json.load(open(a.tasks, encoding="utf-8"))
    n_ok = 0
    try:
        for t in tasks:
            orch.out_path = os.path.splitext(a.ifc)[0] + f"_{t['id']}.ifc"
            res = orch.run(t["task"])
            n_ok += res.status == "success"
            print(f"{t['id']}: {res.status}" + (f" ({res.reason})" if res.reason else "")
                  + f" | Runden: {len(res.rounds)} | {res.seconds:.1f} s")
    finally:
        store.close()
    print(f"{n_ok}/{len(tasks)} erfolgreich. Protokolle: {a.log}, {llm_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

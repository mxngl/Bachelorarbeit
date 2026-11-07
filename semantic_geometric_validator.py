#!/usr/bin/env python3
"""
semantic_geometric_validator.py — Semantic–Geometric Validator (Stufe 2 der Prüfung, nach der Reconstruction).

Aufgabe: Ist das rekonstruierte IFC-Modell auch *räumlich und fachlich* plausibel? Der Structural Validator
prüft den Graphen (Beziehungen, Pflichtangaben). Dieser Validator liest die rekonstruierte IFC-Datei, berechnet
mit IfcOpenShell die Weltgeometrie und prüft nur die Bauteile, die die Operation betroffen hat (neu oder
verschoben), sowie deren Nachbarn im selben Geschoss.

Prüfgruppen (Codes siehe RULES):
  G1  Geometrie gültig/plausibel   Körper vorhanden, Volumen > 0, Maße passen zur Klasse (Warnung)
  G2  Öffnung in der Wand          Öffnung liegt innerhalb der Wand (Länge, Höhe), Öffnungen einer Wand
                                   überlappen sich nicht, Fenster/Tür liegt in ihrer Öffnung
  G3  Lage im Geschoss             Höhenlage passt zum Geschoss, Bauteil liegt nahe am Gebäude (Warnung)
  G4  Kollision                    Durchdringung zweier Bauteile über der Toleranz (Warnung, Messwert im Bericht)
  G5  Bestand unverändert          Bauteile, die die Operation nicht betreffen darf, behalten ihre Lage

Schweregrad gestaffelt: harte Verstöße (G1 Geometrie fehlt, G2, G5) sind Fehler und lösen die Reparaturschleife
aus; Plausibilität, Lage und Kollision sind Warnungen mit Messwert.

Nur *neue* Befunde zählen, wie beim Structural Validator: Dieselben Prüfungen laufen auf dem Ausgangsmodell für
die Bauteile, die es dort schon gab. Ein Befund mit gleichem Schlüssel (Code, Bauteile, Gegenstand) ist "existing".

Kollision: Volumen der Durchdringung, geschätzt durch Abtasten der Überlappungsbox auf einem Raster (Punkt-in-
Körper-Test mit Strahlenzählung). Ab 1 dm³ und 1 cm Durchdringung in jeder Richtung. Normale Anschlüsse
(IfcRelConnectsPathElements) und Öffnungen/Räume zählen nicht.

KI-Hinweis (TUHH-Handreichung "KI-Tools in Studium und Lehre"): mit Unterstützung von Claude (Anthropic)
erstellt. Prompts und Ausgaben gehören in Anhang F (AI usage documentation).

Nutzung:
  python semantic_geometric_validator.py AC20-FZK-Haus.ifc ergebnis_add.ifc
  python semantic_geometric_validator.py AC20-FZK-Haus.ifc ergebnis_add.ifc --op '{"op":"add_element",...}' --json bericht.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

import ifcopenshell
import ifcopenshell.geom as geom
import ifcopenshell.util.element as ue
import ifcopenshell.util.placement as placement
import ifcopenshell.util.unit as unit

from structural_validator import Finding, Report, ERROR, WARNING

# --------------------------------------------------------------------------- Regeln und Schwellen

RULES: dict[str, tuple[str, str, str]] = {
    "geometry_missing":     ("G1", ERROR,   "Bauteil liefert keine Geometrie"),
    "geometry_degenerate":  ("G1", ERROR,   "Körper hat kein Volumen oder ungültige Koordinaten"),
    "dimension_range":      ("G1", WARNING, "Maße liegen außerhalb des üblichen Bereichs der Klasse"),
    "opening_outside_wall": ("G2", ERROR,   "Öffnung liegt nicht innerhalb der Wand (Länge/Höhe)"),
    "openings_overlap":     ("G2", ERROR,   "Zwei Öffnungen derselben Wand überlappen sich"),
    "filling_outside_opening": ("G2", ERROR, "Fenster/Tür liegt nicht in ihrer Öffnung"),
    "outside_storey_height": ("G3", WARNING, "Unterkante liegt außerhalb des Höhenbereichs des Geschosses"),
    "outside_building":     ("G3", WARNING, "Bauteil liegt weit außerhalb des übrigen Geschosses"),
    "collision":            ("G4", WARNING, "Durchdringung mit einem anderen Bauteil über der Toleranz"),
    "existing_changed":     ("G5", ERROR,   "Bauteil geändert, das die Operation nicht betreffen darf"),
}

TOL_OPENING = 0.02          # m, Toleranz Öffnung in Wand / Füllung in Öffnung
TOL_COLLISION_VOLUME = 1e-3  # m³ (1 dm³)
TOL_COLLISION_DEPTH = 0.01   # m, Durchdringung in jeder Richtung
TOL_MOVED = 1e-6            # m, Lageunterschied, ab dem ein Bauteil als verschoben gilt
OUTSIDE_BUILDING = 3.0      # m Abstand zur Hülle des Geschosses
MAX_SAMPLES = 150_000
SKIP_COLLISION = ("IfcOpeningElement", "IfcSpace", "IfcVirtualElement", "IfcAnnotation", "IfcGrid", "IfcSite")

# Klasse -> (Beschreibung, Funktion Maße -> Meldungen); Maße aus dem lokalen Rahmen des Bauteils
def _rng(name, v, lo, hi):
    return [f"{name} {v:.3f} m (üblich {lo}–{hi} m)"] if not (lo <= v <= hi) else []


def _dims_wall(dx, dy, dz):
    return (_rng("Dicke", min(dx, dy), 0.04, 1.2) + _rng("Höhe", dz, 0.5, 8.0) + _rng("Länge", max(dx, dy), 0.1, 100))


def _dims_column(dx, dy, dz):
    return _rng("Querschnitt", min(dx, dy), 0.1, 3.0) + _rng("Höhe", dz, 0.5, 10.0)


def _dims_slab(dx, dy, dz):
    return _rng("Dicke", min(dx, dy, dz), 0.04, 1.5)


def _dims_door(dx, dy, dz):
    return _rng("Breite", max(dx, dy), 0.5, 3.5) + _rng("Höhe", dz, 1.6, 3.5)


def _dims_window(dx, dy, dz):
    return _rng("Breite", max(dx, dy), 0.2, 6.0) + _rng("Höhe", dz, 0.2, 4.0)


DIMENSION_RULES = {"IfcWall": _dims_wall, "IfcColumn": _dims_column, "IfcSlab": _dims_slab,
                   "IfcDoor": _dims_door, "IfcWindow": _dims_window}


class GeoReport(Report):
    def summary(self) -> str:
        e = [f for f in self.findings if not f.existing]
        ex = [f for f in self.findings if f.existing]
        n = lambda fs, s: sum(1 for f in fs if f.severity == s)
        return (f"{'OK' if self.ok else 'FEHLER'}: neu {n(e, ERROR)} Fehler, {n(e, WARNING)} Warnungen | "
                f"schon im Ausgangsmodell {n(ex, ERROR)} Fehler, {n(ex, WARNING)} Warnungen | "
                f"{self.edges} Bauteile geprüft, {self.seconds:.2f} s")

    def feedback(self, limit: int = 8) -> str:
        if self.ok:
            return ""
        errs = self.new_errors
        lines = [f"Geometrieprüfung fehlgeschlagen: {len(errs)} neue Fehler."]
        for f in errs[:limit]:
            who = f"{f.ifc_class} {f.guid}" if f.ifc_class else "Modell"
            lines.append(f"- [{f.group}/{f.code}] {who}: {f.message}")
        if len(errs) > limit:
            lines.append(f"- ... und {len(errs) - limit} weitere")
        return "\n".join(lines)


# --------------------------------------------------------------------------- Geometrie

def _settings(uncut: bool):
    s = geom.settings()
    s.set("use-world-coords", True)
    if uncut:
        s.set("disable-opening-subtractions", True)
    return s


class Geo:
    """Geometrie eines Modells mit Zwischenspeicher. Alle Längen in Metern."""

    def __init__(self, model: ifcopenshell.file):
        self.model = model
        self.scale = unit.calculate_unit_scale(model)
        self._s = {False: _settings(False), True: _settings(True)}
        self._mesh: dict = {}

    def matrix(self, el) -> np.ndarray | None:
        if el.ObjectPlacement is None:
            return None
        m = np.array(placement.get_local_placement(el.ObjectPlacement), dtype=float)
        m[:3, 3] *= self.scale
        return m

    def mesh(self, el, uncut: bool = False):
        key = (el.GlobalId, uncut)
        if key not in self._mesh:
            res = None
            if getattr(el, "Representation", None) is not None:
                try:
                    sh = geom.create_shape(self._s[uncut], el)
                    v = np.array(sh.geometry.verts, dtype=float).reshape(-1, 3)
                    f = np.array(sh.geometry.faces, dtype=int).reshape(-1, 3)
                    if len(v) and len(f):
                        res = (v, f)
                except Exception:
                    res = None
            self._mesh[key] = res
        return self._mesh[key]

    def aabb(self, el, uncut: bool = False):
        m = self.mesh(el, uncut)
        return None if m is None else (m[0].min(0), m[0].max(0))

    def local_extent(self, el, uncut: bool = False):
        """Ausdehnung im lokalen Rahmen des Bauteils (Länge, Dicke, Höhe unabhängig von der Drehung)."""
        m, mat = self.mesh(el, uncut), self.matrix(el)
        if m is None or mat is None:
            return None
        inv = np.linalg.inv(mat)
        lv = (inv[:3, :3] @ m[0].T).T + inv[:3, 3]
        return lv.min(0), lv.max(0)

    def volume(self, el) -> float:
        m = self.mesh(el)
        if m is None:
            return 0.0
        v, f = m
        tri = v[f]
        return float(abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)


_DIR = np.array([0.5773, 0.3719, 0.7239])
_DIR /= np.linalg.norm(_DIR)


def _inside(points: np.ndarray, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Punkt-in-Körper-Test durch Zählen der Strahltreffer (Möller–Trumbore), Strahl in irrationaler Richtung."""
    tri = verts[faces]
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(_DIR, e2)
    a = (e1 * h).sum(1)
    ok = np.abs(a) > 1e-12
    v0, e1, e2, h, a = v0[ok], e1[ok], e2[ok], h[ok], a[ok]
    out = np.zeros(len(points), dtype=bool)
    chunk = max(100, int(3e6 // max(len(a), 1)))
    for i in range(0, len(points), chunk):
        s = points[i:i + chunk, None, :] - v0[None]
        u = (s * h[None]).sum(2) / a
        q = np.cross(s, e1[None])
        v = (q @ _DIR) / a
        t = (q * e2[None]).sum(2) / a
        hit = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        out[i:i + chunk] = hit.sum(1) % 2 == 1
    return out


def overlap_volume(a, b) -> tuple[float, tuple]:
    """Geschätztes Durchdringungsvolumen zweier Netze (m³) und Maße der Überlappungsbox."""
    (va, fa), (vb, fb) = a, b
    lo = np.maximum(va.min(0), vb.min(0))
    hi = np.minimum(va.max(0), vb.max(0))
    dims = hi - lo
    if (dims <= TOL_COLLISION_DEPTH).any():
        return 0.0, tuple(np.maximum(dims, 0))
    n = np.maximum(1, np.ceil(dims / max(0.02, (dims.prod() / MAX_SAMPLES) ** (1 / 3)))).astype(int)
    step = dims / n
    axes = [lo[i] + (np.arange(n[i]) + 0.5) * step[i] for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    pts = pts[_inside(pts, va, fa)]
    if len(pts):
        pts = pts[_inside(pts, vb, fb)]
    return float(len(pts) * step.prod()), tuple(dims)


# --------------------------------------------------------------------------- Prüfungen

def _storey(el):
    c = ue.get_container(el)
    while c is not None and not c.is_a("IfcBuildingStorey"):
        c = ue.get_container(c)
    return c


def _skip_collision(el) -> bool:
    return any(el.is_a(c) for c in SKIP_COLLISION)


class Checks:
    """Prüfungen für eine Menge von Bauteilen (GlobalIds) in einem Modell."""

    def __init__(self, model: ifcopenshell.file, geo: Geo):
        self.model, self.geo = model, geo
        self.findings: list[Finding] = []
        self._path_pairs: set | None = None

    def add(self, code, el, message, subject="", other=None, **detail):
        group, sev, _ = RULES[code]
        guids = sorted(x for x in (el.GlobalId if el else None, other.GlobalId if other else None) if x)
        self.findings.append(Finding(group, code, sev, None, el.GlobalId if el else None,
                                     el.is_a() if el else None, message,
                                     subject=f"{subject}#{'|'.join(guids)}", detail=detail))

    def run(self, guids) -> list[Finding]:
        els = []
        for g_ in sorted(guids):
            try:
                els.append(self.model.by_guid(g_))
            except Exception:
                continue
        for el in els:
            self.geometry(el)
            if el.is_a("IfcOpeningElement"):
                self.opening(el)
            if el.is_a("IfcDoor") or el.is_a("IfcWindow"):
                self.filling(el)
            self.storey(el)
        self.collisions(els)
        return self.findings

    # -- G1 ---------------------------------------------------------------------------------
    def geometry(self, el):
        m = self.geo.mesh(el)
        if m is None:
            self.add("geometry_missing", el, "Keine Geometrie berechenbar (Darstellung fehlt oder ist unlesbar)")
            return
        if not np.isfinite(m[0]).all() or self.geo.volume(el) < 1e-6:
            self.add("geometry_degenerate", el, f"Volumen {self.geo.volume(el):.6f} m³", volume=self.geo.volume(el))
            return
        rule = next((f for c, f in DIMENSION_RULES.items() if el.is_a(c)), None)
        ext = self.geo.local_extent(el)
        if rule and ext is not None:
            d = ext[1] - ext[0]
            msgs = rule(*[float(x) for x in d])
            if msgs:
                self.add("dimension_range", el, "; ".join(msgs), subject="dims", dims=[round(float(x), 3) for x in d])

    # -- G2 ---------------------------------------------------------------------------------
    def host_of(self, opening):
        rels = getattr(opening, "VoidsElements", None) or []
        return rels[0].RelatingBuildingElement if rels else None

    def opening(self, o):
        host = self.host_of(o)
        if host is None:
            return                                   # Struktur: Aufgabe des Structural Validators
        wl = self.geo.local_extent(host, uncut=True)
        mat = self.geo.matrix(host)
        om = self.geo.mesh(o)
        if wl is None or mat is None or om is None:
            return
        inv = np.linalg.inv(mat)
        lv = (inv[:3, :3] @ om[0].T).T + inv[:3, 3]
        lo, hi = lv.min(0), lv.max(0)
        bad = []
        for ax, name in ((0, "Länge"), (2, "Höhe")):
            if lo[ax] < wl[0][ax] - TOL_OPENING or hi[ax] > wl[1][ax] + TOL_OPENING:
                bad.append(f"{name}: Öffnung {lo[ax]:.3f}…{hi[ax]:.3f} m, Wand {wl[0][ax]:.3f}…{wl[1][ax]:.3f} m")
        if bad:
            self.add("opening_outside_wall", host, "Öffnung " + o.GlobalId + " ragt aus der Wand: " + "; ".join(bad),
                     subject=o.GlobalId, other=o, opening=o.GlobalId)
        for rel in getattr(host, "HasOpenings", None) or []:
            o2 = rel.RelatedOpeningElement
            if o2.GlobalId <= o.GlobalId:
                continue
            m2 = self.geo.mesh(o2)
            if m2 is None:
                continue
            lv2 = (inv[:3, :3] @ m2[0].T).T + inv[:3, 3]
            lo2, hi2 = lv2.min(0), lv2.max(0)
            ox = min(hi[0], hi2[0]) - max(lo[0], lo2[0])
            oz = min(hi[2], hi2[2]) - max(lo[2], lo2[2])
            if ox > TOL_OPENING and oz > TOL_OPENING:
                self.add("openings_overlap", o, f"Öffnungen {o.GlobalId} und {o2.GlobalId} überlappen {ox:.3f} × {oz:.3f} m",
                         other=o2, overlap=[round(ox, 3), round(oz, 3)])

    def filling(self, f):
        fills = getattr(f, "FillsVoids", None) or []
        if not fills:
            return
        o = fills[0].RelatingOpeningElement
        host = self.host_of(o)
        om, fm = self.geo.mesh(o), self.geo.mesh(f)
        if host is None or om is None or fm is None:
            return
        mat = self.geo.matrix(host)
        inv = np.linalg.inv(mat)
        lo_ov = (inv[:3, :3] @ om[0].T).T + inv[:3, 3]
        lo_o, hi_o = lo_ov.min(0), lo_ov.max(0)
        lf = (inv[:3, :3] @ fm[0].T).T + inv[:3, 3]
        lo_f, hi_f = lf.min(0), lf.max(0)
        bad = []
        for ax, name in ((0, "Länge"), (2, "Höhe")):
            if lo_f[ax] < lo_o[ax] - TOL_OPENING or hi_f[ax] > hi_o[ax] + TOL_OPENING:
                bad.append(f"{name}: Bauteil {lo_f[ax]:.3f}…{hi_f[ax]:.3f} m, Öffnung {lo_o[ax]:.3f}…{hi_o[ax]:.3f} m")
        if bad:
            self.add("filling_outside_opening", f, "ragt aus der Öffnung: " + "; ".join(bad), subject=o.GlobalId, other=o)

    # -- G3 ---------------------------------------------------------------------------------
    def storey(self, el):
        if el.is_a("IfcOpeningElement"):
            return
        st = _storey(el)
        bb = self.geo.aabb(el)
        if st is None or bb is None:
            return
        elev = self._elevation(st)
        above = [self._elevation(s) for s in self.model.by_type("IfcBuildingStorey")]
        above = [e for e in above if e is not None and elev is not None and e > elev + 1e-6]
        if elev is None:
            return
        top = min(above) if above else elev + 6.0
        zmin = float(bb[0][2])
        if zmin < elev - 1.0 or zmin > top:
            self.add("outside_storey_height", el, f"Unterkante {zmin:.2f} m, Geschoss {st.Name!r} von {elev:.2f} bis {top:.2f} m",
                     subject=st.GlobalId, z_min=round(zmin, 3), storey=st.GlobalId)
        # Abstand zur Hülle der übrigen Bauteile des Geschosses
        others = [e for e in self._storey_elements(st) if e.GlobalId != el.GlobalId and not _skip_collision(e)]
        boxes = [b for b in (self.geo.aabb(e) for e in others) if b is not None]
        if len(boxes) >= 3:
            lo = np.min([b[0] for b in boxes], 0)
            hi = np.max([b[1] for b in boxes], 0)
            c = (bb[0] + bb[1]) / 2
            d = np.maximum(np.maximum(lo[:2] - c[:2], c[:2] - hi[:2]), 0)
            if np.linalg.norm(d) > OUTSIDE_BUILDING:
                self.add("outside_building", el, f"Mittelpunkt {np.linalg.norm(d):.1f} m außerhalb der Hülle des Geschosses",
                         subject=st.GlobalId, distance=round(float(np.linalg.norm(d)), 2))

    def _elevation(self, st):
        mat = self.geo.matrix(st)
        if mat is not None and st.ObjectPlacement is not None:
            return float(mat[2, 3])
        e = getattr(st, "Elevation", None)
        return None if e is None else float(e) * self.geo.scale

    def _storey_elements(self, st):
        out = []
        for rel in getattr(st, "ContainsElements", None) or []:
            out.extend(rel.RelatedElements)
        return out

    # -- G4 ---------------------------------------------------------------------------------
    def connected_pairs(self) -> set:
        if self._path_pairs is None:
            pairs = set()
            for rel in self.model.by_type("IfcRelConnectsPathElements"):
                pairs.add(frozenset((rel.RelatingElement.GlobalId, rel.RelatedElement.GlobalId)))
            self._path_pairs = pairs
        return self._path_pairs

    def collisions(self, els):
        done = set()
        for el in els:
            if _skip_collision(el):
                continue
            ba = self.geo.mesh(el)
            if ba is None:
                continue
            st = _storey(el)
            cands = self._storey_elements(st) if st is not None else []
            lo_a, hi_a = ba[0].min(0), ba[0].max(0)
            for other in cands:
                if other.GlobalId == el.GlobalId or _skip_collision(other):
                    continue
                pair = frozenset((el.GlobalId, other.GlobalId))
                if pair in done or pair in self.connected_pairs():
                    continue
                done.add(pair)
                bb = self.geo.aabb(other)
                if bb is None or (np.minimum(hi_a, bb[1]) - np.maximum(lo_a, bb[0]) <= TOL_COLLISION_DEPTH).any():
                    continue
                vol, dims = overlap_volume(ba, self.geo.mesh(other))
                if vol > TOL_COLLISION_VOLUME:
                    self.add("collision", el, f"Durchdringung mit {other.is_a()} {other.GlobalId}: {vol:.3f} m³ "
                             f"(Überlappungsbox {dims[0]:.2f} × {dims[1]:.2f} × {dims[2]:.2f} m)",
                             other=other, volume=round(vol, 4), box=[round(float(x), 3) for x in dims])


# --------------------------------------------------------------------------- Validator

def dependents(model, el) -> set:
    """Bauteil plus abhängige Bauteile (Öffnungen, Füllungen, Teile)."""
    out, todo = set(), [el]
    while todo:
        e = todo.pop()
        if e.GlobalId in out:
            continue
        out.add(e.GlobalId)
        for rel in getattr(e, "HasOpenings", None) or []:
            todo.append(rel.RelatedOpeningElement)
        for rel in getattr(e, "HasFillings", None) or []:
            todo.append(rel.RelatedBuildingElement)
        for rel in getattr(e, "IsDecomposedBy", None) or []:
            todo.extend(rel.RelatedObjects)
    return out


MOVING_OPS = {"move_element": "guid", "reassign_container": "guid"}


class SemanticGeometricValidator:
    def __init__(self, original: ifcopenshell.file):
        self.original = original
        self.geo0 = Geo(original)
        self.guids0 = {e.GlobalId for e in original.by_type("IfcProduct")}
        self.m0 = {e.GlobalId: self.geo0.matrix(e) for e in original.by_type("IfcProduct") if e.ObjectPlacement}

    def affected(self, result: ifcopenshell.file, geo: Geo) -> tuple[set, set]:
        """(neue, verschobene) Bauteile im Ergebnis."""
        new, moved = set(), set()
        for e in result.by_type("IfcProduct"):
            if not e.is_a("IfcElement") or e.ObjectPlacement is None:
                continue
            if e.GlobalId not in self.guids0:
                new.add(e.GlobalId)
            else:
                m0, m1 = self.m0.get(e.GlobalId), geo.matrix(e)
                if m0 is not None and m1 is not None and np.abs(m0 - m1).max() > TOL_MOVED:
                    moved.add(e.GlobalId)
        return new, moved

    def validate(self, result: ifcopenshell.file, op: dict | None = None) -> GeoReport:
        t0 = time.time()
        geo = Geo(result)
        new, moved = self.affected(result, geo)
        findings = Checks(result, geo).run(new | moved)
        base_guids = (new | moved) & self.guids0
        base = Checks(self.original, self.geo0).run(base_guids) if base_guids else []
        base_keys = {f.key for f in base}
        base_pairs = {(f.code, f.subject) for f in base}
        # G5: Bestand unverändert
        if op is not None:
            allowed = set()
            if op.get("op") in MOVING_OPS:
                try:
                    allowed = dependents(result, result.by_guid(op[MOVING_OPS[op["op"]]]))
                except Exception:
                    allowed = set()
            for gid in sorted(moved - allowed):
                el = result.by_guid(gid)
                c = Checks(result, geo)
                c.add("existing_changed", el, f"Lage von {el.is_a()} {gid} hat sich geändert, obwohl "
                      f"{op.get('op')} sie nicht betreffen darf")
                findings += c.findings
        for f in findings:
            f.existing = (f.code, f.subject) in base_pairs
        n = len(result.by_type("IfcProduct"))
        rep = GeoReport(findings, n, len(new) + len(moved), time.time() - t0,
                        f"geprüft: {len(new)} neue, {len(moved)} verschobene Bauteile")
        order = {ERROR: 0, WARNING: 1}
        rep.findings.sort(key=lambda f: (f.existing, order[f.severity], f.group, f.code, f.guid or ""))
        return rep


def print_rules() -> None:
    for code, (grp, sev, text) in RULES.items():
        print(f"{grp}  {sev:7s} {code:24s} {text}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Semantic–Geometric Validator: rekonstruiertes IFC prüfen")
    ap.add_argument("original", nargs="?", help="Ausgangs-IFC")
    ap.add_argument("result", nargs="?", help="rekonstruiertes IFC")
    ap.add_argument("--op", help="Operation als JSON (aktiviert G5)")
    ap.add_argument("--json")
    ap.add_argument("--all", action="store_true", help="auch bereits vorhandene Befunde auflisten")
    ap.add_argument("--list-rules", action="store_true")
    args = ap.parse_args(argv)
    if args.list_rules:
        print_rules()
        return 0
    if not args.original or not args.result:
        ap.error("Ausgangs- und Ergebnis-IFC angeben")
    v = SemanticGeometricValidator(ifcopenshell.open(args.original))
    rep = v.validate(ifcopenshell.open(args.result), json.loads(args.op) if args.op else None)
    print(rep.summary())
    print(rep.diff_summary)
    for f in rep.new_errors + rep.new_warnings + (rep.existing if args.all else []):
        tag = "bestehend" if f.existing else f.severity
        print(f"  [{f.group}/{f.code}] {tag}: {f.ifc_class} {f.guid}: {f.message}")
    if not args.all and rep.existing:
        print(f"  ({len(rep.existing)} bestehende Befunde, mit --all anzeigen)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep.to_json(), fh, ensure_ascii=False, indent=2)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())

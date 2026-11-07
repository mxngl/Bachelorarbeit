import ifcopenshell
m = ifcopenshell.open("ergebnis.ifc")

def produkte(e, tiefe=6, gesehen=None):
    gesehen = gesehen if gesehen is not None else set()
    if e.id() in gesehen or tiefe < 0:
        return
    gesehen.add(e.id())
    if e.is_a("IfcProduct"):
        print("  Element:", e.is_a(), "|", e.Name, "|", e.GlobalId)
        return
    for rel in m.get_inverse(e):
        if rel.is_a("IfcRelDefinesByProperties"):
            for o in rel.RelatedObjects:
                print("  Element:", o.is_a(), "|", o.Name, "|", o.GlobalId)
        else:
            produkte(rel, tiefe - 1, gesehen)

for sid, text in [(97, "Textwert"), (14980, "verschobener Punkt")]:
    e = m.by_id(sid)
    print(f"#{sid} {text}:", e)
    produkte(e)

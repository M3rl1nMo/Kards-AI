"""Classify every unresolved native-card text fragment into mechanism buckets.

This is the required work-queue report from the handoff task book. It is a
heuristic, machine-assisted classifier: every fragment of an unresolved or
partial card is assigned to exactly one of the 12 mechanism categories, listed
with its card id, card name, and exact text so no card is handled ad hoc.

It reads the *honest* rule store produced by ``tools/build_card_rules.py``
(whose status is engine-coverage-aware), so the buckets reflect true execution
coverage rather than the over-optimistic parser status.

Outputs:
  - docs/rule_coverage.json            (machine-readable, full fragment lists)
  - RULE_IMPLEMENTATION_REPORT.md      (refreshed coverage + category breakdown)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.cards.loader import CardDatabase

CATALOG = ROOT / "data/source/kards_info_cards.json"
RULE_STORE = ROOT / "data/rules/card_rules.json"
COVERAGE_JSON = ROOT / "docs/rule_coverage.json"
REPORT = ROOT / "RULE_IMPLEMENTATION_REPORT.md"

# Task-book mechanism categories (ordered for stable reporting).
CATEGORIES = [
    ("resource_cost", "Resource / cost modification (incl. first-per-turn)"),
    ("attribute_change", "Directed or global attribute changes"),
    ("movement", "Movement, retreat, frontline control, deployment"),
    ("draw_search", "Draw, search, reveal, Develop, spawn, copy, shuffle, discard"),
    ("damage_combat", "Damage, combat, destroy, repair, heal, HQ defense"),
    ("choice_target", "Choice, target selection, randomness, deterministic RNG"),
    ("temporary", "Temporary effects, duration, delayed triggers, expiry"),
    ("keyword", "Keyword / ability semantics"),
    ("countermeasure", "Countermeasure trigger, cancel, consume"),
    ("condition", "Conditions, thresholds, history state, zone judgement"),
    ("special", "Irreducible special mechanics"),
    ("ambiguous", "Insufficient source text / semantic ambiguity"),
]

PLANNED_ACTION = {
    "resource_cost": "cost_modifier model on PlayerState + card_play_cost hook",
    "attribute_change": "buff/debuff/modify_* via EffectResolver",
    "movement": "move/spawn/return_to_hand via EffectResolver + native",
    "draw_search": "draw/develop/search/copy primitives (seeded RNG)",
    "damage_combat": "damage/destroy/pin/heal via EffectResolver + native",
    "choice_target": "target selector AST + seeded RNG choice",
    "temporary": "effect with expiry/event scope (no permanent mutation)",
    "keyword": "KeywordEngine / AbilityEngine semantics",
    "countermeasure": "CountermeasureResolver trigger/cancel/consume",
    "condition": "conditional Action wrapper + history predicates",
    "special": "mechanism-level Handler (never card-named)",
    "ambiguous": "needs human / official KARDS rule confirmation",
}

REASON = {
    "resource_cost": "Cost text not yet parsed into a player cost modifier.",
    "attribute_change": "Stat change not yet mapped to a verified Action.",
    "movement": "Zone/movement effect not yet mapped to a verified Action.",
    "draw_search": "Zone operation not yet mapped to a verified Action.",
    "damage_combat": "Damage/combat effect not yet mapped to a verified Action.",
    "choice_target": "Needs deterministic target/choice selection.",
    "temporary": "Temporary effect needs expiry/event scope, not permanent mutation.",
    "keyword": "Catalog ability not yet wired into combat/rule flow.",
    "countermeasure": "Countermeasure trigger/cancel/consume not fully implemented.",
    "condition": "Conditional/threshold/history effect not yet implemented.",
    "special": "Irreducible mechanic; candidate for a mechanism-level Handler.",
    "ambiguous": "Source text is insufficient or semantically ambiguous.",
}

# Strip the engine-coverage note prefix so classification keys on the real text.
_ENGINE_NOTE = re.compile(r"^\[engine does not implement kind=[^\]]*\] ")


def classify(fragment: str) -> str:
    f = fragment.lower()
    if re.search(r"\b(counter|cancel the effect|intercept|trigger this card|when the enemy|if your hq is about to)\b", f):
        return "countermeasure"
    if re.search(r"\b(this turn|next turn|until\b|for the rest of|this game|at the start of your next|at the end of (your )?next)\b", f):
        return "temporary"
    if re.search(r"\b(choose|adjacent|random |all (enemy|friendly) units?|all units|each (enemy|friendly))\b", f):
        return "choice_target"
    if re.search(r"\b(kredit|credits|cost|operation cost|extra kredit)\b", f):
        return "resource_cost"
    if re.search(r"\b(draw|develop|search|reveal|from your deck|from the deck|copy|shuffle|discard|mill|to your hand|to the enemy hand|on top of|return .* to hand|send .* to hand)\b", f):
        return "draw_search"
    if re.search(r"\b(frontline|support line|retreat|move to|deploy|to the battlefield|same front|the battlefield)\b", f):
        return "movement"
    if re.search(r"\b(blitz|fury|guard|shock|salvage|ambush|smokescreen|mobilize|intel|bond|heavy armor|covert|alpine|airborne|mobilized)\b", f):
        return "keyword"
    if re.search(r"\b(deal|damage|destroy|pin|suppress|repair|heal|hq|combat|fight|lethal)\b", f):
        return "damage_combat"
    if re.search(r"([+-]\s*\d+|\bgets\b|\bgain\b|\bset\b|\bswap\b|\breduce\b|\bincrease\b|\blower\b|\braise\b|\battack\b|\bdefense\b)", f):
        return "attribute_change"
    if re.search(r"\b(if |when |whenever|while |instead|unless|for each|equal to|you control|enemy has|has \d+|cost \d+ or more)\b", f):
        return "condition"
    if re.search(r"\b(instead|dealt to .* instead|redirect|both|each)\b", f):
        return "special"
    return "ambiguous"


def main() -> None:
    cards = CardDatabase.from_file(CATALOG)
    card_by_id = {c.id: c for c in cards}

    store = json.loads(RULE_STORE.read_text(encoding="utf-8"))
    store_by_id = {entry["card_id"]: entry for entry in store["rules"]}

    status_counts = {"implemented": 0, "partial": 0, "unresolved": 0}
    by_category: dict[str, dict] = {key: {"cards": [], "fragments": []} for key, _ in CATEGORIES}
    by_type_status: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in status_counts})
    by_category_type: dict[str, dict[str, int]] = {key: defaultdict(int) for key, _ in CATEGORIES}

    for card in cards:
        entry = store_by_id.get(card.id)
        if entry is None:
            continue
        status = entry["status"]
        status_counts[status] += 1
        by_type_status[card.type][status] += 1
        if status == "implemented":
            continue
        for frag in entry.get("unresolved_fragments") or []:
            text = _ENGINE_NOTE.sub("", frag)
            cat = classify(text)
            bucket = by_category[cat]
            if card.id not in bucket["cards"]:
                bucket["cards"].append(card.id)
            bucket["fragments"].append({
                "card_id": card.id,
                "card_name": card.name,
                "type": card.type,
                "fragment": frag,
                "classified_from": text,
            })
            by_category_type[cat][card.type] += 1

    coverage = {
        "schema_version": 1,
        "generated_by": "tools/classify_coverage.py",
        "note": "Reads engine-coverage-aware status from data/rules/card_rules.json.",
        "total_cards": len(cards),
        "status_counts": status_counts,
        "categories": {
            key: {
                "label": label,
                "card_count": len(data["cards"]),
                "fragment_count": len(data["fragments"]),
                "planned_action": PLANNED_ACTION[key],
                "status": "pending",
                "reason": REASON[key],
                "by_type": dict(by_category_type[key]),
                "card_ids": data["cards"],
                "fragments": data["fragments"],
            }
            for (key, label), data in zip(CATEGORIES, by_category.values())
        },
    }
    COVERAGE_JSON.write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Refresh the human-readable report.
    rows = ["# Native Rule Implementation Report", "",
            "Generated by `tools/build_card_rules.py` (engine-coverage-aware) and `tools/classify_coverage.py` (mechanism categories).",
            "", "## Coverage", "", "| Status | Cards |", "|---|---:|"]
    rows.append("| implemented | {0} |".format(status_counts["implemented"]))
    rows.append("| partial | {0} |".format(status_counts["partial"]))
    rows.append("| unresolved | {0} |".format(status_counts["unresolved"]))
    rows.extend(["", "## By card type", "", "| Type | Implemented | Partial | Unresolved |", "|---|---:|---:|---:|"])
    for ctype in sorted(by_type_status):
        v = by_type_status[ctype]
        rows.append("| {0} | {1} | {2} | {3} |".format(ctype, v["implemented"], v["partial"], v["unresolved"]))
    rows.extend(["", "## Unresolved fragments by mechanism category", "",
                 "Every unresolved/partial fragment is bucketed into exactly one mechanism category. "
                 "Counts are heuristic (machine-assisted) and must be reviewed before claiming parity.",
                 "", "| Category | Cards | Fragments | Planned action |", "|---|---:|---:|---|"])
    for key, label in CATEGORIES:
        data = coverage["categories"][key]
        rows.append("| {0} | {1} | {2} | {3} |".format(label, data["card_count"], data["fragment_count"], PLANNED_ACTION[key]))
    rows.extend(["", "### Category detail", ""])
    for key, label in CATEGORIES:
        data = coverage["categories"][key]
        rows.append("#### {0}".format(label))
        rows.append("- Cards: {0} | Fragments: {1}".format(data["card_count"], data["fragment_count"]))
        rows.append("- Planned action: {0}".format(data["planned_action"]))
        rows.append("- Status: {0}".format(data["status"]))
        rows.append("- Reason: {0}".format(data["reason"]))
        rows.append("- By type: {0}".format(dict(data["by_type"])))
        rows.append("")
    rows.extend(["Unresolved cards remain non-executable by design. A card is only marked implemented "
                 "when every parsed sentence maps to a kind the execution layer actually handles.", ""])
    REPORT.write_text("\n".join(rows), encoding="utf-8")

    print(json.dumps({
        "total_cards": len(cards),
        "status_counts": status_counts,
        "categories": {key: {"cards": coverage["categories"][key]["card_count"], "fragments": coverage["categories"][key]["fragment_count"]} for key, _ in CATEGORIES},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

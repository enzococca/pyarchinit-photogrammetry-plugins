"""The legend of the plugin as Markdown tables, from the vocabulary itself (``vocabulary.load()``): the guide
reproduces them between the markers ``<!-- legend:start -->`` and ``<!-- legend:end -->`` and a test checks that
they still match.

    python scripts/legend_table.py [--write GUIDE.md ...]   # print the tables, or write them into the guides"""
from __future__ import annotations

import re
import sys
from pathlib import Path

START, END = "<!-- legend:start -->", "<!-- legend:end -->"
HEADERS = {"en": (("Code", "ICOMOS-ISCS name", "UNI 11182 name", "Colour", "Hatch (matplotlib / DXF)"),
                  ("Code", "English name", "Italian name", "Colour", "Hatch (matplotlib / DXF)")),
           "it": (("Codice", "Nome ICOMOS-ISCS", "Nome UNI 11182", "Colore", "Retino (matplotlib / DXF)"),
                  ("Codice", "Nome inglese", "Nome italiano", "Colore", "Retino (matplotlib / DXF)"))}
TITLES = {"en": ("Glossary terms (ICOMOS-ISCS and UNI 11182)", "Local terms (masonry, not in a standard)",
                 "Interventions (families of EN 15898)"),
          "it": ("Termini dei glossari (ICOMOS-ISCS e UNI 11182)", "Termini locali (murature, non normativi)",
                 "Interventi (famiglie di EN 15898)")}


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|")


def _code(text: str) -> str:
    return "`" + str(text).replace("|", "\\|") + "`"


def _row(code, a, b, term) -> str:
    swatch = f'<span style="color:{term.colour}">&#9632;</span> `{term.colour}`'
    hatch = f"{_code(term.hatch.get('mpl', ''))} / {_code(term.hatch.get('dxf', ''))}"
    return f"| {_code(code)} | {_cell(a)} | {_cell(b)} | {swatch} | {hatch} |"


def _table(headers, rows) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + rows


def tables(lang: str = "en") -> str:
    """The three tables (glossaries, local terms, interventions) of the default legend, in ``lang``."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ppg_conservation.vocabulary import UNNAMED, load

    vocab = load()
    glossary, local, treat = [], [], []
    for term in vocab.terms():
        if term.code == UNNAMED:
            continue
        if term.kind == "treatment":
            treat.append(_row(term.code, term.names["en"], term.names["it"], term))
        elif term.local:
            local.append(_row(term.code, term.names["en"], term.names["it"], term))
        elif term.code.startswith("UNI."):
            iscs = ", ".join(vocab.term(c).names["en"] for c in term.iscs or ()) or "-"
            glossary.append(_row(term.code, iscs, term.names["it"], term))
        else:
            uni = vocab.term(term.uni).names["it"] if term.uni and term.uni in vocab else "-"
            glossary.append(_row(term.code, term.names["en"], uni, term))
    head, other = HEADERS[lang]
    t1, t2, t3 = TITLES[lang]
    out = [f"**{t1}**", ""] + _table(head, glossary) + ["", f"**{t2}**", ""] + _table(other, local) + \
        ["", f"**{t3}**", ""] + _table(other, treat)
    return "\n".join(out)


def block(lang: str) -> str:
    return f"{START}\n{tables(lang)}\n{END}"


def current(text: str) -> str | None:
    """The block between the markers in a guide, or None."""
    m = re.search(re.escape(START) + r".*?" + re.escape(END), text, re.S)
    return m.group(0) if m else None


def write(path: Path) -> None:
    lang = "it" if path.name.endswith(".it.md") else "en"
    text = path.read_text(encoding="utf-8")
    old = current(text)
    if old is None:
        raise SystemExit(f"{path}: no {START} ... {END} block")
    path.write_text(text.replace(old, block(lang)), encoding="utf-8")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--write"]:
        for p in args[1:]:
            write(Path(p))
    else:
        print(tables(args[0] if args else "en"))

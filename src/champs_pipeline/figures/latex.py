"""LaTeX export of tables with booktabs rules.

Only the body of the table is written, as a ``tabular`` environment or, for a
table that runs over several pages, as a ``longtable`` environment. The
manuscript provides the caption and the notes. The output requires the
``booktabs`` package, the ``longtable`` package for long tables, and the
``array`` package for a column with a declaration such as
``>{\raggedright\arraybackslash}p{4cm}``.
"""

import re
import textwrap

# One column of a LaTeX column specification: l, c, r, or a paragraph column such as p{5cm},
# optionally preceded by a declaration such as >{\raggedright\arraybackslash}.
COLUMN_SPECIFICATION = re.compile(r"(?:>\{[^}]*\})?(?:[lcr]|[pmb]\{[^}]*\})")

SPECIAL_CHARACTERS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape(text):
    """Return the text with all LaTeX special characters escaped."""
    return "".join(SPECIAL_CHARACTERS.get(character, character) for character in str(text))


def row(cells):
    return " & ".join(escape(cell) for cell in cells) + r" \\"


def header_cell(name, column, width):
    """A column header, set on several lines if it is longer than ``width`` characters.

    The lines are broken at spaces and aligned like the column. The header of a
    paragraph column is not broken, because LaTeX wraps it to the column width.
    """
    lines = [str(name)]
    if width is not None and column in ("l", "c", "r"):
        lines = textwrap.wrap(str(name), width, break_long_words=False, break_on_hyphens=False)
    if len(lines) == 1:
        return escape(lines[0])
    stacked = r" \\ ".join(escape(line) for line in lines)
    return rf"\begin{{tabular}}[b]{{@{{}}{column}@{{}}}}{stacked}\end{{tabular}}"


def tabular(table, align=None, group_column=None, header_width=None, long=False):
    """Return a booktabs ``tabular`` with one row per row of the data frame.

    ``align`` is the LaTeX column specification; the default aligns the first
    column left and all others right. If ``group_column`` is given, the group
    name is printed only in the first row of each run of consecutive rows with
    the same value, and groups are separated by additional vertical space.
    Column headers longer than ``header_width`` characters are set on several
    lines. If ``long`` is true, a ``longtable`` is returned instead, which
    breaks across pages and repeats the header on each page.
    All cells must be non-empty strings or numbers formatted by the caller.
    """
    if table.isna().any().any():
        raise ValueError("the table contains empty cells")
    if align is None:
        align = "l" + "r" * (table.shape[1] - 1)
    columns = COLUMN_SPECIFICATION.findall(align)
    if len(columns) != table.shape[1]:
        raise ValueError(f"align {align!r} specifies {len(columns)} columns, "
                         f"the table has {table.shape[1]}")
    group_index = None
    if group_column is not None:
        group_index = list(table.columns).index(group_column)

    environment = "tabular"
    if long:
        environment = "longtable"
    headers = [header_cell(name, column, header_width)
               for name, column in zip(table.columns, columns)]
    lines = [rf"\begin{{{environment}}}{{{align}}}", r"\toprule", " & ".join(headers) + r" \\",
             r"\midrule"]
    if long:
        lines += [r"\endhead", r"\bottomrule", r"\endfoot"]
    previous_group = None
    for index, values in enumerate(table.itertuples(index=False)):
        cells = list(values)
        if group_index is not None:
            group = cells[group_index]
            if index > 0 and group == previous_group:
                cells[group_index] = ""
            elif index > 0:
                lines.append(r"\addlinespace")
            previous_group = group
        lines.append(row(cells))
    if not long:
        lines.append(r"\bottomrule")
    lines.append(rf"\end{{{environment}}}")
    return "\n".join(lines) + "\n"

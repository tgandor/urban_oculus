"""
Useful for df.to_latex()
"""

import pandas as pd
import re
from IPython.display import Markdown


def _wrap(df, top, col, smallest, float_format):
    """If this was inlined, do_format would bind to the variables in the other function."""
    top_N = df[col].sort_values(ascending=smallest)[:top]
    top = set(top_N)
    tip = max(top_N)

    def do_format(value):
        # print(f'{value} in {top_N}?')
        if value == tip:
            return f'\\textbf{"{"}{float_format % value}{"}"}'
        elif value in top:
            return f'\\underline{"{"}{float_format % value}{"}"}'
        else:
            return float_format % value
    return do_format


def highlight_best_formatters(df, top=3, smallest=False, float_format='%.3f'):
    formatters = {}
    for col in df.columns:
        formatters[col] = _wrap(df, top, col, smallest, float_format)
    return formatters


def to_latex_topN(df: pd.DataFrame, top=3, print=print, **to_latex_kwargs):
    """The alignment of the output still fails when there are underscores, but..."""
    output = df.to_latex(formatters=highlight_best_formatters(df, top=top), **to_latex_kwargs
        ).replace('\\textbackslash ', '\\'
        ).replace('\\{', '{'
        ).replace('\\}', '}')
    if print:
        print(output)
        return None
    return output


def to_markdown_topN(df):
    max_mask = (df == df.max())
    md = df.to_markdown()
    lines = md.split('\n')
    new_lines = lines[:2]
    for line, (idx, mask_row) in zip(lines[2:], max_mask.iterrows()):
        fields = line.split('|')
        assert fields[1].strip() == idx
        if mask_row.any():
            new_fields = fields[:2]
            for field, ismax in zip(fields[2:], mask_row):
                new_fields.append(re.sub(r'\S+', lambda x: f'**{x.group()}**', field) if ismax else field)
            new_fields.append('')
            line = '|'.join(new_fields)
        new_lines.append(line)
        # print(line, mask_row)
    new_md = '\n'.join(new_lines)
    return Markdown(new_md)


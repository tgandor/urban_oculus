"""
Useful for df.to_latex()
"""


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


def to_latex_topN(df, top=3, print=print, **to_latex_kwargs):
    """The alignment of the output still fails when there are underscores, but..."""
    output = df.to_latex(formatters=highlight_best_formatters(df, top=top), **to_latex_kwargs
        ).replace('\\textbackslash ', '\\'           
        ).replace('\\{', '{'
        ).replace('\\}', '}')
    if print:
        print(output)
        return None
    return output

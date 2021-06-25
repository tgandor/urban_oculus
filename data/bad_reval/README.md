# Wrong rich results

Which were not taking into account repeated `img['dtMatches']` and `img['dtIgnore']` (in short: crowd objects).

Some symptoms: recall can be > 100%...

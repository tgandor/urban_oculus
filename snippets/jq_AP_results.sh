#!/bin/bash

jq -rj 'input_filename, " ", .results.bbox.AP, "\n"' "$@"

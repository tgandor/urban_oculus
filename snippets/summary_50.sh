#!/bin/bash

rm -f apendix_data.tex

for i in ../reval_50/[RX]* ; do
    model=`basename $i | sed 's/_/\\\\_/'`
    echo "\section{Model $model, \$T_c = 0.5\$}" | tee -a apendix_data.tex
    summary_table -g 2>&1 $i | tee -a apendix_data.tex
done

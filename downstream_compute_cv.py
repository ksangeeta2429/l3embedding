#!/usr/bin/env python3

import pandas as pd
import numpy as np
import csv
import sys

filename = sys.argv[1]
outfilename = filename.replace('csv', 'cv')

# TODO: Integrate with Google Sheets app
df = pd.read_table(filename, header=0, sep=',',
                   names=['Path', 'Fold', 'Acc'])

# Only taking rows with valid accuracy
df['Acc'].replace('', np.nan, inplace=True)
df.dropna(subset=['Acc'], inplace=True)

# Compute average accuracy, grouping by hyp
df_cv = df['Acc'].groupby(
    df['Path']).mean().to_frame()

# Save to groupby file
df_cv.to_csv(outfilename, sep="\t", quoting=csv.QUOTE_NONE)
#!/bin/bash

# join cells w/ atac
xsv join -d "\t" peak Hackathon2024.Training.Set.Peak2Gene.Pairs.txt peak Hackathon2024.ATAC.txt > merge1.csv
xsv join -d "\t" peak Hackathon2024.Testing.Set.Peak2Gene.Pairs.txt peak Hackathon2024.ATAC.txt > merge2.csv

# convert rna to comma separated
xsv cat rows -d "\t" Hackathon2024.RNA.txt > merge_rna.csv

# join cell-atac w/ rna
xsv join gene merge1.csv gene merge_rna.csv > train.csv
xsv join gene merge2.csv gene merge_rna.csv > test.csv

# remove merge artifacts
rm -f merge*
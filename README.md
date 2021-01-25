# Learn to rank paper annotation

Zotero was used to import DOIs and download associated papers and meta information.
Using Zotero simplifies the process but is not required.

## Requirements tools  

pdftotext from poppler-utils

## Requirements python 
* numpy
* pandas
* gensim
* sklearn

## Input data structure

Data structure follows export data format of Zotero.

### file papers.csv

CSV file with a header row with columns (order of columns does not mater).  
```
"File Attachments", "Item Type", "Publication Year", "Author", "Title", "Publication Title", "DOI", "Url"
```

### directory files 

Directory with PDF files of papers.
For example:

```
files/3640:
'Salonen et al_2012_The adult intestinal core microbiota is determined by analysis depth and health.pdf'

files/3642:
'Lahti et al_2013_Associations between the human intestinal microbiota, iLactobacillus.pdf'

files/3644:
'Faust et al_2015_Metagenomics meets time series analysis.pdf'
```

### files anotated_*.csv
Known files that match specified tag. One file per line.
Example `anotated_human.csv`:

```
files/3687/Shankar et al_2015_Using Bayesian modelling to investigate factors governing antibiotic-induced.pdf
files/3685/Shankar et al_2015_A systematic evaluation of high-dimensional, ensemble-based regression for.pdf
```

## Steps to reproduce
1. Run `read_texts.py` to convert PDF to TXT and construct data_clean.json
2. Run `learn_to_rank.py` to construct list of papers in order of relevance.

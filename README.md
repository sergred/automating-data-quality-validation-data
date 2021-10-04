# automating-data-quality-validation-data
This repository contains the FBPosts dataset that is used in the experimental setup of the paper "Automating Data Quality Validation for Dynamic Data Ingestion".
The tsv files are tab separated. ```FBPosts_dirty.tsv``` contains crawled Facebook posts. ```FBPosts_clean.tsv``` is a variant that was semi-automatically cleaned with the [OpenRefine](https://openrefine.org/) tool. The records that could not be cleaned were removed. The ```FBPosts_dirty_shortened.tsv``` contains the original records that could be cleaned in the ```FBPosts_clean.tsv```. The ```partitions/{clean/dirty}/FBPosts_{clean/dirty}_{idx}.tsv``` files contain the corresponding data partitions of week ```idx```, from 1 to 53 respectively.

# Demo
For a short demo, run ```pip install -r requirements.txt``` and ```python demo.py```.

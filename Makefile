# This makefile generates the www for the Neurosymbolic notebook
# Usage:
# make all: generates the www
# make clean: cleans the www and all generated files
# requires jq and jupyter-book

# Find all notebooks in the tutorial directory
NOTEBOOKS := $(wildcard tutorial/*.ipynb)
# Define corresponding output notebook paths in the www folder
WWW_NOTEBOOKS := $(NOTEBOOKS:tutorial/%.ipynb=www/%.ipynb)

all: www

transfer: www
	ghp-import -n -p -f www/_build/html

www: $(WWW_NOTEBOOKS) www/README.md
	jupyter-book build www

# Pattern rule for converting notebooks
www/%.ipynb: tutorial/%.ipynb
	jq -M 'del(.metadata.widgets)' $< > $@

www/README.md:
	cp README.md www/README.md

clean:
	rm -f $(WWW_NOTEBOOKS)
	rm -f www/README.md
	jupyter-book clean www --all

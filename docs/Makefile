MKDOCS ?= python -m mkdocs

.PHONY: help html serve clean

help:
	@echo "make html   Build the documentation"
	@echo "make serve  Start the local preview server"
	@echo "make clean  Remove generated documentation"

html:
	cd .. && $(MKDOCS) build --strict

serve:
	cd .. && $(MKDOCS) serve

clean:
	rm -rf _build/html
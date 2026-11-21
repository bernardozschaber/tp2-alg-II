MAIN = main
ENGINE ?= pdflatex

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib sections/*.tex
	$(ENGINE) -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	$(ENGINE) -interaction=nonstopmode $(MAIN).tex
	$(ENGINE) -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).pdf

.PHONY: all clean

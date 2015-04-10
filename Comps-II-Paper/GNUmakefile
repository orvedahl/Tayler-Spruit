# basename of main tex file
file = comps_2.tex

# make with a bibliography
biblio:=t

# bibliography file
bib = Bibliography.bib

# main tex file dependencies
filedepends = intro.tex appendix.tex shortcuts.tex

# where are the images stored
imgsource = Plots

# command to convert eps to pdf
EPStoPDF = epstopdf

# what are we making
#all: $(file:.tex=.pdf) $(file:.tex=.ps)
all: $(file:.tex=.pdf)

# list of all eps images
eps_source = $(wildcard $(imgsource)/*.eps)

# list of all eps images, but change the .eps to .pdf
pdf_source = $(eps_source:.eps=.pdf)

ifeq ($(biblio), t)
# make the pdf
$(file:.tex=.pdf): $(file) $(bib) $(filedepends) $(pdf_source)
	@pdflatex $(file) < /dev/null
	@bibtex $(basename $(file))
	@pdflatex $(file) < /dev/null
	@pdflatex $(file) < /dev/null
	@pdflatex $(file) < /dev/null
	@echo
	@echo
	@echo "---SUCCESS---"
	@echo

# make the dvi
$(file:.tex=.dvi): $(file) $(bib) $(filedepends) $(eps_source)
	@latex $(file) < /dev/null
	@bibtex $(basename $(file))
	@latex $(file) < /dev/null
	@latex $(file) < /dev/null
	@latex $(file) < /dev/null
	@echo
	@echo
	@echo "---SUCCESS---"
	@echo
else
# make the pdf without bibtex
$(file:.tex=.pdf): $(file) $(filedepends) $(pdf_source)
	@pdflatex $(file) < /dev/null
	@pdflatex $(file) < /dev/null
	@pdflatex $(file) < /dev/null
	@pdflatex $(file) < /dev/null
	@echo
	@echo
	@echo "---SUCCESS---"
	@echo

# make the dvi without bibtex
$(file:.tex=.dvi): $(file) $(filedepends) $(eps_source)
	@latex $(file) < /dev/null
	@latex $(file) < /dev/null
	@latex $(file) < /dev/null
	@latex $(file) < /dev/null
	@echo
	@echo
	@echo "---SUCCESS---"
	@echo
endif

# way to make only the pdf (default is dvi, ps and pdf)
pdf: $(file:.tex=.pdf)

# make the ps
%.ps: %.dvi
	dvips -t letter -o $@ $<

# convert eps to pdf
%.pdf: %.eps
	$(EPStoPDF) $<

# remove only the final pdf
rmd:
	-rm $(file:.tex=.pdf)

# display the final pdf
show:
	@evince $(file:.tex=.pdf)

.PHONY: clean realclean

RM = -rm -f
clean:
	$(RM) $(pdf_source) $(file:.tex=.dvi)
	$(RM) $(file:.tex=.blg) $(file:.tex=.log)
	$(RM) $(file:.tex=.bbl) $(file:.tex=.aux)
	$(RM) $(file:.tex=.ps)

realclean: clean
	$(RM) $(file:.tex=.pdf)


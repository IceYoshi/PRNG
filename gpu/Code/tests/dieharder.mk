
$(RESULT_DIR)/dieharder-%: $(RND_BIN)/rnd_to_stdout-%
	@echo
	@echo testing random number quality of $<
	@echo ======================================================================
	bash -c 'set -o pipefail; set -e; \
	$< | $(DIE_HARDER_BIN) $(DIE_HARDER_FLAGS) | tee $@' || (rm $@; false)
	@echo
	@echo


dieharder-%: $(RESULT_DIR)/dieharder-%
	@true

dieharder-all: $(RANDOM_NUMBER_GENERATORS:%=dieharder-%)
	@true

report-dieharder.tex: lib/dieharder_graph.sh $(RANDOM_NUMBER_GENERATORS:%=$(RESULT_DIR)/dieharder-%)
	lib/dieharder_graph.sh $(RANDOM_NUMBER_GENERATORS:%=$(RESULT_DIR)/dieharder-%) > $@

report-dieharder.pdf: report-dieharder.tex
	pdflatex $<

report-dieharder.html: lib/dieharder_graph_html.sh
	lib/dieharder_graph_html.sh $(RANDOM_NUMBER_GENERATORS:%=$(RESULT_DIR)/dieharder-%) > $@

tests += dieharder

reports += report-dieharder.tex report-dieharder.pdf report-dieharder.html

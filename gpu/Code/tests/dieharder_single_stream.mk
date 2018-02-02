
$(RESULT_DIR)/dieharder_single_stream-%: $(RND_BIN)/rnd_to_stdout_single_stream-%
	@echo
	@echo testing random number quality of $<
	@echo ======================================================================
	bash -c 'set -o pipefail; set -e; \
	$< | $(DIE_HARDER_BIN) $(DIE_HARDER_FLAGS) | tee $@' || (rm $@; false)
	@echo
	@echo


dieharder_single_stream-%: $(RESULT_DIR)/dieharder_single_stream-%
	@true

dieharder_single_stream-all: $(RANDOM_NUMBER_GENERATORS:%=dieharder_single_stream-%)
	@true

report-dieharder_single_stream.tex: lib/dieharder_graph.sh $(RANDOM_NUMBER_GENERATORS:%=$(RESULT_DIR)/dieharder_single_stream-%)
	lib/dieharder_single_stream_graph.sh $(RANDOM_NUMBER_GENERATORS:%=$(RESULT_DIR)/dieharder_single_stream-%) > $@

report-dieharder_single_stream.pdf: report-dieharder_single_stream.tex
	pdflatex $<

report-dieharder_single_stream.html: lib/dieharder_graph_html.sh
	lib/dieharder_graph_html.sh $(RANDOM_NUMBER_GENERATORS:%=$(RESULT_DIR)/dieharder_single_stream-%) > $@

tests += dieharder_single_stream

reports += report-dieharder_single_stream.tex report-dieharder_single_stream.pdf report-dieharder_single_stream.html

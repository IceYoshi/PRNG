
$(RESULT_DIR)/bench_raw_performance-%: $(RND_BIN)/bench_raw_performance-% lib/cuda_info
	@echo benchmarking $<
	(lib/cuda_info; echo "date: `date`") | sed 's/^/# /' > $@
	bash -c '\
	set -o pipefail; \
	set -e; \
	for i in `seq 1 $(RAW_PERFORMANCE_NUM_ROUNDS)`; do  \
		$< | tee -a $@ ; \
	done ' || (rm $@; false)

$(RND_BIN)/bench_raw_performance-%: \
	NVCCFLAGS += \
		-DRAW_PERFORMANCE_NUM_KERNEL_CALLS="$(RAW_PERFORMANCE_NUM_KERNEL_CALLS)" \
		-DRAW_PERFORMANCE_NUM_THREADS="$(RAW_PERFORMANCE_NUM_THREADS)" \
		-DRAW_PERFORMANCE_NUM_RANDOMS_PER_THREAD="$(RAW_PERFORMANCE_NUM_RANDOMS_PER_THREAD)"

bench_raw_performance-%: $(RESULT_DIR)/bench_raw_performance-%
	@true

bench_raw_performance-all: $(RANDOM_NUMBER_GENERATORS:%=bench_raw_performance-%)
	@true

report-raw_performance.data: $(RANDOM_NUMBER_GENERATORS:%=$(RESULT_DIR)/bench_raw_performance-%)
	TEST_PREFIX="$(RESULT_DIR)/bench_raw_performance-" lib/report_performance.sh $^ > $@


_raw_performance_num_randoms = \
	$(shell bash -c "echo $$(( $(RAW_PERFORMANCE_NUM_KERNEL_CALLS) \
							* $(RAW_PERFORMANCE_NUM_THREADS) \
							* $(RAW_PERFORMANCE_NUM_RANDOMS_PER_THREAD) ))")

_raw_performance_gfx_card = $(shell lib/gfx_name.sh $(RANDOM_NUMBER_GENERATORS:%=$(RESULT_DIR)/bench_raw_performance-%))

report-raw_performance.ps: report-raw_performance.data lib/perf.plt
	bash -c "\
		echo DATA=\'$<\'; \
		echo OUTPUT=\'$@\'; \
		echo set title \'Raw Performance, $(_raw_performance_gfx_card), $(_raw_performance_num_randoms) Random Numbers\' ;\
		cat lib/perf.plt" \
	| gnuplot

tests += bench_raw_performance

reports += report-raw_performance.ps

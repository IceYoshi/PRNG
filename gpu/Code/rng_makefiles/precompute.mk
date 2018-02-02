# this file is auto generated, do not edit!

build_targets += $(RND_BIN)/bench_asian_options-precompute

$(RND_BIN)/bench_asian_options-precompute: stubs/bench_asian_options.cu $(RND_DIR)/precompute.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"precompute.hpp\" 	-DRNG=precompute $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_asian_options-precompute 2>&1 	|| (cat $(BUILD_LOGS)/bench_asian_options-precompute; false)
	@cat $(BUILD_LOGS)/bench_asian_options-precompute
	@echo
	@echo

build_targets += $(RND_BIN)/bench_raw_performance-precompute

$(RND_BIN)/bench_raw_performance-precompute: stubs/bench_raw_performance.cu $(RND_DIR)/precompute.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"precompute.hpp\" 	-DRNG=precompute $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_raw_performance-precompute 2>&1 	|| (cat $(BUILD_LOGS)/bench_raw_performance-precompute; false)
	@cat $(BUILD_LOGS)/bench_raw_performance-precompute
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout-precompute

$(RND_BIN)/rnd_to_stdout-precompute: stubs/rnd_to_stdout.cu $(RND_DIR)/precompute.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"precompute.hpp\" 	-DRNG=precompute $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout-precompute 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout-precompute; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout-precompute
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout_single_stream-precompute

$(RND_BIN)/rnd_to_stdout_single_stream-precompute: stubs/rnd_to_stdout_single_stream.cu $(RND_DIR)/precompute.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"precompute.hpp\" 	-DRNG=precompute $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout_single_stream-precompute 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-precompute; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-precompute
	@echo
	@echo


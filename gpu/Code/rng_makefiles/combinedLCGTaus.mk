# this file is auto generated, do not edit!

build_targets += $(RND_BIN)/bench_asian_options-combinedLCGTaus

$(RND_BIN)/bench_asian_options-combinedLCGTaus: stubs/bench_asian_options.cu $(RND_DIR)/combinedLCGTaus.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"combinedLCGTaus.hpp\" 	-DRNG=combinedLCGTaus $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_asian_options-combinedLCGTaus 2>&1 	|| (cat $(BUILD_LOGS)/bench_asian_options-combinedLCGTaus; false)
	@cat $(BUILD_LOGS)/bench_asian_options-combinedLCGTaus
	@echo
	@echo

build_targets += $(RND_BIN)/bench_raw_performance-combinedLCGTaus

$(RND_BIN)/bench_raw_performance-combinedLCGTaus: stubs/bench_raw_performance.cu $(RND_DIR)/combinedLCGTaus.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"combinedLCGTaus.hpp\" 	-DRNG=combinedLCGTaus $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_raw_performance-combinedLCGTaus 2>&1 	|| (cat $(BUILD_LOGS)/bench_raw_performance-combinedLCGTaus; false)
	@cat $(BUILD_LOGS)/bench_raw_performance-combinedLCGTaus
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout-combinedLCGTaus

$(RND_BIN)/rnd_to_stdout-combinedLCGTaus: stubs/rnd_to_stdout.cu $(RND_DIR)/combinedLCGTaus.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"combinedLCGTaus.hpp\" 	-DRNG=combinedLCGTaus $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout-combinedLCGTaus 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout-combinedLCGTaus; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout-combinedLCGTaus
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout_single_stream-combinedLCGTaus

$(RND_BIN)/rnd_to_stdout_single_stream-combinedLCGTaus: stubs/rnd_to_stdout_single_stream.cu $(RND_DIR)/combinedLCGTaus.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"combinedLCGTaus.hpp\" 	-DRNG=combinedLCGTaus $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout_single_stream-combinedLCGTaus 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-combinedLCGTaus; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-combinedLCGTaus
	@echo
	@echo


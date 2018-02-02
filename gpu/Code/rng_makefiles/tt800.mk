# this file is auto generated, do not edit!

build_targets += $(RND_BIN)/bench_asian_options-tt800

$(RND_BIN)/bench_asian_options-tt800: stubs/bench_asian_options.cu $(RND_DIR)/tt800.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"tt800.hpp\" 	-DRNG=tt800 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_asian_options-tt800 2>&1 	|| (cat $(BUILD_LOGS)/bench_asian_options-tt800; false)
	@cat $(BUILD_LOGS)/bench_asian_options-tt800
	@echo
	@echo

build_targets += $(RND_BIN)/bench_raw_performance-tt800

$(RND_BIN)/bench_raw_performance-tt800: stubs/bench_raw_performance.cu $(RND_DIR)/tt800.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"tt800.hpp\" 	-DRNG=tt800 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_raw_performance-tt800 2>&1 	|| (cat $(BUILD_LOGS)/bench_raw_performance-tt800; false)
	@cat $(BUILD_LOGS)/bench_raw_performance-tt800
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout-tt800

$(RND_BIN)/rnd_to_stdout-tt800: stubs/rnd_to_stdout.cu $(RND_DIR)/tt800.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"tt800.hpp\" 	-DRNG=tt800 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout-tt800 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout-tt800; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout-tt800
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout_single_stream-tt800

$(RND_BIN)/rnd_to_stdout_single_stream-tt800: stubs/rnd_to_stdout_single_stream.cu $(RND_DIR)/tt800.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"tt800.hpp\" 	-DRNG=tt800 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout_single_stream-tt800 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-tt800; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-tt800
	@echo
	@echo


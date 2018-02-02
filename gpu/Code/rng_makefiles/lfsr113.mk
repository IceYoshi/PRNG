# this file is auto generated, do not edit!

build_targets += $(RND_BIN)/bench_asian_options-lfsr113

$(RND_BIN)/bench_asian_options-lfsr113: stubs/bench_asian_options.cu $(RND_DIR)/lfsr113.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"lfsr113.hpp\" 	-DRNG=lfsr113 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_asian_options-lfsr113 2>&1 	|| (cat $(BUILD_LOGS)/bench_asian_options-lfsr113; false)
	@cat $(BUILD_LOGS)/bench_asian_options-lfsr113
	@echo
	@echo

build_targets += $(RND_BIN)/bench_raw_performance-lfsr113

$(RND_BIN)/bench_raw_performance-lfsr113: stubs/bench_raw_performance.cu $(RND_DIR)/lfsr113.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"lfsr113.hpp\" 	-DRNG=lfsr113 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_raw_performance-lfsr113 2>&1 	|| (cat $(BUILD_LOGS)/bench_raw_performance-lfsr113; false)
	@cat $(BUILD_LOGS)/bench_raw_performance-lfsr113
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout-lfsr113

$(RND_BIN)/rnd_to_stdout-lfsr113: stubs/rnd_to_stdout.cu $(RND_DIR)/lfsr113.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"lfsr113.hpp\" 	-DRNG=lfsr113 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout-lfsr113 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout-lfsr113; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout-lfsr113
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout_single_stream-lfsr113

$(RND_BIN)/rnd_to_stdout_single_stream-lfsr113: stubs/rnd_to_stdout_single_stream.cu $(RND_DIR)/lfsr113.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"lfsr113.hpp\" 	-DRNG=lfsr113 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout_single_stream-lfsr113 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-lfsr113; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-lfsr113
	@echo
	@echo


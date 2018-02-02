# this file is auto generated, do not edit!

build_targets += $(RND_BIN)/bench_asian_options-kiss07

$(RND_BIN)/bench_asian_options-kiss07: stubs/bench_asian_options.cu $(RND_DIR)/kiss07.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"kiss07.hpp\" 	-DRNG=kiss07 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_asian_options-kiss07 2>&1 	|| (cat $(BUILD_LOGS)/bench_asian_options-kiss07; false)
	@cat $(BUILD_LOGS)/bench_asian_options-kiss07
	@echo
	@echo

build_targets += $(RND_BIN)/bench_raw_performance-kiss07

$(RND_BIN)/bench_raw_performance-kiss07: stubs/bench_raw_performance.cu $(RND_DIR)/kiss07.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"kiss07.hpp\" 	-DRNG=kiss07 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_raw_performance-kiss07 2>&1 	|| (cat $(BUILD_LOGS)/bench_raw_performance-kiss07; false)
	@cat $(BUILD_LOGS)/bench_raw_performance-kiss07
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout-kiss07

$(RND_BIN)/rnd_to_stdout-kiss07: stubs/rnd_to_stdout.cu $(RND_DIR)/kiss07.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"kiss07.hpp\" 	-DRNG=kiss07 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout-kiss07 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout-kiss07; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout-kiss07
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout_single_stream-kiss07

$(RND_BIN)/rnd_to_stdout_single_stream-kiss07: stubs/rnd_to_stdout_single_stream.cu $(RND_DIR)/kiss07.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"kiss07.hpp\" 	-DRNG=kiss07 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout_single_stream-kiss07 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-kiss07; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-kiss07
	@echo
	@echo


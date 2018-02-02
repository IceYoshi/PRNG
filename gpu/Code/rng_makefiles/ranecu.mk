# this file is auto generated, do not edit!

build_targets += $(RND_BIN)/bench_asian_options-ranecu

$(RND_BIN)/bench_asian_options-ranecu: stubs/bench_asian_options.cu $(RND_DIR)/ranecu.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"ranecu.hpp\" 	-DRNG=ranecu $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_asian_options-ranecu 2>&1 	|| (cat $(BUILD_LOGS)/bench_asian_options-ranecu; false)
	@cat $(BUILD_LOGS)/bench_asian_options-ranecu
	@echo
	@echo

build_targets += $(RND_BIN)/bench_raw_performance-ranecu

$(RND_BIN)/bench_raw_performance-ranecu: stubs/bench_raw_performance.cu $(RND_DIR)/ranecu.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"ranecu.hpp\" 	-DRNG=ranecu $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_raw_performance-ranecu 2>&1 	|| (cat $(BUILD_LOGS)/bench_raw_performance-ranecu; false)
	@cat $(BUILD_LOGS)/bench_raw_performance-ranecu
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout-ranecu

$(RND_BIN)/rnd_to_stdout-ranecu: stubs/rnd_to_stdout.cu $(RND_DIR)/ranecu.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"ranecu.hpp\" 	-DRNG=ranecu $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout-ranecu 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout-ranecu; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout-ranecu
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout_single_stream-ranecu

$(RND_BIN)/rnd_to_stdout_single_stream-ranecu: stubs/rnd_to_stdout_single_stream.cu $(RND_DIR)/ranecu.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"ranecu.hpp\" 	-DRNG=ranecu $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout_single_stream-ranecu 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-ranecu; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-ranecu
	@echo
	@echo


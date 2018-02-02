# this file is auto generated, do not edit!

build_targets += $(RND_BIN)/bench_asian_options-md5

$(RND_BIN)/bench_asian_options-md5: stubs/bench_asian_options.cu $(RND_DIR)/md5.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"md5.hpp\" 	-DRNG=md5 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_asian_options-md5 2>&1 	|| (cat $(BUILD_LOGS)/bench_asian_options-md5; false)
	@cat $(BUILD_LOGS)/bench_asian_options-md5
	@echo
	@echo

build_targets += $(RND_BIN)/bench_raw_performance-md5

$(RND_BIN)/bench_raw_performance-md5: stubs/bench_raw_performance.cu $(RND_DIR)/md5.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"md5.hpp\" 	-DRNG=md5 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_raw_performance-md5 2>&1 	|| (cat $(BUILD_LOGS)/bench_raw_performance-md5; false)
	@cat $(BUILD_LOGS)/bench_raw_performance-md5
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout-md5

$(RND_BIN)/rnd_to_stdout-md5: stubs/rnd_to_stdout.cu $(RND_DIR)/md5.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"md5.hpp\" 	-DRNG=md5 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout-md5 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout-md5; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout-md5
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout_single_stream-md5

$(RND_BIN)/rnd_to_stdout_single_stream-md5: stubs/rnd_to_stdout_single_stream.cu $(RND_DIR)/md5.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"md5.hpp\" 	-DRNG=md5 $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout_single_stream-md5 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-md5; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-md5
	@echo
	@echo


# this file is auto generated, do not edit!

build_targets += $(RND_BIN)/bench_asian_options-park_miller

$(RND_BIN)/bench_asian_options-park_miller: stubs/bench_asian_options.cu $(RND_DIR)/park_miller.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"park_miller.hpp\" 	-DRNG=park_miller $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_asian_options-park_miller 2>&1 	|| (cat $(BUILD_LOGS)/bench_asian_options-park_miller; false)
	@cat $(BUILD_LOGS)/bench_asian_options-park_miller
	@echo
	@echo

build_targets += $(RND_BIN)/bench_raw_performance-park_miller

$(RND_BIN)/bench_raw_performance-park_miller: stubs/bench_raw_performance.cu $(RND_DIR)/park_miller.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"park_miller.hpp\" 	-DRNG=park_miller $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_raw_performance-park_miller 2>&1 	|| (cat $(BUILD_LOGS)/bench_raw_performance-park_miller; false)
	@cat $(BUILD_LOGS)/bench_raw_performance-park_miller
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout-park_miller

$(RND_BIN)/rnd_to_stdout-park_miller: stubs/rnd_to_stdout.cu $(RND_DIR)/park_miller.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"park_miller.hpp\" 	-DRNG=park_miller $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout-park_miller 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout-park_miller; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout-park_miller
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout_single_stream-park_miller

$(RND_BIN)/rnd_to_stdout_single_stream-park_miller: stubs/rnd_to_stdout_single_stream.cu $(RND_DIR)/park_miller.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"park_miller.hpp\" 	-DRNG=park_miller $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout_single_stream-park_miller 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-park_miller; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-park_miller
	@echo
	@echo


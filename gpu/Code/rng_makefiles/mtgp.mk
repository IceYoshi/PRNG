# this file is auto generated, do not edit!

build_targets += $(RND_BIN)/bench_asian_options-mtgp

$(RND_BIN)/bench_asian_options-mtgp: stubs/bench_asian_options.cu $(RND_DIR)/mtgp.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"mtgp.hpp\" 	-DRNG=mtgp $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_asian_options-mtgp 2>&1 	|| (cat $(BUILD_LOGS)/bench_asian_options-mtgp; false)
	@cat $(BUILD_LOGS)/bench_asian_options-mtgp
	@echo
	@echo

build_targets += $(RND_BIN)/bench_raw_performance-mtgp

$(RND_BIN)/bench_raw_performance-mtgp: stubs/bench_raw_performance.cu $(RND_DIR)/mtgp.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"mtgp.hpp\" 	-DRNG=mtgp $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_raw_performance-mtgp 2>&1 	|| (cat $(BUILD_LOGS)/bench_raw_performance-mtgp; false)
	@cat $(BUILD_LOGS)/bench_raw_performance-mtgp
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout-mtgp

$(RND_BIN)/rnd_to_stdout-mtgp: stubs/rnd_to_stdout.cu $(RND_DIR)/mtgp.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"mtgp.hpp\" 	-DRNG=mtgp $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout-mtgp 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout-mtgp; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout-mtgp
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout_single_stream-mtgp

$(RND_BIN)/rnd_to_stdout_single_stream-mtgp: stubs/rnd_to_stdout_single_stream.cu $(RND_DIR)/mtgp.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"mtgp.hpp\" 	-DRNG=mtgp $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout_single_stream-mtgp 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-mtgp; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-mtgp
	@echo
	@echo


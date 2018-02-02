# this file is auto generated, do not edit!

build_targets += $(RND_BIN)/bench_asian_options-tea

$(RND_BIN)/bench_asian_options-tea: stubs/bench_asian_options.cu $(RND_DIR)/tea.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"tea.hpp\" 	-DRNG=tea $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_asian_options-tea 2>&1 	|| (cat $(BUILD_LOGS)/bench_asian_options-tea; false)
	@cat $(BUILD_LOGS)/bench_asian_options-tea
	@echo
	@echo

build_targets += $(RND_BIN)/bench_raw_performance-tea

$(RND_BIN)/bench_raw_performance-tea: stubs/bench_raw_performance.cu $(RND_DIR)/tea.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"tea.hpp\" 	-DRNG=tea $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_raw_performance-tea 2>&1 	|| (cat $(BUILD_LOGS)/bench_raw_performance-tea; false)
	@cat $(BUILD_LOGS)/bench_raw_performance-tea
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout-tea

$(RND_BIN)/rnd_to_stdout-tea: stubs/rnd_to_stdout.cu $(RND_DIR)/tea.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"tea.hpp\" 	-DRNG=tea $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout-tea 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout-tea; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout-tea
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout_single_stream-tea

$(RND_BIN)/rnd_to_stdout_single_stream-tea: stubs/rnd_to_stdout_single_stream.cu $(RND_DIR)/tea.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"tea.hpp\" 	-DRNG=tea $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout_single_stream-tea 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-tea; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-tea
	@echo
	@echo


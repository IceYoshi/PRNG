# this file is auto generated, do not edit!

build_targets += $(RND_BIN)/bench_asian_options-drand48gpu

$(RND_BIN)/bench_asian_options-drand48gpu: stubs/bench_asian_options.cu $(RND_DIR)/drand48gpu.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"drand48gpu.hpp\" 	-DRNG=drand48gpu $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_asian_options-drand48gpu 2>&1 	|| (cat $(BUILD_LOGS)/bench_asian_options-drand48gpu; false)
	@cat $(BUILD_LOGS)/bench_asian_options-drand48gpu
	@echo
	@echo

build_targets += $(RND_BIN)/bench_raw_performance-drand48gpu

$(RND_BIN)/bench_raw_performance-drand48gpu: stubs/bench_raw_performance.cu $(RND_DIR)/drand48gpu.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"drand48gpu.hpp\" 	-DRNG=drand48gpu $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/bench_raw_performance-drand48gpu 2>&1 	|| (cat $(BUILD_LOGS)/bench_raw_performance-drand48gpu; false)
	@cat $(BUILD_LOGS)/bench_raw_performance-drand48gpu
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout-drand48gpu

$(RND_BIN)/rnd_to_stdout-drand48gpu: stubs/rnd_to_stdout.cu $(RND_DIR)/drand48gpu.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"drand48gpu.hpp\" 	-DRNG=drand48gpu $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout-drand48gpu 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout-drand48gpu; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout-drand48gpu
	@echo
	@echo

build_targets += $(RND_BIN)/rnd_to_stdout_single_stream-drand48gpu

$(RND_BIN)/rnd_to_stdout_single_stream-drand48gpu: stubs/rnd_to_stdout_single_stream.cu $(RND_DIR)/drand48gpu.hpp config.mk
	$(NVCC) -o $@ $< -DRANDOM_NUMBER_GENERATOR=\"drand48gpu.hpp\" 	-DRNG=drand48gpu $(NVCCFLAGS) $(WARNINGS) $(LIBS)  	--ptxas-options=-v > $(BUILD_LOGS)/rnd_to_stdout_single_stream-drand48gpu 2>&1 	|| (cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-drand48gpu; false)
	@cat $(BUILD_LOGS)/rnd_to_stdout_single_stream-drand48gpu
	@echo
	@echo


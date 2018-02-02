# main makefile. 

-include config.mk
-include local.mk

NVCCFLAGS += -I$(RND_DIR) $(CUDA_INCLUDES) $(CUDA_LIBS) $(CUDA_LIB_PATHS)

# this will create the mandatory directories
_MAKE_DIRECTORIES := $(shell mkdir $(RND_BIN) $(RESULT_DIR) $(RND_MAKEFILE_DIR) $(BUILD_LOGS) 2> /dev/null)

usage help:
	@echo "make [TEST-(RNG|all)]*"
	@echo
	@echo "tests are:"
	@echo $(tests) | tr ' ' '\n' | sed 's/^/	/'
	@echo
	@echo "random number generators are:"
	@echo $(RANDOM_NUMBER_GENERATORS) | tr ' ' '\n' | sed 's/^/	/'
	@echo
	@echo "reports are:"
	@echo $(reports) | tr ' ' '\n' | sed 's/^/	/'

clean:
	rm -rf $(RND_MAKEFILE_DIR) $(RND_BIN) $(BUILD_LOGS)


# generate makefiles for all stubs
$(RND_MAKEFILE_DIR)/%.mk: ./lib/generate_makefile.sh $(TEST_DIR)
	@STUB_DIR=$(STUB_DIR) ./lib/generate_makefile.sh `basename $@ .mk` > $@

# includes all test descriptions
-include $(TEST_DIR)/*.mk

# includes makefile that describe how to build the different RND benchmark programs
-include $(RANDOM_NUMBER_GENERATORS:%=$(RND_MAKEFILE_DIR)/%.mk)

# this target will all binaries. build_targets variable is set by generate_makefile.sh 
lib/cuda_info: lib/cuda_info.cu config.mk 
	$(NVCC) -o $@ $< $(NVCCFLAGS) $(WARNINGS) $(LIBS)

build_all: $(build_targets) lib/cuda_info


.PHONY: usage clean help

# this is needed to prevent the deletion of intermediate files (for quality-%
# and bench-% rules). 
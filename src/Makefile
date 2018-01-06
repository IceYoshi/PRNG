##################################################################################
# Makefile - Configuration file for GNU make (http://www.gnu.org/software/make/)
# Creation : 08 Feb 2015
# Time-stamp: <Dim 2015-02-08 23:15 svarrette>
#
# Copyright (c) 2015 Sebastien Varrette <Sebastien.Varrette@uni.lu>
#
# Available Commands
# ------------------
# make           : Compile files, binaries are generated in the current directory
# make force     : Force the complete re-compilation, even if not needed
# make clean     : Remove backup files (*~) and other generated files
#
############################## Variables Declarations ############################

BUILD_DIR = ../build

all:
	$(MAKE) -C $(BUILD_DIR)

%:
	$(MAKE) -C $(BUILD_DIR) $@


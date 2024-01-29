HOMEDIR := $(realpath $(dir $(word $(words $(MAKEFILE_LIST)), $(MAKEFILE_LIST))))
# This little trick finds where this makefile exists.
$(info HOMEDIR set to $(HOMEDIR))

# Set to 1 in command line to enable debug options.
debug ?= 0

CXX = g++
AR = ar
RANLIB = ranlib

SRC_EXTS = .cpp .cxx .c
HDR_EXTS = .hpp .hxx .h
LIB_NAME = libgraphgaslite

LIBRARY = $(HOMEDIR)/lib/$(LIB_NAME).a

# gcc 4.8 bug, fixed by -Wl,--no-as-needed, see http://stackoverflow.com/a/19463892
MT_OPTS = -pthread -Wl,--no-as-needed

CXX_LANG = -Wall -Wextra -pedantic -std=c++11 $(MT_OPTS)
CXX_OPT  = -O3 -funroll-loops -march=native -flto
CXX_DBG  = -g -fkeep-inline-functions
ifeq ($(debug), 1)
CXX_COMPILE = $(CXX_DBG)
else
CXX_COMPILE = $(CXX_OPT)
endif
INC_OPTS = -I$(HOMEDIR)/include
LIB_OPTS =

# User-supply flags through command line arguments
CL_CXXFLAGS =
CL_LDFLAGS  =

CXXFLAGS = $(CXX_LANG) $(CXX_COMPILE) $(INC_OPTS) $(CL_CXXFLAGS)
LDFLAGS  = $(CXX_LANG) $(CXX_COMPILE) $(LIB_OPTS) $(CL_LDFLAGS)


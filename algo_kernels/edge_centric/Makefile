
include ../../makefile.inc

SRC_DIR = .
BIN_DIR = bin

ifdef MKLPATH
HAS_LAPACK = 1
CXXFLAGS += -DUSE_MKL -DMKL_ILP64 -m64 \
            -I$(MKLPATH)/include
LDFLAGS += -L$(MKLPATH)/lib/intel64 \
           -Wl,--start-group -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -Wl,--end-group -lm -ldl \
           -Wl,-rpath $(MKLPATH)/lib/intel64
else \
ifdef ATLASPATH
HAS_LAPACK = 1
CXXFLAGS += -DUSE_ATLAS \
            -I$(ATLASPATH)/include
LDFLAGS += -L$(ATLASPATH)/lib \
           -Wl,--start-group -llapack -lcblas -latlas -Wl,--end-group
endif


APPS = \
	   pagerank \
	   sssp \


ifeq ($(HAS_LAPACK),1)
APPS += als
endif

default: $(addprefix $(BIN_DIR)/,$(APPS))

include ../common_harness/makefile.harness

$(foreach APP,$(APPS),$(eval $(call compile_rule,$(APP))))


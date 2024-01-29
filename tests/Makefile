#
# Makefile for building unit tests with Google Test.
#

include ../makefile.inc

## Google Test setting.
GTEST_REPO = ../googletest
GTEST_BASE = $(GTEST_REPO)/googletest
GTEST_PATH = $(GTEST_BASE)/include
GTEST_LIB  = /usr/local/lib/libgtest.a /usr/local/lib/libgtest_main.a

## Google Test additional flags.
CXX_DEBUG = -g
CXXFLAGS = $(CXX_LANG) $(CXX_DEBUG) $(INC_OPTS) $(CL_CXXFLAGS) \
	-I$(GTEST_PATH)
LDFLAGS  = $(CXX_LANG) $(CXX_DEBUG) $(LIB_OPTS) $(CL_LDFLAGS) \
	$(GTEST_LIB) -lpthread


## Tests
TESTS += \
		 comm_sync \
		 graph \
		 engine \



## Test build rules
BIN_DIR = bin
BINS = $(foreach TEST,$(TESTS),$(BIN_DIR)/$(TEST))
OBJS = $(foreach TEST,$(TESTS),$(TEST).o)
DEPS = $(OBJS:.o=.d)

all: $(GTEST_LIB) $(BINS)

# $(BIN_DIR) is an order-only prerequisite
define compile_rule
$(BIN_DIR)/$1: $1.o | $(BIN_DIR)
	$$(CXX) $$< -o $$@ $$(CXXFLAGS) $$(LDFLAGS)
endef
$(foreach TEST,$(TESTS),$(eval $(call compile_rule,$(TEST))))

# .cpp.o and .c.o means the same as %.o:%.cpp and %.o:%.c
$(SRC_EXTS:=.o):
	$(CXX) $< -c -o $@ $(CXXFLAGS) -MP -MMD

$(BIN_DIR):
	@mkdir -p $@

clean:
	rm -f $(OBJS) $(DEPS) $(BINS) *~
	rm -rf $(BIN_DIR)


## Google Test setup and clean
$(GTEST_LIB):
#	@if [ ! -d "$(GTEST_REPO)" ]; then git clone git@github.com-CBackyx:google/googletest.git $(GTEST_REPO); fi
# 	cd $(GTEST_BASE)/make; CXXFLAGS=-std=c++11 make
	cd $(GTEST_BASE)/

clean_gtest:
	rm -rf $(GTEST_REPO)


## Run tests
run_all: $(GTEST_LIB) $(BINS)
	@for b in `ls $(BIN_DIR)/*`; do $$b || exit; done


.PHONY: clean clean_gtest run_all

-include $(DEPS)


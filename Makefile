
default: edge_kernels

all: default tests

.PHONY: edge_kernels tests run_tests clean

edge_kernels:
	$(MAKE) -C algo_kernels/edge_centric --no-print-directory


tests:
	$(MAKE) -C tests --no-print-directory

run_tests:
	$(MAKE) run_all -C tests --no-print-directory

clean:
	$(MAKE) clean -C algo_kernels/edge_centric --no-print-directory
	$(MAKE) clean -C tests --no-print-directory


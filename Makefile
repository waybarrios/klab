CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Werror

.PHONY: all
all: bin/klab native/build/libkernel_lab.so

bin/klab: native/cli.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -DKLAB_PROJECT_ROOT='"$(CURDIR)"' $< -o $@.new
	mv $@.new $@

native/build/libkernel_lab.so: native/driver.cpp
	@mkdir -p native/build
	$(CXX) $(CXXFLAGS) -fPIC -shared $< -ldl -pthread -o $@.new
	mv $@.new $@

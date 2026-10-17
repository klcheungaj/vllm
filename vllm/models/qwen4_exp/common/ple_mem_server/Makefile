# PLE memory server and benchmark. Needs only a C++17 compiler and UCX.
#   make                      # UCX from pkg-config
#   make UCX_HOME=/opt/ucx    # UCX from an install prefix
#   make OUT=/tmp/ple         # put the binaries elsewhere

CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
OUT ?= .

ifdef UCX_HOME
UCX_CFLAGS := -I$(UCX_HOME)/include
UCX_LIBS := -L$(UCX_HOME)/lib -Wl,-rpath,$(UCX_HOME)/lib -lucp -luct -lucs -lucm
else
UCX_CFLAGS := $(shell pkg-config --cflags ucx 2>/dev/null)
UCX_LIBS := $(shell pkg-config --libs ucx 2>/dev/null)
UCX_LIBDIR := $(shell pkg-config --variable=libdir ucx 2>/dev/null)
ifeq ($(UCX_LIBS),)
UCX_LIBS := -lucp -luct -lucs -lucm
else ifneq ($(UCX_LIBDIR),)
UCX_LIBS += -Wl,-rpath,$(UCX_LIBDIR)
endif
endif

HEADERS := ple_mem_protocol.h ple_mem_util.h ple_mem_client.h
BINS := $(OUT)/ple_mem_server $(OUT)/ple_mem_bench

all: $(BINS)

$(OUT)/ple_mem_server: ple_mem_server.cpp $(HEADERS)
	@mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $(UCX_CFLAGS) $< -o $@ $(UCX_LIBS) $(LDFLAGS)

$(OUT)/ple_mem_bench: ple_mem_bench.cpp $(HEADERS)
	@mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $(UCX_CFLAGS) $< -o $@ $(UCX_LIBS) $(LDFLAGS)

clean:
	rm -f $(BINS)

.PHONY: all clean

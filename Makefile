ifndef PREFIX
	PREFIX = test
endif

ifndef SIZE
	SIZE = 1000
endif

PYTHON = $(shell which python3)

# FILE := all_pairs
FILE := fast_pairs
PAIRS := $(basename $(shell cat $(FILE)))

default: knn

allall: knn all

print:
	@echo $(PAIRS)

.SECONDEXPANSION:

knn: $(addsuffix .knn, $(PAIRS))

all: $(addsuffix .all, $(PAIRS))

$(PAIRS): $$(addsuffix .knn, $$@) $$(addsuffix .all, $$@)

$(addsuffix .knn, $(PAIRS)): prefix
	mkdir -p $(PREFIX)/$(basename $@)
	cd $(PREFIX)/$(basename $@) &&\
	$(MAKE) -f ../../Makefile-pair knn PAIR=$(addsuffix .txt, $(basename $@)) SIZE=$(SIZE)

$(addsuffix .all, $(PAIRS)): prefix
	mkdir -p $(PREFIX)/$(basename $@)
	cd $(PREFIX)/$(basename $@) &&\
	$(MAKE) -f ../../Makefile-pair all PAIR=$(addsuffix .txt, $(basename $@)) SIZE=$(SIZE)

$(addsuffix .dirs, $(PAIRS)): prefix
	mkdir -p $(PREFIX)/$(basename $@)
	
prefix:
	mkdir -p $(PREFIX)

.PHONY: clean $(addsuffix .clean, $(PAIRS))

clean:
	rm -rf $(PREFIX)

$(addsuffix .clean, $(PAIRS)):
	rm -rf $(PREFIX)/$(basename $@)/*

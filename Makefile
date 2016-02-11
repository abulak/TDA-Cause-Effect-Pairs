ifndef PREFIX
	PREFIX = test
endif

ifndef SIZE
	SIZE = 1000
endif

ifndef BINS
	BINS = 0
endif

PYTHON = $(shell which python3)

#FILE := SIM_pairs
FILE := CEP_pairs_fast
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
	cp ./Makefile-pair $(PREFIX)/$(basename $@)/Makefile
	$(MAKE) -C $(PREFIX)/$(basename $@) knn PAIR=$(addsuffix .txt, $(basename $@)) SIZE=$(SIZE) BINS=$(BINS)

$(addsuffix .all, $(PAIRS)): prefix
	mkdir -p $(PREFIX)/$(basename $@)
	cd $(PREFIX)/$(basename $@) &&\
	$(MAKE) -f ../../Makefile-pair all PAIR=$(addsuffix .txt, $(basename $@)) SIZE=$(SIZE) BINS=$(BINS)

$(addsuffix .dirs, $(PAIRS)): prefix
	mkdir -p $(PREFIX)/$(basename $@)
	
prefix:
	mkdir -p $(PREFIX)
	cp ./pairs/pairmeta.txt ./$(PREFIX)/

.PHONY: clean $(addsuffix .clean, $(PAIRS))

clean:
	rm -rf $(PREFIX)

$(addsuffix .clean, $(PAIRS)):
	rm -rf $(PREFIX)/$(basename $@)/*

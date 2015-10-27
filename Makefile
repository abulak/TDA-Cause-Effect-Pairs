ifndef PREFIX
	PREFIX = test
endif

ifndef SIZE
	SIZE = 1000
endif

PYTHON = $(shell which python3)

FILE := pairs-to-process
PAIRS := $(basename $(shell cat $(FILE)))

all: $(PAIRS)

print:
	@echo $(PAIRS)

.SECONDEXPANSION:

$(PAIRS): $$(addsuffix .knn, $$@) $$(addsuffix .all, $$@)

$(addsuffix .knn, $(PAIRS)): prefix
	mkdir -p $(PREFIX)/$(basename $@)
	cd $(PREFIX)/$(basename $@) &&\
	$(MAKE) -f ../../Makefile-pair knn PAIR=$(addsuffix .txt, $(basename $@)) SIZE=$(SIZE)

$(addsuffix .all, $(PAIRS)): prefix
	mkdir -p $(PREFIX)/$(basename $@)
	cd $(PREFIX)/$(basename $@) &&\
	$(MAKE) -f ../../Makefile-pair all PAIR=$(addsuffix .txt, $(basename $@)) SIZE=$(SIZE)

prefix:
	mkdir -p $(PREFIX)

.PHONY: clean $(addsuffix .clean, $(PAIRS))

clean:
	rm -rf $(PREFIX)

$(addsuffix .clean, $(PAIRS)):
	rm -rf $(PREFIX)/$(basename $@)/*
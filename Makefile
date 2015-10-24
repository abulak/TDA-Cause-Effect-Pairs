ifndef PREFIX
	PREFIX = test
endif
PAIRS_DIR = ./pairs
PYTHON = /usr/bin/python3

# TXT_FILES_IN_PAIRS = $(wildcard $(PAIRS_DIR)/*.txt)
# PAIRS = $(basename $(notdir $(filter-out $(wildcard $(PAIRS_DIR)/*e*), $(TXT_FILES_IN_PAIRS))))

FILE := pairs-to-process
PAIRS := $(basename $(shell cat $(FILE)))

all: $(addsuffix .pair, $(PAIRS))

print:
	@echo $(PAIRS)

%.pair: %.persistence_pairs
	
%.persistence_pairs: %.outliers
	$(PYTHON) ./TDA.py $(PREFIX) $(basename $(basename $@))

%.outliers: prefix
	mkdir -p $(PREFIX)/$(basename $@)
	$(PYTHON) ./identify-outliers.py $(PREFIX) $(addsuffix .txt, $(basename $@))

prefix:
	mkdir -p $(PREFIX)

.PHONY: clean
clean:
	rm -rf $(PREFIX)
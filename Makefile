ifndef SIZE
	SIZE = 2000
endif

ifndef QUANTISE
	QUANTISE = 0
endif

FILE := pairs_to_process
PAIRS := $(shell cat $(FILE))

.PHONY: clean $(addsuffix .clean, $(PAIRS)) $(PAIRS)

all: $(PAIRS)

$(PAIRS):
	mkdir -p $@
	cp ./Makefile-pair $(basename $@)/Makefile
	$(MAKE) -C $@ SIZE=$(SIZE) QUANTISE=$(QUANTISE)

.SECONDEXPANSION:



clean:
	rm -rf $(PAIRS)

$(addsuffix .clean, $(PAIRS)):
	@rm $(basename $@)/*

print:
	@echo $(PAIRS)

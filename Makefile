SRC      = src
BUILD    = build
TOOLS    = tools
ROM      = $(BUILD)/snes_gpt.sfc

CA65     = ca65
LD65     = ld65
PYTHON   = python3
CA65FLAGS = --cpu 65816 -I $(SRC) -I $(BUILD) --bin-include-dir $(BUILD)

SRCS     = main.asm gpt.asm math.asm vector.asm inference.asm
OBJS     = $(SRCS:%.asm=$(BUILD)/%.o)
GEN_OBJS = $(BUILD)/tables.o $(BUILD)/font.o

# Build ROM
$(ROM): $(OBJS) $(GEN_OBJS) $(SRC)/lorom.cfg | $(BUILD)
	$(LD65) -C $(SRC)/lorom.cfg -o $@ $(OBJS) $(GEN_OBJS)

# Assemble source files
$(BUILD)/%.o: $(SRC)/%.asm $(SRC)/snes.inc | $(BUILD)
	$(CA65) $(CA65FLAGS) -o $@ $<

# Assemble generated files
$(BUILD)/tables.o: $(BUILD)/tables.asm $(SRC)/snes.inc | $(BUILD)
	$(CA65) $(CA65FLAGS) -o $@ $<

$(BUILD)/font.o: $(BUILD)/font.asm | $(BUILD)
	$(CA65) $(CA65FLAGS) -o $@ $<

# main.asm needs weights for .incbin
$(BUILD)/main.o: $(BUILD)/trained_weights_q8x8.bin

# Generate lookup tables
$(BUILD)/tables.asm: $(TOOLS)/gen_tables.py | $(BUILD)
	$(PYTHON) $< $(BUILD)

# Generate font data
$(BUILD)/font.asm: $(TOOLS)/gen_font.py | $(BUILD)
	$(PYTHON) $< $(BUILD)

# Train model and export weights
$(BUILD)/trained_weights_q8x8.bin: $(TOOLS)/export_weights.py | $(BUILD)
	$(PYTHON) $< $(BUILD)

$(BUILD):
	mkdir -p $(BUILD)

.PHONY: clean all
all: $(ROM)

clean:
	rm -rf $(BUILD)

"""
Generate a minimal 2bpp 8x8 font for SNES Mode 0.
Covers: space, '.', 0-9, A-Z, a-z (mapped to uppercase glyphs).
Outputs font.asm with tile data.

SNES 2bpp format: for each 8-pixel row, 2 bytes (bitplane 0, bitplane 1).
16 bytes per tile. We use color 3 (both bitplanes set) for "on" pixels
and color 0 for "off" pixels, so bp0 = bp1 = pixel pattern.
"""


def char_to_pixels(ch):
    """Return 8 rows of 8-pixel patterns for a character.
    Each row is an 8-bit integer where bit 7 = leftmost pixel."""

    # Minimal 5x7 font bitmaps (in a 8x8 cell, offset 1px from left)
    # Format: list of 8 integers (top to bottom), each 8 bits wide
    font_5x7 = {
        ' ': [
            0b00000000,
            0b00000000,
            0b00000000,
            0b00000000,
            0b00000000,
            0b00000000,
            0b00000000,
            0b00000000,
        ],
        '.': [
            0b00000000,
            0b00000000,
            0b00000000,
            0b00000000,
            0b00000000,
            0b01100000,
            0b01100000,
            0b00000000,
        ],
        '0': [
            0b01110000,
            0b10001000,
            0b10011000,
            0b10101000,
            0b11001000,
            0b10001000,
            0b01110000,
            0b00000000,
        ],
        '1': [
            0b00100000,
            0b01100000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b01110000,
            0b00000000,
        ],
        '2': [
            0b01110000,
            0b10001000,
            0b00001000,
            0b00110000,
            0b01000000,
            0b10000000,
            0b11111000,
            0b00000000,
        ],
        '3': [
            0b01110000,
            0b10001000,
            0b00001000,
            0b00110000,
            0b00001000,
            0b10001000,
            0b01110000,
            0b00000000,
        ],
        '4': [
            0b00010000,
            0b00110000,
            0b01010000,
            0b10010000,
            0b11111000,
            0b00010000,
            0b00010000,
            0b00000000,
        ],
        '5': [
            0b11111000,
            0b10000000,
            0b11110000,
            0b00001000,
            0b00001000,
            0b10001000,
            0b01110000,
            0b00000000,
        ],
        '6': [
            0b01110000,
            0b10000000,
            0b11110000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b01110000,
            0b00000000,
        ],
        '7': [
            0b11111000,
            0b00001000,
            0b00010000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b00000000,
        ],
        '8': [
            0b01110000,
            0b10001000,
            0b10001000,
            0b01110000,
            0b10001000,
            0b10001000,
            0b01110000,
            0b00000000,
        ],
        '9': [
            0b01110000,
            0b10001000,
            0b10001000,
            0b01111000,
            0b00001000,
            0b00001000,
            0b01110000,
            0b00000000,
        ],
        'A': [
            0b01110000,
            0b10001000,
            0b10001000,
            0b11111000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b00000000,
        ],
        'B': [
            0b11110000,
            0b10001000,
            0b10001000,
            0b11110000,
            0b10001000,
            0b10001000,
            0b11110000,
            0b00000000,
        ],
        'C': [
            0b01110000,
            0b10001000,
            0b10000000,
            0b10000000,
            0b10000000,
            0b10001000,
            0b01110000,
            0b00000000,
        ],
        'D': [
            0b11110000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b11110000,
            0b00000000,
        ],
        'E': [
            0b11111000,
            0b10000000,
            0b10000000,
            0b11110000,
            0b10000000,
            0b10000000,
            0b11111000,
            0b00000000,
        ],
        'F': [
            0b11111000,
            0b10000000,
            0b10000000,
            0b11110000,
            0b10000000,
            0b10000000,
            0b10000000,
            0b00000000,
        ],
        'G': [
            0b01110000,
            0b10001000,
            0b10000000,
            0b10111000,
            0b10001000,
            0b10001000,
            0b01110000,
            0b00000000,
        ],
        'H': [
            0b10001000,
            0b10001000,
            0b10001000,
            0b11111000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b00000000,
        ],
        'I': [
            0b01110000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b01110000,
            0b00000000,
        ],
        'J': [
            0b00111000,
            0b00010000,
            0b00010000,
            0b00010000,
            0b00010000,
            0b10010000,
            0b01100000,
            0b00000000,
        ],
        'K': [
            0b10001000,
            0b10010000,
            0b10100000,
            0b11000000,
            0b10100000,
            0b10010000,
            0b10001000,
            0b00000000,
        ],
        'L': [
            0b10000000,
            0b10000000,
            0b10000000,
            0b10000000,
            0b10000000,
            0b10000000,
            0b11111000,
            0b00000000,
        ],
        'M': [
            0b10001000,
            0b11011000,
            0b10101000,
            0b10101000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b00000000,
        ],
        'N': [
            0b10001000,
            0b11001000,
            0b10101000,
            0b10011000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b00000000,
        ],
        'O': [
            0b01110000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b01110000,
            0b00000000,
        ],
        'P': [
            0b11110000,
            0b10001000,
            0b10001000,
            0b11110000,
            0b10000000,
            0b10000000,
            0b10000000,
            0b00000000,
        ],
        'Q': [
            0b01110000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b10101000,
            0b10010000,
            0b01101000,
            0b00000000,
        ],
        'R': [
            0b11110000,
            0b10001000,
            0b10001000,
            0b11110000,
            0b10100000,
            0b10010000,
            0b10001000,
            0b00000000,
        ],
        'S': [
            0b01110000,
            0b10001000,
            0b10000000,
            0b01110000,
            0b00001000,
            0b10001000,
            0b01110000,
            0b00000000,
        ],
        'T': [
            0b11111000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b00000000,
        ],
        'U': [
            0b10001000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b01110000,
            0b00000000,
        ],
        'V': [
            0b10001000,
            0b10001000,
            0b10001000,
            0b10001000,
            0b01010000,
            0b01010000,
            0b00100000,
            0b00000000,
        ],
        'W': [
            0b10001000,
            0b10001000,
            0b10001000,
            0b10101000,
            0b10101000,
            0b11011000,
            0b10001000,
            0b00000000,
        ],
        'X': [
            0b10001000,
            0b10001000,
            0b01010000,
            0b00100000,
            0b01010000,
            0b10001000,
            0b10001000,
            0b00000000,
        ],
        'Y': [
            0b10001000,
            0b10001000,
            0b01010000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b00100000,
            0b00000000,
        ],
        'Z': [
            0b11111000,
            0b00001000,
            0b00010000,
            0b00100000,
            0b01000000,
            0b10000000,
            0b11111000,
            0b00000000,
        ],
    }

    return font_5x7.get(ch.upper(), font_5x7.get(ch, font_5x7[' ']))


def pixels_to_2bpp(rows):
    """Convert 8 rows of pixel data to SNES 2bpp tile format.
    Returns 16 bytes: for each row, (bitplane0, bitplane1)."""
    result = []
    for row in rows:
        # For a solid color (color 3), both bitplanes = same pattern
        bp0 = row
        bp1 = row
        result.append(bp0)
        result.append(bp1)
    return result


def main():
    lines = []
    lines.append("; ==========================================================================")
    lines.append("; Font Tile Data for SNES GPT — Generated by gen_font.py")
    lines.append("; ==========================================================================")
    lines.append("; 2bpp 8x8 tiles for SNES Mode 0, BG1")
    lines.append("; 16 bytes per tile (8 rows x 2 bitplanes)")
    lines.append("; Tile layout: tile 0 = space, tile 1 = '.', tiles 2-27 = A-Z")
    lines.append("; (lowercase maps to uppercase glyphs)")
    lines.append(";")
    lines.append('; Tile indices for display:')
    lines.append(';   space=0, "."=1, A=2, B=3, ... Z=27')
    lines.append(';   0=28, 1=29, 2=30, ... 9=37')
    lines.append(';   For hex: A-F reuse tiles 2-7')
    lines.append("")
    lines.append(".p816")
    lines.append('.include "snes.inc"')
    lines.append("")
    lines.append(".segment \"RODATA\"")
    lines.append("")
    lines.append(".export FontTiles")
    lines.append("FontTiles:")

    # Tile order: space, '.', A-Z, 0-9
    tile_chars = [' ', '.'] + [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [chr(c) for c in range(ord('0'), ord('9') + 1)]

    for idx, ch in enumerate(tile_chars):
        pixels = char_to_pixels(ch)
        tile_bytes = pixels_to_2bpp(pixels)
        label = f"'{ch}'" if ch != ' ' else "'SP'"
        lines.append(f"    ; Tile {idx}: {label}")
        for row in range(8):
            b0 = tile_bytes[row * 2]
            b1 = tile_bytes[row * 2 + 1]
            lines.append(f"    .byte ${b0:02X}, ${b1:02X}")
    lines.append("")

    # Export the tile count
    lines.append(f"FONT_TILE_COUNT = {len(tile_chars)}")
    lines.append(f"FONT_TILE_BYTES = {len(tile_chars) * 16}")
    lines.append("")

    import sys
    import os
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    outpath = os.path.join(outdir, "font.asm")
    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {outpath} with {len(tile_chars)} tiles ({len(tile_chars) * 16} bytes)")


if __name__ == "__main__":
    main()

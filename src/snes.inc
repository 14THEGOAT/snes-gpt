; ==========================================================================
; SNES Hardware Register Definitions
; ==========================================================================

; --- PPU Registers ---
INIDISP  = $2100    ; Display control (bit 7 = force blank, bits 3-0 = brightness)
OBSEL    = $2101    ; Object size and character data address
OAMADDL  = $2102    ; OAM address (low)
OAMADDH  = $2103    ; OAM address (high)
OAMDATA  = $2104    ; OAM data write
BGMODE   = $2105    ; BG mode and character size
MOSAIC   = $2106    ; Mosaic size and enable
BG1SC    = $2107    ; BG1 tilemap address and size
BG2SC    = $2108    ; BG2 tilemap address and size
BG3SC    = $2109    ; BG3 tilemap address and size
BG4SC    = $210A    ; BG4 tilemap address and size
BG12NBA  = $210B    ; BG1/BG2 character data address
BG34NBA  = $210C    ; BG3/BG4 character data address
BG1HOFS  = $210D    ; BG1 horizontal scroll
BG1VOFS  = $210E    ; BG1 vertical scroll
BG2HOFS  = $210F    ; BG2 horizontal scroll
BG2VOFS  = $2110    ; BG2 vertical scroll
BG3HOFS  = $2111    ; BG3 horizontal scroll
BG3VOFS  = $2112    ; BG3 vertical scroll
BG4HOFS  = $2113    ; BG4 horizontal scroll
BG4VOFS  = $2114    ; BG4 vertical scroll
VMAIN    = $2115    ; VRAM address increment mode
VMADDL   = $2116    ; VRAM address (low)
VMADDH   = $2117    ; VRAM address (high)
VMDATAL  = $2118    ; VRAM data write (low)
VMDATAH  = $2119    ; VRAM data write (high)
CGADD    = $2121    ; CGRAM (palette) address
CGDATA   = $2122    ; CGRAM data write
W12SEL   = $2123    ; Window 1/2 mask for BG1/BG2
W34SEL   = $2124    ; Window 1/2 mask for BG3/BG4
WOBJSEL  = $2125    ; Window 1/2 mask for OBJ/Color
WH0      = $2126    ; Window 1 left position
WH1      = $2127    ; Window 1 right position
WH2      = $2128    ; Window 2 left position
WH3      = $2129    ; Window 2 right position
WBGLOG   = $212A    ; Window mask logic for BG
WOBJLOG  = $212B    ; Window mask logic for OBJ/Color
TM       = $212C    ; Main screen layer enable
TS       = $212D    ; Sub screen layer enable
TMW      = $212E    ; Main screen window mask
TSW      = $212F    ; Sub screen window mask
CGWSEL   = $2130    ; Color math control A
CGADSUB  = $2131    ; Color math control B
COLDATA  = $2132    ; Fixed color data
SETINI   = $2133    ; Screen mode / video select

; --- CPU I/O Registers ---
NMITIMEN = $4200    ; NMI, V/H count, and joypad enable
WRIO     = $4201    ; Programmable I/O port (output)
WRMPYA   = $4202    ; Hardware multiply operand A (8-bit unsigned)
WRMPYB   = $4203    ; Hardware multiply operand B (8-bit unsigned, triggers multiply)
WRDIVL   = $4204    ; Hardware dividend (low byte)
WRDIVH   = $4205    ; Hardware dividend (high byte)
WRDIVB   = $4206    ; Hardware divisor (triggers divide)
HTIMEL   = $4207    ; H count timer (low)
HTIMEH   = $4208    ; H count timer (high)
VTIMEL   = $4209    ; V count timer (low)
VTIMEH   = $420A    ; V count timer (high)
MDMAEN   = $420B    ; DMA enable (per channel)
HDMAEN   = $420C    ; HDMA enable (per channel)
MEMSEL   = $420D    ; FastROM enable

; --- CPU I/O Registers (Read) ---
RDNMI    = $4210    ; NMI flag and CPU version
TIMEUP   = $4211    ; IRQ flag
HVBJOY   = $4212    ; H/V blank and joypad status
RDIO     = $4213    ; Programmable I/O port (input)
RDDIVL   = $4214    ; Division result (low)
RDDIVH   = $4215    ; Division result (high)
RDMPYL   = $4216    ; Multiply result (low byte)
RDMPYH   = $4217    ; Multiply result (high byte)
JOY1L    = $4218    ; Joypad 1 (low byte)
JOY1H    = $4219    ; Joypad 1 (high byte)
JOY2L    = $421A    ; Joypad 2 (low byte)
JOY2H    = $421B    ; Joypad 2 (high byte)

; --- DMA Channel 0 Registers ---
DMAP0    = $4300    ; DMA control
BBAD0    = $4301    ; DMA destination register
A1T0L    = $4302    ; DMA source address (low)
A1T0H    = $4303    ; DMA source address (high)
A1B0     = $4304    ; DMA source bank
DAS0L    = $4305    ; DMA size (low)
DAS0H    = $4306    ; DMA size (high)

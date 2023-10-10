# flake8: noqa

# STL 파일 이름
STL_FILENAME: str = "specimen5.stl"
# 필라멘트 옵션
FAN_SPEEDS = (0,)  # (FS) 출력 도중 팬의 작동량, ABS는 0%, PLA는 100%
BED_TEMPERATURES = (60, 80, 100)  # (BT) 베드의 온도
EXTRUDER_TEMPERATURES = (
    250,
    255,
    260,
)  # (ET) 압출기의 온도, PLA의 경우 (200, 240), ABS의 경우 (245, 255)
# infill_overlaps = ("15%", "35%")  # (IO) 보류!

# 프린팅 옵션
LAYER_THICKNESSES = (0.1, 0.15, 0.2)  # (LT) z축 레이어의 두께, 0.1mm, 0.2mm
INFILL_SPEEDS = (60,)  # (IS) 인필 프린팅 속도, 50mm/s, 70mm/s
TRAVEL_SPEED = 150  # 헤드가 이동하는 속도, 150mm/s
FILL_DENSITY = "100%"  # 인필의 채움 밀도, 예비실험에서 20%로 뽑았고, 100%로 뽑아야 할듯
FILL_PATTERN = (
    "rectilinear"  # 인필의 채움 패턴, 예비실험에서 grid로 뽑았고, rectilinear로 뽑아야 할듯
)

# 프린터 옵션
PRINTER_NAME = "ultimaker2"
START_GCODE = """
; Printer_Settings_ID: [printer_settings_id]

; # # # # # # START Header

G21 		; metric values
G90 		; absolute positioning
M82 		; set extruder to absolute mode
M107 		; start with the fan off

M140 S[first_layer_bed_temperature]	; start bed heating

G28 X0 Y0 Z0	; move X/Y/Z to endstops
G1 X1 Y6 F15000	; move X/Y to start position
G1 Z35 F9000	; move Z to start position

; Wait for bed and nozzle temperatures
M190 S{first_layer_bed_temperature[0] - 5}	; wait for bed temperature - 5
M140 S[first_layer_bed_temperature]	; continue bed heating
M109 S[first_layer_temperature]	; wait for nozzle temperature

; Purge and prime sequence
M83                ; set extruder to relative mode
G92 E0             ; reset extrusion distance
G0 X0 Y1 F10000    ; starting position
G1 F150 E20        ; compress the bowden tube
G1 E-8 F1200       ; retract filament by 8mm
G0 X50 Y1 F5000    ; move to beginning of purge line avoiding the clip
G0 F1200 Z{first_layer_height/2} ; Adjust Z height
G1 E8 F1200        ; prime extruder

; Purge sequence (start from (50, 1)))
G1 X150 E8 F1000   ; intro line from x=50 to 150mm, y=1
G1 Y1.3 F1000    ; Move up a little to y=1.3
G1 X50 E8 F1000    ; intro line from x=150 to 50mm
G1 Y50 F1000       ; Move down to y=50, avoiding the clip
G1 X1 F1000     ; Move left to x=1
G1 Y150 E8 F1000      ; Move down to y=150
G1 X1.3 F1000   ; Move a little to the right to x=1.3
G1 Y50 E8 F1000       ; Move up to y=50, completing the purge

G92 E0             ; reset extruder reference
M82                ; set extruder to absolute mode
; # # # # # # END Header
"""
START_GCODE = repr(START_GCODE.strip()).removeprefix("'").removesuffix("'")


# # flake8: noqa

# # STL 파일 이름
# STL_FILENAME: str = "specimen3.stl"
# # 필라멘트 옵션
# FAN_SPEEDS = (100,)  # (FS) 출력 도중 팬의 작동량, ABS는 0%, PLA는 100%
# BED_TEMPERATURES = (60, 80, 100)  # (BT) 베드의 온도
# EXTRUDER_TEMPERATURES = (
#     200,
#     220,
#     240,
# )  # (ET) 압출기의 온도, PLA의 경우 (200, 240), ABS의 경우 (245, 255)
# # infill_overlaps = ("15%", "35%")  # (IO) 보류!

# # 프린팅 옵션
# LAYER_THICKNESSES = (0.1, 0.15, 0.2)  # (LT) z축 레이어의 두께, 0.1mm, 0.2mm
# INFILL_SPEEDS = (60,)  # (IS) 인필 프린팅 속도, 50mm/s, 70mm/s
# TRAVEL_SPEED = 150  # 헤드가 이동하는 속도, 150mm/s
# FILL_DENSITY = "100%"  # 인필의 채움 밀도, 예비실험에서 20%로 뽑았고, 100%로 뽑아야 할듯
# FILL_PATTERN = (
#     "rectilinear"  # 인필의 채움 패턴, 예비실험에서 grid로 뽑았고, rectilinear로 뽑아야 할듯
# )

# # 프린터 옵션
# PRINTER_NAME = "ultimaker2"
# START_GCODE = """
# ; Printer_Settings_ID: [printer_settings_id]

# ; # # # # # # START Header

# G21 		; metric values
# G90 		; absolute positioning
# M82 		; set extruder to absolute mode
# M107 		; start with the fan off

# M140 S[first_layer_bed_temperature]	; start bed heating

# G28 X0 Y0 Z0	; move X/Y/Z to endstops
# G1 X1 Y6 F15000	; move X/Y to start position
# G1 Z35 F9000	; move Z to start position

# ; Wait for bed and nozzle temperatures
# M190 S{first_layer_bed_temperature[0] - 5}	; wait for bed temperature - 5
# M140 S[first_layer_bed_temperature]	; continue bed heating
# M109 S[first_layer_temperature]	; wait for nozzle temperature

# ; Purge and prime sequence
# M83                ; set extruder to relative mode
# G92 E0             ; reset extrusion distance
# G0 X0 Y1 F10000    ; starting position
# G1 F150 E20        ; compress the bowden tube
# G1 E-8 F1200       ; retract filament by 8mm
# G0 X50 Y1 F5000    ; move to beginning of purge line avoiding the clip
# G0 F1200 Z{first_layer_height/2} ; Adjust Z height
# G1 E8 F1200        ; prime extruder

# ; Purge sequence (start from (50, 1)))
# G1 X150 E8 F1000   ; intro line from x=50 to 150mm, y=1
# G1 Y1.3 F1000    ; Move up a little to y=1.3
# G1 X50 E8 F1000    ; intro line from x=150 to 50mm
# G1 Y50 F1000       ; Move down to y=50, avoiding the clip
# G1 X1 F1000     ; Move left to x=1
# G1 Y150 E8 F1000      ; Move down to y=150
# G1 X1.3 F1000   ; Move a little to the right to x=1.3
# G1 Y50 E8 F1000       ; Move up to y=50, completing the purge

# G92 E0             ; reset extruder reference
# M82                ; set extruder to absolute mode
# ; # # # # # # END Header
# """
# START_GCODE = repr(START_GCODE.strip()).removeprefix("'").removesuffix("'")

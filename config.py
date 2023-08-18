# flake8: noqa

# STL 파일 이름
STL_FILENAME: str = "specimen.stl"

# 필라멘트 옵션
FAN_SPEEDS = (0, 100)  # (FS) 출력 도중 팬의 작동량, ABS는 0%, PLA는 100%
BED_TEMPERATURES = (60, 80)  # (BT) 베드의 온도
EXTRUDER_TEMPERATURES = (
    200,
    240,
)  # (ET) 압출기의 온도, PLA의 경우 (200, 240), ABS의 경우 (235, 255)
# infill_overlaps = ("15%", "35%")  # (IO) 보류!

# 프린팅 옵션
LAYER_THICKNESSES = (0.1, 0.2)  # (LT) z축 레이어의 두께, 0.1mm, 0.2mm
INFILL_SPEEDS = (50, 70)  # (IS) 인필 프린팅 속도, 50mm/s, 70mm/s
FILL_DENSITY = "20%"  # 인필의 채움 밀도, 예비실험에서 20%로 뽑았고, 100%로 뽑아야 할듯
FILL_PATTERN = "grid"  # 인필의 채움 패턴, 예비실험에서 grid로 뽑았고, rectilinear로 뽑아야 할듯

# 프린터 옵션
PRINTER_NAME = "ultimaker2"
START_GCODE = """
; Printer_Settings_ID: [printer_settings_id]

; # # # # # # START Header
G21        ; metric values
G90        ; absolute positioning
M82        ; set extruder to absolute mode
M107       ; start with the fan off

M140 S[first_layer_bed_temperature] ; start bed heating

G28 X0 Y0 Z0 ; move X/Y/Z to endstops
G1 X1 Y6 F15000 ; move X/Y to start position
G1 Z35 F9000 ; move Z to start position

; Wait for bed and nozzle temperatures
M190 S{first_layer_bed_temperature[0] - 5} ; wait for bed temperature - 5
M140 S[first_layer_bed_temperature] ; continue bed heating
M109 S[first_layer_temperature] ; wait for nozzle temperature

; Purge and prime
M83        ; set extruder to relative mode
G92 E0     ; reset extrusion distance
G0 X0 Y1 F10000
G1 F150 E20 ; compress the bowden tube
G1 E-8 F1200
G0 X30 Y1 F5000
G0 F1200 Z{first_layer_height/2} ; Cut the connection to priming blob
G0 X100 F10000 ; disconnect with the prime blob
G0 X50         ; Avoid the metal clip holding the Ultimaker glass plate
G0 Z0.2 F720
G1 E8 F1200
G1 X80 E3 F1000  ; intro line 1
G1 X110 E4 F1000 ; intro line 2
G1 X140 F600    ; drag filament to decompress bowden tube
G1 X100 F3200   ; wipe backwards a bit
G1 X150 F3200   ; back to where there is no plastic: avoid dragging
G92 E0      ; reset extruder reference
M82        ; set extruder to absolute mode

; Nozzle cleaning sequence
G92 E0
G1 Z2.0 F3000 ; Move Z Axis up little to prevent scratching of Heat Bed
G1 X2.1 Y20 Z0.3 F5000.0 ; Move to start position
G1 X2.1 Y200.0 Z0.3 F1500.0 E15 ; Draw the first line
G1 X2.4 Y200.0 Z0.3 F5000.0 ; Move to side a little
G1 X2.4 Y20 Z0.3 F1500.0 E30 ; Draw the second line
G92 E0
; # # # # # # END Header
"""
START_GCODE = repr(START_GCODE.strip()).removeprefix("'").removesuffix("'")

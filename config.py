# flake8: noqa

# STL 파일 이름
STL_FILENAME: str = "specimen.stl"

# 필라멘트 옵션
FAN_SPEEDS = (0, 100)  # 출력 도중 팬의 작동량, ABS는 0%, PLA는 100%
BED_TEMPERATURES = (60, 80)  # 베드의 온도
EXTRUDER_TEMPERATURES = (200, 240)  # 압출기의 온도, PLA의 경우 (200, 240), ABS의 경우 (235, 255)
# infill_overlaps = ("15%", "35%")  # 보류

# 프린팅 옵션
LAYER_THICKNESSES = (0.1, 0.2)
INFILL_SPEEDS = (50, 70)

ANNInputParams = [
    "bedtemp",
    "exttemp",
    "layerthickness",
    "infillspeed",
    "density",
    "thermalresistance",
    "impactstrength",
    "glasstransitiontemp",
    "thermalconductivity",
    "linearthermalexpansioncoefficient",
]

ANNOutputParams = [
    # "weight",
    # "width1",
    # "width2",
    # "width3",
    # "height",
    # "depth",
    "strength",
    "lengthavg",
]


# _ANN_INPUT_PARAM_ARGS = get_args(ANNInputParams)
# ANN_INPUT_PARAM_INDICES = (
#     _ANN_INPUT_PARAM_ARGS.index("bedtemp"),
#     _ANN_INPUT_PARAM_ARGS.index("exttemp"),
#     _ANN_INPUT_PARAM_ARGS.index("layerthickness"),
#     _ANN_INPUT_PARAM_ARGS.index("infillspeed"),
#     _ANN_INPUT_PARAM_ARGS.index("density"),
#     _ANN_INPUT_PARAM_ARGS.index("thermalresistance"),
#     _ANN_INPUT_PARAM_ARGS.index("impactstrength"),
#     _ANN_INPUT_PARAM_ARGS.index("glasstransitiontemp"),
#     _ANN_INPUT_PARAM_ARGS.index("thermalconductivity"),
#     _ANN_INPUT_PARAM_ARGS.index("linearthermalexpansioncoefficient"),
# )

LSTMInputParams = ["strain"]
LSTMOutputParams = ["stress"]

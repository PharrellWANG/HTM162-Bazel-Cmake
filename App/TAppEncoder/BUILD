cc_library(
    name = "TAppEncCfg",
    srcs = ["TAppEncCfg.cpp"],
    hdrs = ["TAppEncCfg.h"],
    deps = [
        "//HTM162/Lib/TLibCommon:TLibCommon",
    ],
)

cc_library(
    name = "TAppEncTop",
    srcs = ["TAppEncTop.cpp"],
    hdrs = ["TAppEncTop.h"],
    deps = [
        ":TAppEncCfg",
        "//HTM162/Lib/TLibCommon:TLibCommon",
    ],
)

cc_binary(
    name = "TAppEncoder",
    srcs = ["encmain.cpp"],
    deps = [
    	":TAppEncCfg",
    	":TAppEncTop",
        "//HTM162/Lib/TLibCommon:TLibCommon",
    ],
)
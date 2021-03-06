cmake_minimum_required(VERSION 3.8)
project(HTM162_TAppEncoder)
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# config for single individual classifier: start
set(SOURCE_FILES App/TAppClassifier/main.cpp App/TAppClassifier/strtk.hpp)
add_executable(TAppClassifier ${SOURCE_FILES})
find_package(TensorflowCC COMPONENTS Shared)
target_link_libraries(TAppClassifier TensorflowCC::Shared)
# config for single individual classifier: end

set(SOURCE_FILES
  App/TAppEncoder/encmain.cpp
  App/TAppEncoder/TAppEncCfg.cpp
  App/TAppEncoder/TAppEncCfg.h
  App/TAppEncoder/TAppEncTop.cpp
  App/TAppEncoder/TAppEncTop.h
  App/TAppEncoder/AccessUnit.h
  App/TAppEncoder/AnnexBwrite.h
  App/TAppEncoder/CommonDef.h
  App/TAppEncoder/ContextModel.cpp
  App/TAppEncoder/ContextModel.h
  App/TAppEncoder/ContextModel3DBuffer.cpp
  App/TAppEncoder/ContextModel3DBuffer.h
  App/TAppEncoder/ContextTables.h
  App/TAppEncoder/csvfile.h
  App/TAppEncoder/Debug.cpp
  App/TAppEncoder/Debug.h
  App/TAppEncoder/libmd5.cpp
  App/TAppEncoder/libmd5.h
  App/TAppEncoder/MD5.h
  App/TAppEncoder/NAL.h
  App/TAppEncoder/NALwrite.cpp
  App/TAppEncoder/NALwrite.h
  App/TAppEncoder/program_options_lite.cpp
  App/TAppEncoder/program_options_lite.h
  App/TAppEncoder/SEI.cpp
  App/TAppEncoder/SEI.h
  App/TAppEncoder/SEIEncoder.cpp
  App/TAppEncoder/SEIEncoder.h
  App/TAppEncoder/SEIwrite.cpp
  App/TAppEncoder/SEIwrite.h
  App/TAppEncoder/SyntaxElementWriter.cpp
  App/TAppEncoder/SyntaxElementWriter.h
  App/TAppEncoder/TAppComCamPara.cpp
  App/TAppEncoder/TAppComCamPara.h
  App/TAppEncoder/TComBitCounter.h
  App/TAppEncoder/TComBitStream.cpp
  App/TAppEncoder/TComBitStream.h
  App/TAppEncoder/TComCABACTables.cpp
  App/TAppEncoder/TComCABACTables.h
  App/TAppEncoder/TComChromaFormat.cpp
  App/TAppEncoder/TComChromaFormat.h
  App/TAppEncoder/TComCodingStatistics.h
  App/TAppEncoder/TComDataCU.cpp
  App/TAppEncoder/TComDataCU.h
  App/TAppEncoder/TComInterpolationFilter.cpp
  App/TAppEncoder/TComInterpolationFilter.h
  App/TAppEncoder/TComList.h
  App/TAppEncoder/TComLoopFilter.cpp
  App/TAppEncoder/TComLoopFilter.h
  App/TAppEncoder/TComMotionInfo.cpp
  App/TAppEncoder/TComMotionInfo.h
  App/TAppEncoder/TComMv.h
  App/TAppEncoder/TComPattern.cpp
  App/TAppEncoder/TComPattern.h
  App/TAppEncoder/TComPic.cpp
  App/TAppEncoder/TComPic.h
  App/TAppEncoder/TComPicSym.cpp
  App/TAppEncoder/TComPicSym.h
  App/TAppEncoder/TComPicYuv.cpp
  App/TAppEncoder/TComPicYuv.h
  App/TAppEncoder/TComPicYuvMD5.cpp
  App/TAppEncoder/TComPrediction.cpp
  App/TAppEncoder/TComPrediction.h
  App/TAppEncoder/TComRdCost.cpp
  App/TAppEncoder/TComRdCost.h
  App/TAppEncoder/TComRdCostWeightPrediction.cpp
  App/TAppEncoder/TComRdCostWeightPrediction.h
  App/TAppEncoder/TComRectangle.h
  App/TAppEncoder/TComRom.cpp
  App/TAppEncoder/TComRom.h
  App/TAppEncoder/TComSampleAdaptiveOffset.cpp
  App/TAppEncoder/TComSampleAdaptiveOffset.h
  App/TAppEncoder/TComSlice.cpp
  App/TAppEncoder/TComSlice.h
  App/TAppEncoder/TComTrQuant.cpp
  App/TAppEncoder/TComTrQuant.h
  App/TAppEncoder/TComTU.cpp
  App/TAppEncoder/TComTU.h
  App/TAppEncoder/TComWedgelet.cpp
  App/TAppEncoder/TComWedgelet.h
  App/TAppEncoder/TComWeightPrediction.cpp
  App/TAppEncoder/TComWeightPrediction.h
  App/TAppEncoder/TComYuv.cpp
  App/TAppEncoder/TComYuv.h
  App/TAppEncoder/TDecBinCoder.h
  App/TAppEncoder/TDecEntropy.cpp
  App/TAppEncoder/TDecEntropy.h
  App/TAppEncoder/TDecSbac.cpp
  App/TAppEncoder/TDecSbac.h
  App/TAppEncoder/TEncAnalyze.h
  App/TAppEncoder/TEncBinCoder.h
  App/TAppEncoder/TEncBinCoderCABAC.cpp
  App/TAppEncoder/TEncBinCoderCABAC.h
  App/TAppEncoder/TEncBinCoderCABACCounter.cpp
  App/TAppEncoder/TEncBinCoderCABACCounter.h
  App/TAppEncoder/TEncCavlc.cpp
  App/TAppEncoder/TEncCavlc.h
  App/TAppEncoder/TEncCfg.h
  App/TAppEncoder/TEncCu.cpp
  App/TAppEncoder/TEncCu.h
  App/TAppEncoder/TEncEntropy.cpp
  App/TAppEncoder/TEncEntropy.h
  App/TAppEncoder/TEncGOP.cpp
  App/TAppEncoder/TEncGOP.h
  App/TAppEncoder/TEncPic.cpp
  App/TAppEncoder/TEncPic.h
  App/TAppEncoder/TEncPreanalyzer.cpp
  App/TAppEncoder/TEncPreanalyzer.h
  App/TAppEncoder/TEncRateCtrl.cpp
  App/TAppEncoder/TEncRateCtrl.h
  App/TAppEncoder/TEncSampleAdaptiveOffset.cpp
  App/TAppEncoder/TEncSampleAdaptiveOffset.h
  App/TAppEncoder/TEncSbac.cpp
  App/TAppEncoder/TEncSbac.h
  App/TAppEncoder/TEncSearch.cpp
  App/TAppEncoder/TEncSearch.h
  App/TAppEncoder/TEncSlice.cpp
  App/TAppEncoder/TEncSlice.h
  App/TAppEncoder/TEncTop.cpp
  App/TAppEncoder/TEncTop.h
  App/TAppEncoder/TRenFilter.cpp
  App/TAppEncoder/TRenFilter.h
  App/TAppEncoder/TRenImage.cpp
  App/TAppEncoder/TRenImage.h
  App/TAppEncoder/TRenImagePlane.cpp
  App/TAppEncoder/TRenImagePlane.h
  App/TAppEncoder/TRenInterpFilter.cpp
  App/TAppEncoder/TRenInterpFilter.h
  App/TAppEncoder/TRenModel.cpp
  App/TAppEncoder/TRenModel.h
  App/TAppEncoder/TRenModSetupStrParser.cpp
  App/TAppEncoder/TRenModSetupStrParser.h
  App/TAppEncoder/TRenSingleModel.cpp
  App/TAppEncoder/TRenSingleModel.h
  App/TAppEncoder/TRenTop.cpp
  App/TAppEncoder/TRenTop.h
  App/TAppEncoder/TVideoIOYuv.cpp
  App/TAppEncoder/TVideoIOYuv.h
  App/TAppEncoder/TypeDef.h
  App/TAppEncoder/WeightPredAnalysis.cpp
  App/TAppEncoder/WeightPredAnalysis.h
  App/TAppEncoder/AphaTimeCost.h
  App/TAppEncoder/AphaTensorflowGlobalVars.h App/TAppEncoder/AphaTfPha.cpp App/TAppEncoder/AphaTfPha.h)

include_directories(.)
include_directories(App)
include_directories(App/TAppEncoder)

add_executable(TAppEncoder ${SOURCE_FILES})
target_link_libraries(TAppEncoder TensorflowCC::Shared)
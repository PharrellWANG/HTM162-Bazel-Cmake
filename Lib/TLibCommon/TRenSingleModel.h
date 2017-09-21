/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2016, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef __TRENSINGLEMODEL__ 
#define __TRENSINGLEMODEL__

#include "HTM162/Lib/TLibCommon/TRenImage.h"
#include "HTM162/Lib/TLibCommon/CommonDef.h"
#include "HTM162/Lib/TLibCommon/TComPicYuv.h"
#include "HTM162/Lib/TLibCommon/TypeDef.h"
#include "HTM162/Lib/TLibCommon/TAppComCamPara.h"


#include <math.h>
#include <errno.h>
#include <iostream>

#include <string>
#include <cstdio>
#include <cstring>

#if NH_3D_VSO
using namespace std;

#if H_3D_VSO_RM_ASSERTIONS
#define RM_AOT( exp ) AOT ( exp )
#define RM_AOF( exp ) AOF ( exp )
#else
#define RM_AOT( exp ) ((void)0)
#define RM_AOF( exp ) ((void)0)
#endif

#define RenModRemoveBitInc( exp ) bBitInc ? ( RemoveBitIncrement( exp ) ) : ( exp ) 

class TRenSingleModel
{
public: 

  virtual ~TRenSingleModel() { }  
#if H_3D_VSO_EARLY_SKIP
  virtual Void   create    ( Int iMode, Int iWidth, Int iHeight, Int iShiftPrec, Int*** aaaiSubPelShiftTable, Int iHoleMargin, Bool bUseOrgRef, Int iBlendMode, Bool bLimOutput, Bool bEarlySkip ) = 0;
#else
  virtual Void   create    ( Int iMode, Int iWidth, Int iHeight, Int iShiftPrec, Int*** aaaiSubPelShiftTable, Int iHoleMargin, Bool bUseOrgRef, Int iBlendMode, Bool bLimOutput ) = 0;
#endif

  // Setup 
  virtual Void   setLRView         ( Int iViewPos, Pel** apiCurVideoPel, Int* aiCurVideoStride, Pel* piCurDepthPel, Int iCurDepthStride ) = 0;
  virtual Void   setupPart         ( UInt uiHorOffset,       Int iUsedHeight ) = 0;
#if RM_FIX_SETUP
  virtual Void   setupLut          ( Int** ppiShiftLutLeft, Int** ppiBaseShiftLutLeft, Int** ppiShiftLutRight, Int** ppiBaseShiftLutRight, Int iDistToLeft ) = 0;
  virtual Void   setupRefView      ( TComPicYuv* pcOrgVideo ) = 0;

  virtual Void   renderAll                   ()                     = 0;
  virtual Void   setStructSynthViewAsRefView ()                     = 0;
  virtual Void   resetStructError            ()                     = 0;
  virtual Void   setLimOutStruct             ( Int iSourceViewPos ) = 0; 
#else
  virtual Void   setupLutAndRef    ( TComPicYuv* pcOrgVideo, Int** ppiShiftLutLeft, Int** ppiBaseShiftLutLeft, Int** ppiShiftLutRight, Int** ppiBaseShiftLutRight, Int iDistToLeft, Bool bRenderRef ) = 0;
  virtual Void   setupInitialState ( Int curViewPosInModel ) = 0;
#endif

  // Set Data
#if H_3D_VSO_EARLY_SKIP
  virtual Void   setDepth  ( Int iViewPos,                 Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData, const Pel* piOrgData, Int iOrgStride )  = 0;
#else
  virtual Void   setDepth  ( Int iViewPos,                 Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData )  = 0;
#endif
  virtual Void   setVideo  ( Int iViewPos,     Int iPlane, Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData ) = 0;

  // Get Distortion
#if H_3D_VSO_EARLY_SKIP
  virtual RMDist getDistDepth  ( Int iViewPos,             Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData, const Pel * piOrgData , Int iOrgStride)=0;
#else
  virtual RMDist getDistDepth  ( Int iViewPos,             Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData ) = 0;
#endif
  virtual RMDist getDistVideo  ( Int iViewPos, Int iPlane, Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData ) = 0;

  virtual Void   getSynthVideo  ( Int iViewPos, TComPicYuv* pcPicYuv ) = 0;  
  virtual Void   getSynthDepth  ( Int iViewPos, TComPicYuv* pcPicYuv ) = 0;
  virtual Void   getRefVideo    ( Int iViewPos, TComPicYuv* pcPicYuv ) = 0;
};

template < BlenMod iBM, Bool bBitInc >
class TRenSingleModelC : public TRenSingleModel
{
  struct RenModelInPels
  {
    // video
    Pel aiY[5]    ; // y-value
#if H_3D_VSO_COLOR_PLANES
    Pel aiU[5]    ; // u-value
    Pel aiV[5]    ; // v-value
#endif
    // depth
    Pel iD        ; // depth
   Int aiOccludedPos; // Occluded

  };

  struct RenModelLimOutPels
  {
    Pel iDOther;
    Int iFilledOther; 
    // video
    Pel iYOther; 
    Pel iYRef  ; 
#if H_3D_VSO_COLOR_PLANES
    Pel iUOther; 
    Pel iURef  ; 
    Pel iVOther;
    Pel iVRef  ; 
#endif  
    Int iError ;
  };

  struct RenModelOutPels
  {
    // video
    Pel iYLeft    ; 
    Pel iYRight   ; 
    Pel iYBlended ; 
#if H_3D_VSO_COLOR_PLANES
    Pel iULeft    ; 
    Pel iURight   ; 
    Pel iUBlended ; 
    Pel iVLeft    ; 
    Pel iVRight   ; 
    Pel iVBlended ; 
#endif
    // depth
    Pel iDLeft    ;
    Pel iDRight   ; 
    Pel iDBlended ; 

    // state
    Int iFilledLeft ; 
    Int iFilledRight; 

    // error
    Int  iError   ;

    // reference
    Pel iYRef    ; 
#if H_3D_VSO_COLOR_PLANES
    Pel iURef    ; 
    Pel iVRef    ; 
#endif        
  };



public:
  TRenSingleModelC();
  ~TRenSingleModelC();

  // Create Model
#if H_3D_VSO_EARLY_SKIP
  Void   create    ( Int iMode, Int iWidth, Int iHeight, Int iShiftPrec, Int*** aaaiSubPelShiftTable, Int iHoleMargin, Bool bUseOrgRef, Int iBlendMode, Bool bLimOutput, Bool bEarlySkip  );
#else
  Void   create    ( Int iMode, Int iWidth, Int iHeight, Int iShiftPrec, Int*** aaaiSubPelShiftTable, Int iHoleMargin, Bool bUseOrgRef, Int iBlendMode, Bool bLimOutput );
#endif

  // Setup 
  Void   setLRView         ( Int iViewPos, Pel** apiCurVideoPel, Int* aiCurVideoStride, Pel* piCurDepthPel, Int iCurDepthStride );  
  Void   setupPart         ( UInt uiHorOffset,       Int uiUsedHeight ); 
#if RM_FIX_SETUP
  Void   setupLut          (  Int** ppiShiftLutLeft, Int** ppiBaseShiftLutLeft, Int** ppiShiftLutRight, Int** ppiBaseShiftLutRight, Int iDistToLeft);
  Void   setupRefView      ( TComPicYuv* pcOrgVideo );
  
  __inline   Void   renderAll( );
  Void              setStructSynthViewAsRefView ();
  Void              resetStructError            ();
  Void              setLimOutStruct             ( Int iSourceViewPos ); 
#else
  Void   setupLutAndRef    ( TComPicYuv* pcOrgVideo, Int** ppiShiftLutLeft, Int** ppiBaseShiftLutLeft, Int** ppiShiftLutRight, Int** ppiBaseShiftLutRight, Int iDistToLeft, Bool bRenderRef );
  Void   setupInitialState ( Int curViewPosInModel );
#endif


#if H_3D_VSO_EARLY_SKIP
  Void   setDepth  ( Int iViewPos,                 Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData, const Pel* piOrgData, Int iOrgStride );
#else
  Void   setDepth  ( Int iViewPos,                 Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData );
#endif
  Void   setVideo  ( Int iViewPos,     Int iPlane, Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData );

  // Get Distortion
#if H_3D_VSO_EARLY_SKIP
  RMDist getDistDepth  ( Int iViewPos,             Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData, const Pel * piOrgData , Int iOrgStride);
#else
  RMDist getDistDepth  ( Int iViewPos,             Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData );
#endif
  RMDist getDistVideo  ( Int iViewPos, Int iPlane, Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData );

  Void   getSynthVideo  ( Int iViewPos, TComPicYuv* pcPicYuv );  
  Void   getSynthDepth  ( Int iViewPos, TComPicYuv* pcPicYuv );
  Void   getRefVideo    ( Int iViewPos, TComPicYuv* pcPicYuv );

private:

#if !RM_FIX_SETUP
   __inline Void  xRenderAll( );
#endif


#if H_3D_VSO_EARLY_SKIP
    template < Bool bL, SetMod iSM > RMDist xSetOrGet( Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData, Pel* piOrgData, Int iOrgStride );     
#else
    template < Bool bL, SetMod iSM > RMDist xSetOrGet( Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData );
#endif

  // Set and inc Current Row
  template< SetMod bSM  > __inline Void   xSetViewRow(  Int iPosY );
  template< SetMod bSM  > __inline Void   xIncViewRow();

  /////  Rendering /////
  template< typename T, Bool bL > __inline T    xPlus   ( T arg1, T arg2 ) { return bL ? (arg1 + arg2) : (arg1 - arg2); };
  template< typename T, Bool bL > __inline T    xMin    ( T arg1, T arg2 ) { return bL ? std::min(arg1, arg2) : std::max(arg1, arg2) ;};
  template< typename T, Bool bL > __inline T    xMax    ( T arg1, T arg2 ) { return bL ? std::max(arg1, arg2) : std::min(arg1, arg2) ;};
  template< typename T, Bool bL > __inline Void xInc    ( T& arg1        ) { bL ? arg1++ : arg1-- ;};
  template< typename T, Bool bL > __inline Void xDec    ( T& arg1        ) { bL ? arg1-- : arg1++ ;};
  template< typename T, Bool bL > __inline Bool xLess   ( T arg1, T arg2 ) { return bL ? arg1 <  arg2 : arg1 >  arg2; };  
  template< typename T, Bool bL > __inline Bool xGeQ    ( T arg1, T arg2 ) { return bL ? arg1 >= arg2 : arg1 <= arg2; };
  template<             Bool bL > __inline Int  xZero   (                ) { return bL ? 0 : m_iWidth - 1; };
  template<             Bool bL > __inline Int  xWidthMinus1(            ) { return bL ? m_iWidth - 1 : 0; };
#if H_3D_VSO_EARLY_SKIP
  template<             Bool bL > __inline Bool   xDetectEarlySkip    ( Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData,const Pel* piOrgData, Int iOrgStride );
  template< Bool bL, SetMod bSM > __inline RMDist xRender             ( Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData, Bool bFast );
  template< SetMod bSM >            __inline RMDist xGetSSE             ( Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData, Bool bFast );
#else       
  template< Bool bL, SetMod bSM  > __inline RMDist xRender             ( Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData );
  template<          SetMod bSM  > __inline RMDist xGetSSE             ( Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData );
#endif                          
  template< Bool bL              > __inline Void   xInitRenderPart    ( Int iEndChangePos, Int iLastSPos  );
  template< Bool bL, SetMod bSM  > __inline Void   xRenderRange       ( Int iCurSPos, Int iLastSPos, Int iCurPos, RMDist& riError );
  template< Bool bL, SetMod bSM  > __inline Void   xRenderShiftedRange( Int iCurSPos, Int iLastSPos, Int iCurPos, RMDist& riError );
  template< Bool bL, SetMod bSM  > __inline Void   xFillHole          ( Int iCurSPos, Int iLastSPos, Int iCurPos, RMDist& riError );
  template< Bool bL, SetMod bSM  > __inline Void   xExtrapolateMargin ( Int iCurSPos,                Int iCurPos, RMDist& riError );
  template< Bool bL              > __inline Int    xRangeLeft         ( Int iPos );
  template< Bool bL              > __inline Int    xRangeRight        ( Int iPos );
  template< Bool bL              > __inline Int    xRound             ( Int iPos );

#if H_3D_VSO_COLOR_PLANES  
  template< Bool bL, SetMod bSM > __inline Void   xGetBlendedValue    ( Pel& riY, Pel iYL,  Pel iYR, Pel& riU, Pel iUL,  Pel iUR,  Pel& riV, Pel iVL,  Pel iVR, Int iFilledL,  Int iFilledR, Pel iDepthL,  Pel iDepthR );
  template< Bool bL, SetMod bSM > __inline Void   xGetBlendedValueBM1 ( Pel& riY, Pel iYL,  Pel iYR, Pel& riU, Pel iUL,  Pel iUR,  Pel& riV, Pel iVL,  Pel iVR, Int iFilledL,  Int iFilledR, Pel iDepthL,  Pel iDepthR );
  template< Bool bL, SetMod bSM > __inline Void   xGetBlendedValueBM2 ( Pel& riY, Pel iYL,  Pel iYR, Pel& riU, Pel iUL,  Pel iUR,  Pel& riV, Pel iVL,  Pel iVR, Int iFilledL,  Int iFilledR, Pel iDepthL,  Pel iDepthR );
#else
  template< Bool bL, SetMod bSM > __inline Void   xGetBlendedValue    ( Pel& riY, Pel iYL,  Pel iYR, Int iFilledL,  Int iFilledR, Pel iDepthL,  Pel iDepthR  );
  template< Bool bL, SetMod bSM > __inline Void   xGetBlendedValueBM1 ( Pel& riY, Pel iYL,  Pel iYR, Int iFilledL,  Int iFilledR, Pel iDepthL,  Pel iDepthR  );
  template< Bool bL, SetMod bSM > __inline Void   xGetBlendedValueBM2 ( Pel& riY, Pel iYL,  Pel iYR, Int iFilledL,  Int iFilledR, Pel iDepthL,  Pel iDepthR  );
#endif
  __inline Pel    xBlend              ( Pel pVal1, Pel pVal2, Int iWeightVal2 );

  // General
  template<Bool bL, SetMod bSM> __inline Void xSetShiftedPel       (Int iSourcePos, Int iSubSourcePos, Int iTargetSPos, Pel iFilled, RMDist& riError );
  
  template <Bool bL>           __inline Int  xShiftNewData        ( Int iPos, Int iPosInNewData );
  template <Bool bL>           __inline Int  xShift               ( Int iPos );
  template <Bool bL>           __inline Int  xShift               ( Int iPos, Int iPosInNewData );

  __inline Int    xShiftDept         ( Int iPosXinSubPel, Int iDepth ); 

  __inline Int    xGetDist           ( Int iDiffY, Int iDiffU, Int iDiffV );
  __inline Int    xGetDist           ( Int iDiffY );

  // Utilities
  __inline Void   xSetPels   ( Pel*  piPelSource , Int iSourceStride, Int iWidth, Int iHeight, Pel iVal );
  __inline Void   xSetBools  ( Bool* pbSource    , Int iSourceStride, Int iWidth, Int iHeight, Bool bVal );
  __inline Void   xSetInts   ( Int*  piPelSource , Int iSourceStride, Int iWidth, Int iHeight, Int iVal );

#if H_3D_VSO_COLOR_PLANES
  Void            xGetSampleStrTextPtrs ( Int iViewNum, Pel RenModelOutPels::*& rpiSrcY, Pel RenModelOutPels::*& rpiSrcU, Pel RenModelOutPels::*& rpiSrcV );
#else  
  Void            xGetSampleStrTextPtrs ( Int iViewNum, Pel RenModelOutPels::*& rpiSrcY );
#endif 
  Void            xGetSampleStrDepthPtrs ( Int iViewNum, Pel RenModelOutPels::*& rpiSrcD      );
  Void            xGetSampleStrFilledPtrs( Int iViewNum, Int RenModelOutPels::*& rpiSrcFilled );

       
  Void            xSetStructRefView            ();
#if !RM_FIX_SETUP
  Void            xResetStructError            ();
  Void            xSetLimOutStruct             (Int iSourceViewPos ); 
#endif
  
  Void            xInitSampleStructs           ();
#if !RM_FIX_SETUP
  Void            xSetStructSynthViewAsRefView ();
#endif
  Void            xCopy2PicYuv                ( Pel** ppiSrcVideoPel, Int* piStrides, TComPicYuv* rpcPicYuvTarget );

  template< typename S, typename T> 
  Void   xCopyFromSampleStruct ( S* ptSource , Int iSourceStride, T S::* ptSourceElement, T* ptTarget, Int iTargetStride, Int iWidth, Int iHeight )
  {
    AOT( iWidth != m_iWidth ); 
    for (Int iPosY = 0; iPosY < iHeight; iPosY++)
    {
      for (Int iPosX = 0; iPosX < m_iWidth; iPosX++)
      {
        ptTarget[iPosX] = ptSource[iPosX].*ptSourceElement;
      }
      ptSource += iSourceStride;
      ptTarget += iTargetStride;
    }    
  }  

  template< typename S, typename T> 
  Void   xCopyToSampleStruct ( T* ptSource , Int iSourceStride, S* ptTarget, Int iTargetStride, T S::* ptSourceElement, Int iWidth, Int iHeight )
  {
    AOT( iWidth != m_iWidth ); 
    for (Int iPosY = 0; iPosY < iHeight; iPosY++)
    {
      for (Int iPosX = 0; iPosX < m_iWidth; iPosX++)
      {
        ptTarget[iPosX] = ptSource[iPosX].*ptSourceElement;
      }
      ptSource += iSourceStride;
      ptTarget += iTargetStride;
    }    
  }   

private:

  // Image sizes
  Int   m_iWidth;
  Int   m_iHeight;
  Int   m_iStride;
  Int   m_iPad;
  Int   m_iUsedHeight;
  Int   m_iHorOffset; 

  Int   m_iSampledWidth;
  Int   m_iSampledStride;

  RenModelInPels* m_pcInputSamples[2];
  Int             m_iInputSamplesStride;

  // Base
  Pel** m_aapiBaseVideoPel     [2]; // Dim1: ViewPosition 0->Left, 1->Right; Dim2: Plane  0-> Y, 1->U, 2->V
  Int*  m_aaiBaseVideoStrides  [2]; // Dim1: ViewPosition 0->Left, 1->Right; Dim2: Plane  0-> Y, 1->U, 2->V

  Pel*  m_apiBaseDepthPel      [2]; // Dim1: ViewPosition
  Int   m_aiBaseDepthStrides   [2]; // Dim1: ViewPosition


  // LUT
  Int** m_appiShiftLut         [2];
  Int** m_ppiCurLUT;
  Int** m_aaiSubPelShiftL;
  Int** m_aaiSubPelShiftR;

  Int*  m_piInvZLUTLeft;
  Int*  m_piInvZLUTRight;


  //// Reference Data  ////
  TComPicYuv* m_pcPicYuvRef       ;    // Reference PIcYuv

  //// Output Samples
  RenModelOutPels*      m_pcOutputSamples;
  RenModelLimOutPels*   m_pcLimOutputSamples; 

  Int                   m_iOutputSamplesStride;

  Pel*  m_aapiRefVideoPel      [3];    // Dim1: Plane  0-> Y, 1->U, 2->V
  Int   m_aiRefVideoStrides    [3];    // Dim1: Plane  0-> Y, 1->U, 2->V

  // Rendering State
  Bool  m_bInOcclusion;                // Currently rendering in occluded area
  Int   m_iLastOccludedSPos;           // Position of last topmost shifted position

  Int   m_curRangeStart; 
  Int   m_lastRangeStart;  

  const Pel*  m_piNewDepthData;              // Pointer to new depth data
  Int   m_iStartChangePosX;            // Start Position of new data
  Int   m_iNewDataWidth;               // Width of new data
  Pel   m_iCurDepth;                   // Current Depth Value
  Pel   m_iLastDepth;                  // Last Depth Value
  Pel   m_iThisDepth;                  // Depth value to use for setting

  //// Settings ////
  // Input
  Int   m_iMode;                       // 0: Left to Right, 1: Right to Left, 2: Merge
  Bool  m_bUseOrgRef;
  Int   m_iShiftPrec;
  Int   m_iHoleMargin;
#if H_3D_VSO_EARLY_SKIP
  Bool  m_bEarlySkip; 
#endif

  // Derived settings
  Int   m_iGapTolerance;
  Int   m_iBlendZThres;
  Int   m_iBlendDistWeight;

  //// Current Pointers ////

  RenModelInPels*     m_pcInputSamplesRow [2];
  RenModelOutPels*    m_pcOutputSamplesRow;
  RenModelLimOutPels* m_pcLimOutputSamplesRow;

  //// MISC ////
  const Int m_iDistShift;            // Shift in Distortion computation
  Bool      m_bLimOutput;            // Save distortion only

  //// Early Skip 
#if H_3D_VSO_EARLY_SKIP
  Bool* m_pbHorSkip;
#endif
};

#endif // H_3D
#endif //__TRENSINGLEMODEL__



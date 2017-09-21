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

#include "TRenImage.h"
#include "TRenFilter.h"
#include "TRenSingleModel.h"

#if NH_3D_VSO

////////////// TRENSINGLE MODEL ///////////////
template <BlenMod iBM, Bool bBitInc>
TRenSingleModelC<iBM,bBitInc>::TRenSingleModelC()
:  m_iDistShift ( ( ENC_INTERNAL_BIT_DEPTH  - REN_BIT_DEPTH) << 1 )
{
  m_iWidth  = -1;
  m_iHeight = -1;
  m_iStride = -1;
  m_iUsedHeight = -1; 
  m_iHorOffset  = -1; 
  m_iMode   = -1;
  m_iPad    = PICYUV_PAD;
  m_iGapTolerance = -1;
  m_bUseOrgRef = false;

  m_pcPicYuvRef          = NULL;

  m_pcOutputSamples      = NULL; 
  m_pcOutputSamplesRow   = NULL; 
  m_iOutputSamplesStride = -1; 

  m_ppiCurLUT            = NULL;
  m_piInvZLUTLeft        = NULL;
  m_piInvZLUTRight       = NULL;

  m_aapiRefVideoPel[0]   = NULL;
  m_aapiRefVideoPel[1]   = NULL;
  m_aapiRefVideoPel[2]   = NULL;

  m_aiRefVideoStrides[0] = -1;
  m_aiRefVideoStrides[1] = -1;
  m_aiRefVideoStrides[2] = -1;


  for (UInt uiViewNum = 0 ; uiViewNum < 2; uiViewNum++)
  {
    // LUT
    m_appiShiftLut[uiViewNum] = NULL;

    m_pcInputSamples[uiViewNum] = NULL; 
    m_iInputSamplesStride       = -1; 

    m_ppiCurLUT               = NULL;
    m_piInvZLUTLeft           = NULL;
    m_piInvZLUTRight          = NULL;
  }

#if H_3D_VSO_EARLY_SKIP
  m_pbHorSkip = NULL;
#endif
}

template <BlenMod iBM, Bool bBitInc>
TRenSingleModelC<iBM,bBitInc>::~TRenSingleModelC()
{
#if H_3D_VSO_EARLY_SKIP
  if ( m_pbHorSkip ) 
  {
    delete[] m_pbHorSkip;
    m_pbHorSkip = NULL;
  }
#endif

  if ( m_pcInputSamples [0] ) delete[] m_pcInputSamples [0];
  if ( m_pcInputSamples [1] ) delete[] m_pcInputSamples [1];

  if ( m_pcOutputSamples    ) delete[] m_pcOutputSamples   ;

  if ( m_piInvZLUTLeft      ) delete[] m_piInvZLUTLeft ;
  if ( m_piInvZLUTRight     ) delete[] m_piInvZLUTRight;

  if ( m_aapiRefVideoPel[0] ) delete[] ( m_aapiRefVideoPel[0] - ( m_aiRefVideoStrides[0] * m_iPad + m_iPad ) );
  if ( m_aapiRefVideoPel[1] ) delete[] ( m_aapiRefVideoPel[1] - ( m_aiRefVideoStrides[1] * m_iPad + m_iPad ) );
  if ( m_aapiRefVideoPel[2] ) delete[] ( m_aapiRefVideoPel[2] - ( m_aiRefVideoStrides[2] * m_iPad + m_iPad ) );
}

template <BlenMod iBM, Bool bBitInc> Void
#if H_3D_VSO_EARLY_SKIP
TRenSingleModelC<iBM,bBitInc>::create( Int iMode, Int iWidth, Int iHeight, Int iShiftPrec, Int*** aaaiSubPelShiftTable, Int iHoleMargin, Bool bUseOrgRef, Int iBlendMode, Bool bLimOutput, Bool bEarlySkip )
#else
TRenSingleModelC<iBM,bBitInc>::create( Int iMode, Int iWidth, Int iHeight, Int iShiftPrec, Int*** aaaiSubPelShiftTable, Int iHoleMargin, Bool bUseOrgRef, Int iBlendMode, Bool bLimOutput )
#endif

{
  m_bLimOutput = bLimOutput; 
#if H_3D_VSO_EARLY_SKIP
  m_pbHorSkip     = new Bool [MAX_CU_SIZE];
  m_bEarlySkip    = bEarlySkip; 
#endif

  AOF( ( iBlendMode == iBM ) || ( iBM == BLEND_NONE ) ); 
  m_iMode = iMode;

  m_iWidth  = iWidth;
  m_iHeight = iHeight;
  m_iStride = iWidth;

  m_iSampledWidth  = m_iWidth  << iShiftPrec;
  m_iSampledStride = m_iStride << iShiftPrec;

  m_iShiftPrec     = iShiftPrec;
  m_aaiSubPelShiftL = aaaiSubPelShiftTable[0];
  m_aaiSubPelShiftR = aaaiSubPelShiftTable[1];

  if (m_iMode == 2)
  {
    m_piInvZLUTLeft  = new Int[257];
    m_piInvZLUTRight = new Int[257];
  }

  m_iGapTolerance  = ( 2 << iShiftPrec );
  m_iHoleMargin    =  iHoleMargin;

  m_bUseOrgRef = bUseOrgRef;

  m_aiRefVideoStrides[0] = m_iStride + (m_iPad << 1);
  m_aiRefVideoStrides[1] = m_iStride + (m_iPad << 1);
  m_aiRefVideoStrides[2] = m_iStride + (m_iPad << 1);

  m_aapiRefVideoPel  [0] = new Pel[ m_aiRefVideoStrides[0] * (m_iHeight + (m_iPad << 1))];
  m_aapiRefVideoPel  [1] = new Pel[ m_aiRefVideoStrides[1] * (m_iHeight + (m_iPad << 1))];
  m_aapiRefVideoPel  [2] = new Pel[ m_aiRefVideoStrides[2] * (m_iHeight + (m_iPad << 1))];

  m_aapiRefVideoPel  [0] += m_aiRefVideoStrides[0] * m_iPad + m_iPad;
  m_aapiRefVideoPel  [1] += m_aiRefVideoStrides[1] * m_iPad + m_iPad;
  m_aapiRefVideoPel  [2] += m_aiRefVideoStrides[2] * m_iPad + m_iPad;

  m_iInputSamplesStride  = m_iWidth+1;
  m_iOutputSamplesStride = m_iWidth;

  m_pcInputSamples[0]     = new RenModelInPels[m_iInputSamplesStride*m_iHeight];
  m_pcInputSamples[1]     = new RenModelInPels[m_iInputSamplesStride*m_iHeight];

  m_pcOutputSamples       = new RenModelOutPels   [m_iOutputSamplesStride*m_iHeight];
  m_pcLimOutputSamples    = m_bLimOutput ? new RenModelLimOutPels[m_iOutputSamplesStride*m_iHeight] : NULL;
}

template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::setLRView( Int iViewPos, Pel** apiCurVideoPel, Int* aiCurVideoStride, Pel* piCurDepthPel, Int iCurDepthStride )
{
  AOF(( iViewPos == 0) || (iViewPos == 1) );

  RenModelInPels* pcCurInputSampleRow = m_pcInputSamples[iViewPos];
  
  Pel* piDRow = piCurDepthPel;
  Pel* piYRow = apiCurVideoPel[0];
#if H_3D_VSO_COLOR_PLANES
  Pel* piURow = apiCurVideoPel[1];
  Pel* piVRow = apiCurVideoPel[2];
#endif  


  Int iOffsetX = ( iViewPos == VIEWPOS_RIGHT ) ? 1 : 0;

  for ( Int iPosY = 0; iPosY < m_iUsedHeight; iPosY++ )
  {
    if ( iViewPos == VIEWPOS_RIGHT )
    {
      Int iSubPosX = (1 << m_iShiftPrec); 
      pcCurInputSampleRow[0].aiY[iSubPosX] = piYRow[0];
#if H_3D_VSO_COLOR_PLANES 
      pcCurInputSampleRow[0].aiU[iSubPosX] = piURow[0];
      pcCurInputSampleRow[0].aiV[iSubPosX] = piVRow[0];
#endif
    }

    for ( Int iPosX = 0; iPosX < m_iWidth; iPosX++ )
    {
      pcCurInputSampleRow[iPosX].iD = piDRow[iPosX];

      for (Int iSubPosX = 0; iSubPosX < (1 << m_iShiftPrec)+1; iSubPosX++ )
      { 
        Int iShift = (iPosX << m_iShiftPrec) + iSubPosX;
        pcCurInputSampleRow[iPosX+iOffsetX].aiY[iSubPosX] = piYRow[iShift];
#if H_3D_VSO_COLOR_PLANES 
        pcCurInputSampleRow[iPosX+iOffsetX].aiU[iSubPosX] = piURow[iShift];
        pcCurInputSampleRow[iPosX+iOffsetX].aiV[iSubPosX] = piVRow[iShift];
#endif
      }
    } 

    pcCurInputSampleRow += m_iInputSamplesStride; 

    piDRow += iCurDepthStride;
    piYRow += aiCurVideoStride[0];
#if H_3D_VSO_COLOR_PLANES
    piURow += aiCurVideoStride[1];
    piVRow += aiCurVideoStride[2];
#endif
  }

  
  m_aapiBaseVideoPel      [iViewPos] = apiCurVideoPel;
  m_aaiBaseVideoStrides   [iViewPos] = aiCurVideoStride;
  m_apiBaseDepthPel       [iViewPos] = piCurDepthPel;
  m_aiBaseDepthStrides    [iViewPos] = iCurDepthStride;

}
template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::setupPart ( UInt uiHorOffset,       Int iUsedHeight )
{
  AOT( iUsedHeight > m_iHeight );   

  m_iUsedHeight =       iUsedHeight; 
  m_iHorOffset  = (Int) uiHorOffset;
}


#if !RM_FIX_SETUP
template <BlenMod iBM, Bool bBitInc> Void
  TRenSingleModelC<iBM,bBitInc>::setupInitialState( Int curViewPosInModel )
{
  xRenderAll( );

  if ( m_bLimOutput )
  {
    AOT( curViewPosInModel == VIEWPOS_INVALID )
    xSetLimOutStruct( (curViewPosInModel == VIEWPOS_RIGHT)  ? VIEWPOS_LEFT : VIEWPOS_RIGHT );
  }
}
#endif

#if RM_FIX_SETUP
template <BlenMod iBM, Bool bBitInc> Void
  TRenSingleModelC<iBM,bBitInc>::setupRefView      ( TComPicYuv* pcOrgVideo )
{
  m_pcPicYuvRef = pcOrgVideo;
  // Use provided ref view reference
  
  TRenFilter<REN_BIT_DEPTH>::copy(             pcOrgVideo->getAddr( COMPONENT_Y  ) +  m_iHorOffset       * pcOrgVideo->getStride( COMPONENT_Y  ), pcOrgVideo->getStride( COMPONENT_Y  ), m_iWidth,      m_iUsedHeight,      m_aapiRefVideoPel[0], m_aiRefVideoStrides[0]);
  switch ( pcOrgVideo->getChromaFormat() )
  {
  case CHROMA_420:
    TRenFilter<REN_BIT_DEPTH>::sampleCUpHorUp(0, pcOrgVideo->getAddr( COMPONENT_Cb ) + (m_iHorOffset >> 1) * pcOrgVideo->getStride( COMPONENT_Cb ), pcOrgVideo->getStride( COMPONENT_Cb ), m_iWidth >> 1, m_iUsedHeight >> 1, m_aapiRefVideoPel[1], m_aiRefVideoStrides[1]);
    TRenFilter<REN_BIT_DEPTH>::sampleCUpHorUp(0, pcOrgVideo->getAddr( COMPONENT_Cr ) + (m_iHorOffset >> 1) * pcOrgVideo->getStride( COMPONENT_Cr ), pcOrgVideo->getStride( COMPONENT_Cr ), m_iWidth >> 1, m_iUsedHeight >> 1, m_aapiRefVideoPel[2], m_aiRefVideoStrides[2]);
    break;
  case CHROMA_444:
    TRenFilter<REN_BIT_DEPTH>::copy(             pcOrgVideo->getAddr( COMPONENT_Cb  ) +  m_iHorOffset       * pcOrgVideo->getStride( COMPONENT_Cb  ), pcOrgVideo->getStride( COMPONENT_Cb  ), m_iWidth,      m_iUsedHeight,      m_aapiRefVideoPel[1], m_aiRefVideoStrides[1]);
    TRenFilter<REN_BIT_DEPTH>::copy(             pcOrgVideo->getAddr( COMPONENT_Cr  ) +  m_iHorOffset       * pcOrgVideo->getStride( COMPONENT_Cr  ), pcOrgVideo->getStride( COMPONENT_Cr  ), m_iWidth,      m_iUsedHeight,      m_aapiRefVideoPel[2], m_aiRefVideoStrides[2]);
    break;
  default:
    break; 
  }
  xSetStructRefView();
}


template <BlenMod iBM, Bool bBitInc> Void
  TRenSingleModelC<iBM,bBitInc>::setupLut( Int** ppiShiftLutLeft, Int** ppiBaseShiftLutLeft, Int** ppiShiftLutRight,  Int** ppiBaseShiftLutRight,  Int iDistToLeft )
#else
template <BlenMod iBM, Bool bBitInc> Void
  TRenSingleModelC<iBM,bBitInc>::setupLutAndRef( TComPicYuv* pcOrgVideo, Int** ppiShiftLutLeft, Int** ppiBaseShiftLutLeft, Int** ppiShiftLutRight,  Int** ppiBaseShiftLutRight,  Int iDistToLeft, Bool bRenderRef )
#endif
{
#if !RM_FIX_SETUP
  AOT( !m_bUseOrgRef && pcOrgVideo );
#endif
  AOT( (ppiShiftLutLeft  == NULL) && (m_iMode == 0 || m_iMode == 2) );
  AOT( (ppiShiftLutRight == NULL) && (m_iMode == 1 || m_iMode == 2) );

  m_appiShiftLut[0] = ppiShiftLutLeft;
  m_appiShiftLut[1] = ppiShiftLutRight;


  if ( m_iMode == 2 )
  {
    TRenFilter<REN_BIT_DEPTH>::setupZLUT( true, 30, iDistToLeft, ppiBaseShiftLutLeft, ppiBaseShiftLutRight, m_iBlendZThres, m_iBlendDistWeight, m_piInvZLUTLeft, m_piInvZLUTRight );
  }

#if !RM_FIX_SETUP
  // Copy Reference
  m_pcPicYuvRef = pcOrgVideo;

  if ( pcOrgVideo )
  {
    assert( pcOrgVideo->getChromaFormat() == CHROMA_420 );     

    TRenFilter<REN_BIT_DEPTH>::copy(             pcOrgVideo->getAddr( COMPONENT_Y  ) +  m_iHorOffset       * pcOrgVideo->getStride( COMPONENT_Y  ), pcOrgVideo->getStride( COMPONENT_Y  ), m_iWidth,      m_iUsedHeight,      m_aapiRefVideoPel[0], m_aiRefVideoStrides[0]);
    TRenFilter<REN_BIT_DEPTH>::sampleCUpHorUp(0, pcOrgVideo->getAddr( COMPONENT_Cb ) + (m_iHorOffset >> 1) * pcOrgVideo->getStride( COMPONENT_Cb ), pcOrgVideo->getStride( COMPONENT_Cb ), m_iWidth >> 1, m_iUsedHeight >> 1, m_aapiRefVideoPel[1], m_aiRefVideoStrides[1]);
    TRenFilter<REN_BIT_DEPTH>::sampleCUpHorUp(0, pcOrgVideo->getAddr( COMPONENT_Cr ) + (m_iHorOffset >> 1) * pcOrgVideo->getStride( COMPONENT_Cr ), pcOrgVideo->getStride( COMPONENT_Cr ), m_iWidth >> 1, m_iUsedHeight >> 1, m_aapiRefVideoPel[2], m_aiRefVideoStrides[2]);    
    xSetStructRefView();
  }
  else
  {
    if ( bRenderRef )
    {    
      xRenderAll( ); 
      xSetStructSynthViewAsRefView();
    }
  }
#endif
}

template <BlenMod iBM, Bool bBitInc> Void
#if RM_FIX_SETUP
TRenSingleModelC<iBM,bBitInc>::renderAll( )
#else
TRenSingleModelC<iBM,bBitInc>::xRenderAll( )
#endif
{
  // Initial Rendering
#if RM_FIX_SETUP
  resetStructError();
#else
  xResetStructError();
#endif
  xInitSampleStructs();
  switch ( m_iMode )
  {  
  case 0:   
#if H_3D_VSO_EARLY_SKIP
    xRender<true, SET_FULL>( 0, 0, m_iWidth, m_iUsedHeight, m_aiBaseDepthStrides[0], m_apiBaseDepthPel[0],false );
#else
    xRender<true, SET_FULL>( 0, 0, m_iWidth, m_iUsedHeight, m_aiBaseDepthStrides[0], m_apiBaseDepthPel[0] );
#endif   
    break;
  case 1:    
#if H_3D_VSO_EARLY_SKIP
    xRender<false, SET_FULL>( 0, 0, m_iWidth, m_iUsedHeight, m_aiBaseDepthStrides[1], m_apiBaseDepthPel[1],false);
#else
    xRender<false, SET_FULL>( 0, 0, m_iWidth, m_iUsedHeight, m_aiBaseDepthStrides[1], m_apiBaseDepthPel[1] );
#endif
    break;
  case 2:
#if H_3D_VSO_EARLY_SKIP
    xRender<true , SET_FULL>( 0, 0, m_iWidth, m_iUsedHeight, m_aiBaseDepthStrides[0], m_apiBaseDepthPel[0],false);
    xRender<false, SET_FULL>( 0, 0, m_iWidth, m_iUsedHeight, m_aiBaseDepthStrides[1], m_apiBaseDepthPel[1],false);
#else      
    xRender<true,  SET_FULL>( 0, 0, m_iWidth, m_iUsedHeight, m_aiBaseDepthStrides[0], m_apiBaseDepthPel[0] );
    xRender<false, SET_FULL>( 0, 0, m_iWidth, m_iUsedHeight, m_aiBaseDepthStrides[1], m_apiBaseDepthPel[1] );
#endif
    break;
  default:
    AOT(true);
  }
}

template <BlenMod iBM, Bool bBitInc> Void
#if H_3D_VSO_COLOR_PLANES
TRenSingleModelC<iBM,bBitInc>::xGetSampleStrTextPtrs( Int iViewNum, Pel RenModelOutPels::*& rpiSrcY, Pel RenModelOutPels::*& rpiSrcU, Pel RenModelOutPels::*& rpiSrcV )
#else
TRenSingleModelC<iBM,bBitInc>::xGetSampleStrTextPtrs( Int iViewNum, Pel RenModelOutPels::*& rpiSrcY )
#endif
{
  switch ( iViewNum )
  {

  case -1:  
    rpiSrcY = &RenModelOutPels::iYRef;
#if H_3D_VSO_COLOR_PLANES  
    rpiSrcU = &RenModelOutPels::iURef;
    rpiSrcV = &RenModelOutPels::iVRef;
#endif
    break;
  case 0:
    rpiSrcY = &RenModelOutPels::iYLeft;
#if H_3D_VSO_COLOR_PLANES  
    rpiSrcU = &RenModelOutPels::iULeft;
    rpiSrcV = &RenModelOutPels::iVLeft;
#endif
    break;
  case 1:
    rpiSrcY = &RenModelOutPels::iYRight;
#if H_3D_VSO_COLOR_PLANES  
    rpiSrcU = &RenModelOutPels::iURight;
    rpiSrcV = &RenModelOutPels::iVRight;
#endif
    break;
  case 2:
    rpiSrcY = &RenModelOutPels::iYBlended;
#if H_3D_VSO_COLOR_PLANES  
    rpiSrcU = &RenModelOutPels::iUBlended;
    rpiSrcV = &RenModelOutPels::iVBlended;
#endif
    break;
  }
}


template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::xGetSampleStrDepthPtrs( Int iViewNum, Pel RenModelOutPels::*& rpiSrcD )
{
  AOT(iViewNum != 0 && iViewNum != 1);  
  rpiSrcD = (iViewNum == 1) ? &RenModelOutPels::iDRight : &RenModelOutPels::iDLeft; 
}

template <BlenMod iBM, Bool bBitInc> Void
  TRenSingleModelC<iBM,bBitInc>::xGetSampleStrFilledPtrs( Int iViewNum, Int RenModelOutPels::*& rpiSrcFilled )
{
  AOT(iViewNum != 0 && iViewNum != 1);  
  rpiSrcFilled = (iViewNum == 1) ? &RenModelOutPels::iFilledRight : &RenModelOutPels::iFilledLeft; 
}


template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::xSetStructRefView( )
{
  RenModelOutPels* pcCurOutSampleRow = m_pcOutputSamples;
  
  Pel* piYRow = m_aapiRefVideoPel[0];
#if H_3D_VSO_COLOR_PLANES
  Pel* piURow = m_aapiRefVideoPel[1];
  Pel* piVRow = m_aapiRefVideoPel[2];
#endif  

  for ( Int iPosY = 0; iPosY < m_iUsedHeight; iPosY++ )
  {
    for ( Int iPosX = 0; iPosX < m_iWidth; iPosX++ )
    {      
      pcCurOutSampleRow[iPosX].iYRef = piYRow[iPosX];
#if H_3D_VSO_COLOR_PLANES
      pcCurOutSampleRow[iPosX].iURef = piURow[iPosX];
      pcCurOutSampleRow[iPosX].iVRef = piVRow[iPosX];
#endif
    } 

    pcCurOutSampleRow += m_iOutputSamplesStride; 
    
    piYRow += m_aiRefVideoStrides[0];
#if H_3D_VSO_COLOR_PLANES
    piURow += m_aiRefVideoStrides[1];
    piVRow += m_aiRefVideoStrides[2];
#endif
  }
}

template <BlenMod iBM, Bool bBitInc> Void
#if RM_FIX_SETUP
TRenSingleModelC<iBM,bBitInc>::resetStructError( )
#else
TRenSingleModelC<iBM,bBitInc>::xResetStructError( )
#endif
{
  RenModelOutPels* pcCurOutSampleRow = m_pcOutputSamples;

  for ( Int iPosY = 0; iPosY < m_iHeight; iPosY++ )
  {
    for ( Int iPosX = 0; iPosX < m_iWidth; iPosX++ )
    {      
      pcCurOutSampleRow[iPosX].iError = 0;
    } 
    pcCurOutSampleRow += m_iOutputSamplesStride; 
  }
}

template <BlenMod iBM, Bool bBitInc> Void
#if RM_FIX_SETUP
  TRenSingleModelC<iBM,bBitInc>::setLimOutStruct(Int iSourceViewPos )
#else
  TRenSingleModelC<iBM,bBitInc>::xSetLimOutStruct(Int iSourceViewPos )
#endif
{  
  RM_AOF( m_bLimOutput ); 
  RM_AOT( iSourceViewPos < 0 || iSourceViewPos > 1);

  RenModelOutPels*    pcCurOutSampleRow    = m_pcOutputSamples;
  RenModelLimOutPels* pcCurLimOutSampleRow = m_pcLimOutputSamples;

  Int RenModelOutPels::* piFilled = NULL;
  xGetSampleStrFilledPtrs( iSourceViewPos, piFilled );
  
  Pel RenModelOutPels::* piDepth  = NULL;
  xGetSampleStrDepthPtrs ( iSourceViewPos, piDepth  );

  Pel RenModelOutPels::* piSrcY = NULL;
  
#if H_3D_VSO_COLOR_PLANES  
  Pel RenModelOutPels::* piSrcU = NULL;  
  Pel RenModelOutPels::* piSrcV = NULL;
  xGetSampleStrTextPtrs  ( iSourceViewPos, piSrcY, piSrcU, piSrcV );
#else
  xGetSampleStrTextPtrs  ( iSourceViewPos, piSrcY );
#endif
  
  for ( Int iPosY = 0; iPosY < m_iUsedHeight; iPosY++ )
  {
    for ( Int iPosX = 0; iPosX < m_iWidth; iPosX++ )
    {      
      pcCurLimOutSampleRow[iPosX].iYOther      = pcCurOutSampleRow[iPosX].*piSrcY;
      pcCurLimOutSampleRow[iPosX].iYRef        = pcCurOutSampleRow[iPosX].iYRef;

#if H_3D_VSO_COLOR_PLANES      
      pcCurLimOutSampleRow[iPosX].iUOther      = pcCurOutSampleRow[iPosX].*piSrcU;
      pcCurLimOutSampleRow[iPosX].iURef        = pcCurOutSampleRow[iPosX].iURef;

      pcCurLimOutSampleRow[iPosX].iVOther      = pcCurOutSampleRow[iPosX].*piSrcV;      
      pcCurLimOutSampleRow[iPosX].iVRef        = pcCurOutSampleRow[iPosX].iVRef;
#endif
      pcCurLimOutSampleRow[iPosX].iDOther      = pcCurOutSampleRow[iPosX].*piDepth;      
      pcCurLimOutSampleRow[iPosX].iFilledOther = pcCurOutSampleRow[iPosX].*piFilled;      
      pcCurLimOutSampleRow[iPosX].iError       = pcCurOutSampleRow[iPosX].iError;      

    } 
    pcCurOutSampleRow    += m_iOutputSamplesStride; 
    pcCurLimOutSampleRow += m_iOutputSamplesStride; 
  }
}

template <BlenMod iBM, Bool bBitInc> Void
#if RM_FIX_SETUP
TRenSingleModelC<iBM,bBitInc>::setStructSynthViewAsRefView( )
#else
TRenSingleModelC<iBM,bBitInc>::xSetStructSynthViewAsRefView( )
#endif
{
  AOT( m_iMode < 0 || m_iMode > 2);

  RenModelOutPels* pcCurOutSampleRow = m_pcOutputSamples;

  Pel RenModelOutPels::* piSrcY = NULL;

#if H_3D_VSO_COLOR_PLANES  
  Pel RenModelOutPels::* piSrcU = NULL;
  Pel RenModelOutPels::* piSrcV = NULL;
  xGetSampleStrTextPtrs( m_iMode, piSrcY, piSrcU, piSrcV );
#else
  xGetSampleStrTextPtrs( m_iMode, piSrcY );
#endif

  for ( Int iPosY = 0; iPosY < m_iUsedHeight; iPosY++ )
  {
    for ( Int iPosX = 0; iPosX < m_iWidth; iPosX++ )
    {      
      pcCurOutSampleRow[iPosX].iYRef = pcCurOutSampleRow[iPosX].*piSrcY;
#if H_3D_VSO_COLOR_PLANES
      pcCurOutSampleRow[iPosX].iURef = pcCurOutSampleRow[iPosX].*piSrcU;
      pcCurOutSampleRow[iPosX].iVRef = pcCurOutSampleRow[iPosX].*piSrcV;
#endif
    } 
    pcCurOutSampleRow += m_iOutputSamplesStride; 
  }
}

template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::xInitSampleStructs()
{
  RenModelOutPels* pcOutSampleRow      = m_pcOutputSamples;
  RenModelInPels * pcLeftInSampleRow   = m_pcInputSamples[0];
  RenModelInPels * pcRightInSampleRow  = m_pcInputSamples[1];


  for (Int iPosY = 0; iPosY < m_iHeight; iPosY++)
  {
    for (Int iPosX = 0; iPosX < m_iWidth; iPosX++)
    {
      //// Output Samples
      pcOutSampleRow[iPosX].iFilledLeft   = REN_IS_HOLE;
      pcOutSampleRow[iPosX].iFilledRight  = REN_IS_HOLE;

      pcOutSampleRow[iPosX].iDLeft        = 0;
      pcOutSampleRow[iPosX].iDRight       = 0;
      pcOutSampleRow[iPosX].iDBlended     = 0;      
      pcOutSampleRow[iPosX].iError        = 0; 
                                     
      // Y Planes                    
      pcOutSampleRow[iPosX].iYLeft        = 0;
      pcOutSampleRow[iPosX].iYRight       = 0;
      pcOutSampleRow[iPosX].iYBlended     = 0;
#if H_3D_VSO_COLOR_PLANES             
      // U Planes                    
      pcOutSampleRow[iPosX].iULeft        = 1 << (REN_BIT_DEPTH  - 1);
      pcOutSampleRow[iPosX].iURight       = 1 << (REN_BIT_DEPTH  - 1);
      pcOutSampleRow[iPosX].iUBlended     = 1 << (REN_BIT_DEPTH  - 1);
                                                  
      // V Planes                                  
      pcOutSampleRow[iPosX].iVLeft        = 1 << (REN_BIT_DEPTH  - 1);
      pcOutSampleRow[iPosX].iVRight       = 1 << (REN_BIT_DEPTH  - 1);
      pcOutSampleRow[iPosX].iVBlended     = 1 << (REN_BIT_DEPTH  - 1);
#endif
  //// Input Samples
      pcLeftInSampleRow [iPosX].aiOccludedPos = MAX_INT;
      pcRightInSampleRow[iPosX].aiOccludedPos = MIN_INT;
    }

    pcOutSampleRow     += m_iOutputSamplesStride;
    pcLeftInSampleRow  += m_iInputSamplesStride;
    pcRightInSampleRow += m_iInputSamplesStride;
  }  
}


#if H_3D_VSO_EARLY_SKIP
template <BlenMod iBM, Bool bBitInc> RMDist
TRenSingleModelC<iBM,bBitInc>::getDistDepth( Int iViewPos, Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData , const Pel * piOrgData, Int iOrgStride )
#else
template <BlenMod iBM, Bool bBitInc> RMDist
TRenSingleModelC<iBM,bBitInc>::getDistDepth( Int iViewPos, Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData )
#endif
{
  RMDist iSSE = 0;
#if H_3D_VSO_EARLY_SKIP
  Bool   bEarlySkip;
#endif
  switch ( iViewPos )
  {
  case 0:
#if H_3D_VSO_EARLY_SKIP
    bEarlySkip = m_bEarlySkip ? xDetectEarlySkip<true>(iStartPosX,   iStartPosY,   iWidth,   iHeight,   iStride, piNewData, piOrgData, iOrgStride) : false;
    if( !bEarlySkip )
    {
      iSSE = xRender<true, GET_SIMP>( iStartPosX,   iStartPosY,   iWidth,   iHeight,   iStride, piNewData,true );
    }    
#else
    iSSE = xRender<true, GET_SIMP>( iStartPosX,   iStartPosY,   iWidth,   iHeight,   iStride, piNewData );
#endif
    break;
  case 1:
#if H_3D_VSO_EARLY_SKIP
    bEarlySkip = m_bEarlySkip ? xDetectEarlySkip<false>(iStartPosX,   iStartPosY,   iWidth,   iHeight,   iStride, piNewData, piOrgData, iOrgStride) : false;
    if( !bEarlySkip )
    {
      iSSE = xRender<false, GET_SIMP>( iStartPosX,   iStartPosY,   iWidth,   iHeight,   iStride, piNewData,true );
    }    
#else
    iSSE = xRender<false, GET_SIMP>( iStartPosX,   iStartPosY,   iWidth,   iHeight,   iStride, piNewData );
#endif
    break;
  default:
    assert(0);
  }

  return iSSE;
}
#if H_3D_VSO_EARLY_SKIP
template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::setDepth( Int iViewPos, Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData, const Pel* piOrgData, Int iOrgStride )
#else
template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::setDepth( Int iViewPos, Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData )
#endif
{
#if  H_3D_VSO_EARLY_SKIP
  Bool bEarlySkip;
#endif
  switch ( iViewPos )
  {
  case 0:
#if H_3D_VSO_EARLY_SKIP
    bEarlySkip = m_bEarlySkip ? xDetectEarlySkip<true>(iStartPosX,   iStartPosY,   iWidth,   iHeight,   iStride, piNewData, piOrgData,iOrgStride) : false;
    if( !bEarlySkip )
    {
      xRender<true, SET_SIMP>( iStartPosX,   iStartPosY,   iWidth,   iHeight,   iStride, piNewData,true );
    }    
#else
    xRender<true, SET_SIMP>( iStartPosX,   iStartPosY,   iWidth,   iHeight,   iStride, piNewData );
#endif     
    break;
  case 1:
#if H_3D_VSO_EARLY_SKIP
    bEarlySkip = m_bEarlySkip ? xDetectEarlySkip<false>(iStartPosX,   iStartPosY,   iWidth,   iHeight,   iStride, piNewData, piOrgData,iOrgStride) : false;
    if( !bEarlySkip )
    {
      xRender<false, SET_SIMP>( iStartPosX,   iStartPosY,   iWidth,   iHeight,   iStride, piNewData,true ); 
    }    
#else
    xRender<false, SET_SIMP>( iStartPosX,   iStartPosY,   iWidth,   iHeight,   iStride, piNewData );
#endif     
    break;
  default:
    assert(0);
  }
}

template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::getSynthVideo( Int iViewPos, TComPicYuv* pcPicYuv )
{  
  AOT( pcPicYuv->getWidth( COMPONENT_Y  )  != m_iWidth );
  AOT( pcPicYuv->getChromaFormat()         != CHROMA_420 ); 

  AOT( pcPicYuv->getHeight( COMPONENT_Y ) < m_iUsedHeight + m_iHorOffset );

#if H_3D_VSO_COLOR_PLANES
  Pel RenModelOutPels::* piText[3] = { NULL, NULL, NULL };
  xGetSampleStrTextPtrs(iViewPos, piText[0], piText[1], piText[2]); 

  // Temp image for chroma down sampling
  PelImage cTempImage( m_iWidth, m_iUsedHeight, 3, 0);

  Int  aiStrides[3]; 
  Pel* apiData  [3]; 

  cTempImage.getDataAndStrides( apiData, aiStrides ); 

  for (UInt uiCurPlane = 0; uiCurPlane < 3; uiCurPlane++ )
  {
    xCopyFromSampleStruct( m_pcOutputSamples, m_iOutputSamplesStride, piText[uiCurPlane], apiData[uiCurPlane], aiStrides[uiCurPlane] , m_iWidth, m_iUsedHeight);
  }  
  xCopy2PicYuv( apiData, aiStrides, pcPicYuv );
#else
  Pel RenModelOutPels::* piY;
  xGetSampleStrTextPtrs(iViewPos, piY); 
  xCopyFromSampleStruct( m_pcOutputSamples, m_iOutputSamplesStride, piY, pcPicYuv->getAddr(COMPONENT_Y) + m_iHorOffset * pcPicYuv->getStride(COMPONENT_Y), pcPicYuv->getStride(COMPONENT_Y), m_iWidth, m_iUsedHeight );
  pcPicYuv->setChromaTo( 1 << (REN_BIT_DEPTH - 1) );   
#endif  
}

template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::getSynthDepth( Int iViewPos, TComPicYuv* pcPicYuv )
{  
  AOT( iViewPos != 0 && iViewPos != 1); 
  AOT( pcPicYuv->getWidth( COMPONENT_Y)  != m_iWidth  );
  AOT( pcPicYuv->getChromaFormat( )  != CHROMA_420 );
  AOT( pcPicYuv->getHeight( COMPONENT_Y ) < m_iUsedHeight + m_iHorOffset );

  Pel RenModelOutPels::* piD = 0;
  xGetSampleStrDepthPtrs(iViewPos, piD); 
  xCopyFromSampleStruct( m_pcOutputSamples, m_iOutputSamplesStride, piD, pcPicYuv->getAddr( COMPONENT_Y ) + pcPicYuv->getStride( COMPONENT_Y ) * m_iHorOffset, pcPicYuv->getStride( COMPONENT_Y ), m_iWidth, m_iUsedHeight );
  pcPicYuv->setChromaTo( 1 << (REN_BIT_DEPTH - 1) );   
}


template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::getRefVideo ( Int iViewPos, TComPicYuv* pcPicYuv )
{  
  AOT( pcPicYuv->getChromaFormat( ) != CHROMA_420 );
  AOT( pcPicYuv->getWidth( COMPONENT_Y )  != m_iWidth  );
  AOT( pcPicYuv->getHeight( COMPONENT_Y ) <  m_iUsedHeight + m_iHorOffset);

#if H_3D_VSO_COLOR_PLANES
  Pel RenModelOutPels::* piText[3];
  piText[0] = &RenModelOutPels::iYRef;
  piText[1] = &RenModelOutPels::iURef;
  piText[2] = &RenModelOutPels::iVRef;

  // Temp image for chroma down sampling

  PelImage cTempImage( m_iWidth, m_iUsedHeight, 3, 0);
  Int  aiStrides[3]; 
  Pel* apiData  [3]; 

  cTempImage.getDataAndStrides( apiData, aiStrides ); 

  for (UInt uiCurPlane = 0; uiCurPlane < 3; uiCurPlane++ )
  {
    xCopyFromSampleStruct( m_pcOutputSamples, m_iOutputSamplesStride, piText[uiCurPlane], apiData[uiCurPlane], aiStrides[uiCurPlane] , m_iWidth, m_iUsedHeight);
  }  

  xCopy2PicYuv( apiData, aiStrides, pcPicYuv );
#else
  xCopyFromSampleStruct( m_pcOutputSamples, m_iOutputSamplesStride, &RenModelOutPels::iYRef, pcPicYuv->getAddr(COMPONENT_Y), pcPicYuv->getStride(COMPONENT_Y), m_iWidth, m_iUsedHeight );
  pcPicYuv->setChromaTo( 1 << ( REN_BIT_DEPTH - 1 ) );   
#endif  
}

template <BlenMod iBM, Bool bBitInc> RMDist
TRenSingleModelC<iBM,bBitInc>::getDistVideo( Int iViewPos, Int iPlane, Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData )
{
  AOF(false);
  return 0;
}

template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::setVideo( Int iViewPos, Int iPlane, Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData )
{
  AOF(false);
}



template <BlenMod iBM, Bool bBitInc> template<SetMod bSM> __inline Void
TRenSingleModelC<iBM,bBitInc>::xSetViewRow( Int iPosY )
{
  m_pcInputSamplesRow[0] = m_pcInputSamples[0] + m_iInputSamplesStride  * iPosY;
  m_pcInputSamplesRow[1] = m_pcInputSamples[1] + m_iInputSamplesStride  * iPosY;
  if (bSM == SET_FULL || bSM == GET_FULL )
  {  
    m_pcOutputSamplesRow   = m_pcOutputSamples   + m_iOutputSamplesStride * iPosY;  
  }
  else
  {
    m_pcLimOutputSamplesRow   = m_pcLimOutputSamples  + m_iOutputSamplesStride * iPosY;  
  }
}

template <BlenMod iBM, Bool bBitInc> template<SetMod bSM> __inline Void
TRenSingleModelC<iBM,bBitInc>::xIncViewRow( )
{
  m_pcInputSamplesRow[0] += m_iInputSamplesStride ;
  m_pcInputSamplesRow[1] += m_iInputSamplesStride ;
  
  if (bSM == SET_FULL || bSM == GET_FULL )
  {  
    m_pcOutputSamplesRow   += m_iOutputSamplesStride;
  }
  else
  {
    m_pcLimOutputSamplesRow   += m_iOutputSamplesStride;
  }
}
#if H_3D_VSO_EARLY_SKIP
template <BlenMod iBM, Bool bBitInc> template<SetMod bSM> __inline RMDist 
TRenSingleModelC<iBM,bBitInc>::xGetSSE( Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData, Bool bFast)
#else
template <BlenMod iBM, Bool bBitInc> template<SetMod bSM> __inline RMDist 
TRenSingleModelC<iBM,bBitInc>::xGetSSE( Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData)
#endif
{
  const Int iCurViewPos   = 0;
  const Int iOtherViewPos = 1;

  m_piNewDepthData   = piNewData;  
  m_iNewDataWidth    = iWidth;     
  m_iStartChangePosX = iStartPosX; 

  if ((iWidth == 0) || (iHeight == 0))
    return 0;

  xSetViewRow<bSM>      ( iStartPosY);

  // Init Start
  RMDist iError = 0;     
  Int iStartChangePos = m_iStartChangePosX; 
  Int iEndChangePos   = m_iStartChangePosX + iWidth - 1;

  for (Int iPosY = iStartPosY; iPosY < iStartPosY + iHeight; iPosY++ )
  {    
    Int iPosXinNewData        = iWidth - 1;                       
    for ( Int iCurPosX = iEndChangePos; iCurPosX >= iStartChangePos; iCurPosX-- )
    {
      Int iCurDepth   = m_piNewDepthData[iPosXinNewData];
      Int iOldDepth   = m_pcInputSamplesRow[iCurViewPos][iCurPosX].iD; 
      Int iDiff = (iCurDepth - iOldDepth);
      iError += iDiff * iDiff;
      iPosXinNewData--; 
    }
    xIncViewRow<bSM>();
    m_piNewDepthData += iStride;
  }
  return iError;
}

#if H_3D_VSO_EARLY_SKIP
template <BlenMod iBM, Bool bBitInc> template<Bool bL, SetMod bSM> __inline RMDist 
TRenSingleModelC<iBM,bBitInc>::xRender( Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData, Bool bFast)
#else
template <BlenMod iBM, Bool bBitInc> template<Bool bL, SetMod bSM> __inline RMDist 
TRenSingleModelC<iBM,bBitInc>::xRender( Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData)
#endif
{
  const Int iCurViewPos   = bL ? 0 : 1;

  m_piNewDepthData   = piNewData;
  m_iNewDataWidth    = iWidth;
  m_iStartChangePosX = iStartPosX;

  if ((iWidth == 0) || (iHeight == 0))
  {
    return 0;
  }

  // Get Data
  m_ppiCurLUT      = m_appiShiftLut   [iCurViewPos];
  xSetViewRow<bSM>( iStartPosY);

  // Init Start
  RMDist iError = 0;
  Int   iStartChangePos;

  iStartChangePos = m_iStartChangePosX + ( bL ? 0  : (iWidth - 1));

  for (Int iPosY = iStartPosY; iPosY < iStartPosY + iHeight; iPosY++ )
  {
#if H_3D_VSO_EARLY_SKIP
    if( m_bEarlySkip && bFast )
    {
      if ( m_pbHorSkip[iPosY-iStartPosY] )
      {
        xIncViewRow<bSM>();
        m_piNewDepthData += iStride;
        continue;
      }
    }
#endif
    m_bInOcclusion = false;

    Int iLastSPos;
    Int iEndChangePos         = m_iStartChangePosX + ( bL ? (iWidth - 1) : 0 ) ;

    Int iEndChangePosInSubPel = iEndChangePos << m_iShiftPrec;
    Int iPosXinNewData        = bL ? iWidth - 1 : 0; 
    Int iMinChangedSPos       = bL ? m_iSampledWidth : -1;
    if ( iEndChangePos == xWidthMinus1<bL>() )
    {
      m_iCurDepth           = m_piNewDepthData[iPosXinNewData];
      Int iCurSPos          = xShiftDept(iEndChangePosInSubPel, m_iCurDepth );

      m_curRangeStart       = xRangeLeft<bL>( iCurSPos );
      xExtrapolateMargin<bL, bSM>  ( iCurSPos, iEndChangePos, iError );

      Int iOldDepth          = m_pcInputSamplesRow[iCurViewPos][iEndChangePos].iD;
      iMinChangedSPos        = xMin<Int, bL>( ( iOldDepth <= m_iCurDepth ) ? iCurSPos : xShiftDept(iEndChangePosInSubPel, iOldDepth ), iMinChangedSPos);
      iLastSPos           = iCurSPos;
      m_lastRangeStart    = m_curRangeStart;
      m_iLastDepth        = m_iCurDepth;
      m_iLastOccludedSPos = iLastSPos;

      if ( bSM == SET_FULL || bSM == SET_SIMP )
      {
        m_pcInputSamplesRow[iCurViewPos][iEndChangePos].iD = m_piNewDepthData[iPosXinNewData];
      }

      xDec<Int, bL>(iPosXinNewData);
      xDec<Int, bL>(iEndChangePos);
    }
    else
    {
      m_iLastDepth = m_pcInputSamplesRow [iCurViewPos][xPlus<Int,bL>(iEndChangePos,1)].iD;
      iLastSPos    = xShiftDept(xPlus<Int,bL>(iEndChangePosInSubPel, ( 1 << m_iShiftPrec ) ), m_iLastDepth );
      xInitRenderPart<bL>( iEndChangePos, iLastSPos );
    }

    //// RENDER NEW DATA
    Int iCurPosX;
    for ( iCurPosX = iEndChangePos; xGeQ<Int,bL>(iCurPosX,iStartChangePos); xDec<Int,bL>(iCurPosX))
    {
      Int iCurPosXInSubPel = iCurPosX << m_iShiftPrec;
      m_iCurDepth     = m_piNewDepthData[iPosXinNewData]        ;
      Int iCurSPos    = xShiftDept(iCurPosXInSubPel,m_iCurDepth);  

      Int iOldDepth   = m_pcInputSamplesRow[iCurViewPos][iCurPosX].iD;
      iMinChangedSPos = xMin<Int,bL>( ( iOldDepth <= m_iCurDepth ) ? iCurSPos : xShiftDept(iCurPosXInSubPel, iOldDepth ), iMinChangedSPos);

      xRenderRange<bL,bSM>(iCurSPos, iLastSPos, iCurPosX, iError );
      iLastSPos       = iCurSPos;
      m_iLastDepth    = m_iCurDepth;

      if ( bSM == SET_FULL || bSM == SET_SIMP )
      {
        m_pcInputSamplesRow[iCurViewPos][iCurPosX].iD = m_piNewDepthData[iPosXinNewData];
      }
      xDec<Int,bL>(iPosXinNewData);
    }

    //// RE-RENDER DATA LEFT TO NEW DATA

    while ( xGeQ<Int,bL>(iCurPosX, xZero<bL>() ) )
    {
      Int iCurPosXInSubPel = iCurPosX << m_iShiftPrec;
      m_iCurDepth  = m_pcInputSamplesRow[iCurViewPos][iCurPosX].iD;
      Int iCurSPos = xShiftDept(iCurPosXInSubPel,m_iCurDepth);      
      xRenderRange<bL,bSM>( iCurSPos, iLastSPos, iCurPosX, iError );
      if ( xLess<Int,bL>(iCurSPos,iMinChangedSPos) )
      {
        break;
      }

      xDec<Int,bL>(iCurPosX);
      iLastSPos    = iCurSPos;
      m_iLastDepth = m_iCurDepth;
    }




    xIncViewRow<bSM>();
    m_piNewDepthData += iStride;
  }
  return iError;
}

template <BlenMod iBM, Bool bBitInc> template<Bool bL> __inline Void
TRenSingleModelC<iBM,bBitInc>::xInitRenderPart(  Int iEndChangePos, Int iLastSPos )
{
  const Int iCurViewPos = bL ? 0 : 1;   
  m_iLastOccludedSPos = m_pcInputSamplesRow[iCurViewPos][ xPlus<Int,bL>(iEndChangePos,1) ].aiOccludedPos;  
  m_bInOcclusion      = xGeQ<Int,bL>( iLastSPos, m_iLastOccludedSPos ); 

  if( m_bInOcclusion )
  {
    m_lastRangeStart = xRound<bL>( m_iLastOccludedSPos );  
  }
  else
  {
    m_iLastOccludedSPos = iLastSPos; 
    m_lastRangeStart = xRangeLeft<bL>( iLastSPos );  
  }
};

template <BlenMod iBM, Bool bBitInc> template<Bool bL, SetMod bSM> __inline Void
TRenSingleModelC<iBM,bBitInc>::xRenderShiftedRange(Int iCurSPos, Int iLastSPos, Int iCurPos, RMDist& riError )
{
  RM_AOF( (xGeQ<Int,bL>(iLastSPos,iCurSPos)) );
  Int iDeltaSPos = bL ? iLastSPos - iCurSPos : iCurSPos - iLastSPos;

  m_curRangeStart = xRangeLeft<bL>( iCurSPos ); 
  if ( iDeltaSPos > m_iGapTolerance )
  {
    xFillHole<bL,bSM>( iCurSPos, iLastSPos, iCurPos, riError );
  }
  else
  {
    if (!xGeQ<Int,bL>(iLastSPos, bL ? 0 : ( m_iSampledWidth - 1) ))
    {
      return;
    }

    RM_AOT( iDeltaSPos    > m_iGapTolerance );

    m_iThisDepth = m_iCurDepth;

    for (Int iFillSPos = xMax<Int,bL>(xZero<bL>(), m_curRangeStart ); xLess<Int,bL>(iFillSPos,m_lastRangeStart); xInc<Int,bL>(iFillSPos))
    {
      Int iDeltaCurSPos  = (iFillSPos << m_iShiftPrec) - (bL ? iCurSPos : iLastSPos); 

      RM_AOT( iDeltaCurSPos > iDeltaSPos );
      RM_AOT( iDeltaCurSPos < 0 );
      RM_AOT( m_aaiSubPelShiftL[iDeltaSPos][iDeltaCurSPos] == 0xdeaddead);

      xSetShiftedPel<bL, bSM>( iCurPos, m_aaiSubPelShiftL[iDeltaSPos][iDeltaCurSPos], iFillSPos, REN_IS_FILLED, riError );
    }
  };
  m_lastRangeStart = m_curRangeStart; 
}

template <BlenMod iBM, Bool bBitInc> template<Bool bL, SetMod bSM> __inline Void
TRenSingleModelC<iBM,bBitInc>::xRenderRange(Int iCurSPos, Int iLastSPos, Int iCurPos, RMDist& riError )
{
  const Int iCurViewPos = bL ? 0 : 1; 

  if ( bSM == SET_FULL || bSM == SET_SIMP )
  {
    m_pcInputSamplesRow[iCurViewPos][ iCurPos ].aiOccludedPos = m_iLastOccludedSPos;
  }

  if ( xLess<Int,bL>(iCurSPos,m_iLastOccludedSPos ))
  {
    m_bInOcclusion      = false;
    m_iLastOccludedSPos = iCurSPos; 
    xRenderShiftedRange<bL,bSM>(iCurSPos, iLastSPos, iCurPos, riError );
  }
  else
  {
    if ( !m_bInOcclusion )
    {      
      RM_AOF( (xGeQ<Int,bL>(iLastSPos, m_iLastOccludedSPos)) ); 
      Int iRightSPosFP = xRound<bL>( iLastSPos );      
      if ( ( iRightSPosFP == xPlus<Int,bL>(m_lastRangeStart, -1) ) && xGeQ<Int,bL>(iRightSPosFP, xZero<bL>()) )
      {
        m_iThisDepth = m_iLastDepth;
        xSetShiftedPel<bL, bSM>( xPlus<Int,bL>(iCurPos,1), bL ? 0 : (1 << m_iShiftPrec), iRightSPosFP, REN_IS_FILLED, riError );
      }
      m_lastRangeStart = iRightSPosFP;
      m_bInOcclusion   = true; 
    }
  }
}

template <BlenMod iBM, Bool bBitInc> template<Bool bL, SetMod bSM> __inline Void
TRenSingleModelC<iBM,bBitInc>::xFillHole( Int iCurSPos, Int iLastSPos, Int iCurPos, RMDist& riError )
{
  if (iLastSPos < 0)
  {
    return;
  }

  Int iStartFillSPos = iCurSPos;
  Int iStartFillPos  = iCurPos;
  Int iLastPos       = xPlus<Int,bL>( iCurPos,1);

  Int iStartFillSPosFP = m_curRangeStart; 
  if (iStartFillSPosFP == xRound<bL>(iStartFillSPos))
  {
    if ( xGeQ<Int,bL>(iStartFillSPosFP, xZero<bL>())  && xLess<Int,bL>(iStartFillSPosFP, m_lastRangeStart) )
    {
      m_iThisDepth = m_iCurDepth;
      xSetShiftedPel<bL, bSM>    ( iStartFillPos, bL ? 0 : ( 1 << m_iShiftPrec), iStartFillSPosFP, REN_IS_FILLED, riError );
    }
  }
  else
  {
    xDec<Int,bL>( iStartFillSPosFP );
  }

  m_iThisDepth = m_iLastDepth;
  for (Int iFillSPos = xMax<Int,bL>(xPlus<Int,bL>(iStartFillSPosFP,1),xZero<bL>()); xLess<Int,bL>(iFillSPos, m_lastRangeStart); xInc<Int,bL>(iFillSPos))
  {
    xSetShiftedPel<bL, bSM>( iLastPos, bL ? 0 : (1 << m_iShiftPrec),  iFillSPos, REN_IS_HOLE, riError );
  }
}

template <BlenMod iBM, Bool bBitInc> template<Bool bL, SetMod bSM> __inline Void
TRenSingleModelC<iBM,bBitInc>::xExtrapolateMargin(Int iCurSPos, Int iCurPos, RMDist& riError )
{
  Int iSPosFullPel = xMax<Int,bL>(xZero<bL>(),m_curRangeStart);

  m_iThisDepth = m_iCurDepth;
  if ( xGeQ<Int,bL>(xWidthMinus1<bL>(), iSPosFullPel) )
  {
    xSetShiftedPel<bL, bSM>( iCurPos, bL ? 0 : (1 << m_iShiftPrec) , iSPosFullPel, REN_IS_FILLED, riError );
  }
  for (Int iFillSPos = xPlus<Int,bL>(iSPosFullPel ,1); xGeQ<Int,bL>( xWidthMinus1<bL>(), iFillSPos ); xInc<Int,bL>(iFillSPos))
  {
    xSetShiftedPel<bL, bSM>( iCurPos, bL ? 0 : ( 1 << m_iShiftPrec ), iFillSPos, REN_IS_HOLE, riError );
  }
}

template <BlenMod iBM, Bool bBitInc> template <Bool bL> __inline Int
TRenSingleModelC<iBM,bBitInc>::xShiftNewData( Int iPosX, Int iPosInNewData )
{
  RM_AOT( iPosInNewData <               0 );
  RM_AOF( iPosInNewData < m_iNewDataWidth );
  return (iPosX << m_iShiftPrec) - m_ppiCurLUT[0][ RenModRemoveBitInc( m_piNewDepthData[iPosInNewData] )];
}


template <BlenMod iBM, Bool bBitInc> __inline Int
TRenSingleModelC<iBM,bBitInc>::xShiftDept( Int iPosXinSubPel, Int iDepth )
{
  return (iPosXinSubPel) - m_ppiCurLUT[0][ RenModRemoveBitInc( iDepth )];
}


template <BlenMod iBM, Bool bBitInc> template <Bool bL> __inline Int
TRenSingleModelC<iBM,bBitInc>::xShift( Int iPosX )
{
 RM_AOT( iPosX <        0);
 RM_AOF( iPosX < m_iWidth);
 return (iPosX  << m_iShiftPrec) - m_ppiCurLUT[0][ RenModRemoveBitInc( m_pcInputSamplesRow[(bL ? 0 : 1)][iPosX].iD )];
}


template <BlenMod iBM, Bool bBitInc> template <Bool bL> __inline Int
TRenSingleModelC<iBM,bBitInc>::xShift( Int iPos, Int iPosInNewData )
{
  if ( (iPosInNewData >= 0) && (iPosInNewData < m_iNewDataWidth) )
  {
    return xShiftNewData(iPos ,iPosInNewData );
  }
  else
  {
    return xShift<bL>(iPos);
  }
}

template <BlenMod iBM, Bool bBitInc> template<Bool bL> __inline Int
TRenSingleModelC<iBM,bBitInc>::xRangeLeft( Int iPos )
{
  if ( bL )
  {
    return  ( iPos +  (1 << m_iShiftPrec) - 1) >> m_iShiftPrec;
  }
  else
  {
    return iPos >> m_iShiftPrec;
  }
}



template <BlenMod iBM, Bool bBitInc> template<Bool bL> __inline Int
TRenSingleModelC<iBM,bBitInc>::xRangeRight( Int iPos )
{
  if ( bL )
  {
    return xRangeLeft<true>(iPos)    - 1;
  }
  else
  {
    return xRangeLeft<false>( iPos ) + 1;   
  }  
}

template <BlenMod iBM, Bool bBitInc> template<Bool bL> __inline Int
TRenSingleModelC<iBM,bBitInc>::xRound( Int iPos )
{
  if( bL )
  {  
    return  (iPos + (( 1 << m_iShiftPrec ) >> 1 )) >> m_iShiftPrec;
  }
  else
  {
    return  (m_iShiftPrec == 0) ? iPos : xRound<true>(iPos - 1);
  }
}


template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::xSetPels( Pel* piPelSource , Int iSourceStride, Int iWidth, Int iHeight, Pel iVal )
{
  for (Int iYPos = 0; iYPos < iHeight; iYPos++)
  {
    for (Int iXPos = 0; iXPos < iWidth; iXPos++)
    {
      piPelSource[iXPos] = iVal;
    }
    piPelSource += iSourceStride;
  }
}

template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::xSetInts( Int* piPelSource , Int iSourceStride, Int iWidth, Int iHeight, Int iVal )
{
  for (Int iYPos = 0; iYPos < iHeight; iYPos++)
  {
    for (Int iXPos = 0; iXPos < iWidth; iXPos++)
    {
      piPelSource[iXPos] = iVal;
    }
    piPelSource += iSourceStride;
  }
}


template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::xSetBools( Bool* pbPelSource , Int iSourceStride, Int iWidth, Int iHeight, Bool bVal )
{
  for (Int iYPos = 0; iYPos < iHeight; iYPos++)
  {
    for (Int iXPos = 0; iXPos < iWidth; iXPos++)
    {
      pbPelSource[iXPos] = bVal;
    }
    pbPelSource += iSourceStride;
  }
}

template <BlenMod iBM, Bool bBitInc> template<Bool bL, SetMod bSM> __inline Void
TRenSingleModelC<iBM,bBitInc>::xSetShiftedPel(Int iSourcePos, Int iSubSourcePos, Int iTargetSPos, Pel iFilled, RMDist& riError )
{
  RM_AOT( iSourcePos    <  0                   );
  RM_AOT( iSourcePos    >= m_iWidth            );
  RM_AOT( iSubSourcePos < 0                    );
  RM_AOT( iSubSourcePos >  (1 << m_iShiftPrec) );
  RM_AOT( iTargetSPos   < 0                    );
  RM_AOT( iTargetSPos   >= m_iWidth            );

  RenModelInPels * pcInSample  = m_pcInputSamplesRow[ bL ? VIEWPOS_LEFT : VIEWPOS_RIGHT ] + iSourcePos ;

  Pel iY;
  Pel iYCurNew = pcInSample->aiY[iSubSourcePos];    
#if H_3D_VSO_COLOR_PLANES
  Pel iU;
  Pel iUCurNew = pcInSample->aiU[iSubSourcePos];
  Pel iV;
  Pel iVCurNew = pcInSample->aiV[iSubSourcePos];;
#endif    

  const Bool bFullMode = ( bSM == GET_FULL || bSM == SET_FULL ); 

  RenModelOutPels*    pcOutSample    = bFullMode ? ( m_pcOutputSamplesRow    + iTargetSPos ) : NULL; 
  RenModelLimOutPels* pcLimOutSample = bFullMode ? NULL  : ( m_pcLimOutputSamplesRow + iTargetSPos ); 

  if ( iBM == BLEND_NONE )
  {
    iY = iYCurNew; 
#if H_3D_VSO_COLOR_PLANES
    iU = iUCurNew; 
    iV = iVCurNew; 
#endif
  }
  else
  { 
    Pel iYOther      = bFullMode ? ( bL ? pcOutSample->iYRight : pcOutSample->iYLeft) : pcLimOutSample->iYOther; 
#if H_3D_VSO_COLOR_PLANES
    Pel iUOther      = bFullMode ? ( bL ? pcOutSample->iURight : pcOutSample->iULeft ) : pcLimOutSample->iUOther; 
    Pel iVOther      = bFullMode ? ( bL ? pcOutSample->iVRight : pcOutSample->iVLeft ) : pcLimOutSample->iVOther; 
#endif
    Int iFilledOther = bFullMode ? ( bL ? pcOutSample->iFilledRight : pcOutSample->iFilledLeft ) : pcLimOutSample->iFilledOther; 
    Pel iDOther      = bFullMode ? ( bL ? pcOutSample->iDRight      : pcOutSample->iDLeft      ) : pcLimOutSample->iDOther; 

    xGetBlendedValue<bL, bSM>(
      iY,
      bL ? iYCurNew : iYOther,
      bL ? iYOther  : iYCurNew,
#if H_3D_VSO_COLOR_PLANES
      iU,
      bL ? iUCurNew  : iUOther,
      bL ? iUOther   : iUCurNew,
      iV,
      bL ? iVCurNew  : iVOther,
      bL ? iVOther   : iVCurNew,          
#endif
      bL ? iFilled      : iFilledOther,
      bL ? iFilledOther : iFilled,
      m_piInvZLUTLeft [RenModRemoveBitInc( bL ? m_iThisDepth : iDOther)],
      m_piInvZLUTRight[RenModRemoveBitInc( bL ? iDOther      : m_iThisDepth)]
      ); 
  }


  Int iDist = xGetDist( 
    iY - ( bFullMode ? pcOutSample->iYRef : pcLimOutSample->iYRef ) 
#if H_3D_VSO_COLOR_PLANES
  , iU - ( bFullMode ? pcOutSample->iURef : pcLimOutSample->iURef )
  , iV - ( bFullMode ? pcOutSample->iVRef : pcLimOutSample->iVRef )
#endif
  );

  if ( bSM == GET_FULL || bSM == GET_SIMP )
  {    
    riError += ( iDist - ( bFullMode ? pcOutSample->iError : pcLimOutSample->iError ) );
  }
  else // bSM == SET_FULL
  {
    Int& riErrorStr = bFullMode ? pcOutSample->iError : pcLimOutSample->iError; 
    riErrorStr      = iDist; 

    if ( bFullMode )
    {     
      if ( iBM != BLEND_NONE )
      {        
        pcOutSample->iYBlended   = iY; 
#if H_3D_VSO_COLOR_PLANES
        pcOutSample->iUBlended   = iU; 
        pcOutSample->iVBlended   = iV; 
#endif
      }

      if ( bL )
      {
        pcOutSample->iDLeft      = m_iThisDepth; 
        pcOutSample->iFilledLeft = iFilled; 
        pcOutSample->iYLeft      = iYCurNew;
#if  H_3D_VSO_COLOR_PLANES
        pcOutSample->iULeft      = iUCurNew;
        pcOutSample->iVLeft      = iVCurNew;
#endif
      }
      else
      {
        pcOutSample->iDRight      = m_iThisDepth; 
        pcOutSample->iFilledRight = iFilled; 
        pcOutSample->iYRight      = iYCurNew;
#if  H_3D_VSO_COLOR_PLANES
        pcOutSample->iURight      = iUCurNew;
        pcOutSample->iVRight      = iVCurNew;
#endif
      }
    }
  }    
}

template <BlenMod iBM, Bool bBitInc> __inline Int
TRenSingleModelC<iBM,bBitInc>::xGetDist( Int iDiffY, Int iDiffU, Int iDiffV )
{

  if ( !bBitInc )
  {
    return (          (iDiffY * iDiffY )
               +  ((( (iDiffU * iDiffU )
                     +(iDiffV * iDiffV )
                    )
                   ) >> 2
                  )
           );
  }
  else
  {
    return (          ((iDiffY * iDiffY) >> m_iDistShift)
               +  ((( ((iDiffU * iDiffU) >> m_iDistShift)
                     +((iDiffV * iDiffV) >> m_iDistShift)
                    )
                   ) >> 2
                  )
           );
  
  }
}

template <BlenMod iBM, Bool bBitInc> __inline Int
TRenSingleModelC<iBM,bBitInc>::xGetDist( Int iDiffY )
{
  if ( !bBitInc )
  {
    return (iDiffY * iDiffY);
  }
  else
  {
    return ((iDiffY * iDiffY) >> m_iDistShift);
  }

}



template <BlenMod iBM, Bool bBitInc>  template< Bool bL, SetMod bSM > __inline Void
  TRenSingleModelC<iBM,bBitInc>::xGetBlendedValue( Pel& riY, Pel iYL,  Pel iYR,  
#if H_3D_VSO_COLOR_PLANES
                                                   Pel& riU, Pel iUL,  Pel iUR,  Pel& riV, Pel iVL,  Pel iVR,  
#endif
                                                   Int iFilledL,  Int iFilledR, Pel iDepthL,  Pel iDepthR  )
{  
  RM_AOT( iBM != BLEND_AVRG && iBM != BLEND_LEFT && iBM != BLEND_RIGHT );

  if (iBM != BLEND_AVRG )
  {
    if (iBM == BLEND_LEFT )
    {
      xGetBlendedValueBM1<bL, bSM>( riY, iYL,  iYR,  
#if H_3D_VSO_COLOR_PLANES
        riU, iUL,  iUR,  riV, iVL,  iVR,  
#endif
        iFilledL,  iFilledR, iDepthL,  iDepthR  );
    }
    else
    {
      xGetBlendedValueBM2<bL, bSM>(  riY, iYL,  iYR,  
#if H_3D_VSO_COLOR_PLANES
        riU, iUL,  iUR,  riV, iVL,  iVR,  
#endif
        iFilledL,  iFilledR, iDepthL,  iDepthR );
    }
    return;
  }

  if (  (iFilledL != REN_IS_HOLE ) && ( iFilledR != REN_IS_HOLE) )
  {
    Int iDepthDifference = iDepthR - iDepthL;

    if ( abs ( iDepthDifference ) <= m_iBlendZThres )
    {
      {
        riY = xBlend( iYL, iYR, m_iBlendDistWeight );
#if H_3D_VSO_COLOR_PLANES    
        riU = xBlend( iUL, iUR, m_iBlendDistWeight );
        riV = xBlend( iVL, iVR, m_iBlendDistWeight );
#endif
      }
    }
    else if ( iDepthDifference < 0 )
    {
      riY = iYL;
#if H_3D_VSO_COLOR_PLANES
      riU = iUL;
      riV = iVL;
#endif
    }
    else
    {      
      riY = iYR;
#if H_3D_VSO_COLOR_PLANES
      riU = iUR;
      riV = iVR;
#endif
    }
  }
  else if ( (iFilledL == REN_IS_HOLE) && (iFilledR == REN_IS_HOLE))
  {
    if ( iDepthR < iDepthL )
    {
        riY =  iYR;
#if H_3D_VSO_COLOR_PLANES
        riU =  iUR;
        riV =  iVR;
#endif
    }
    else
    {
        riY =  iYL;
#if H_3D_VSO_COLOR_PLANES
        riU =  iUL;
        riV =  iVL;
#endif
    }
  }
  else
  {
    if (iFilledR == REN_IS_HOLE)
    {
        riY = iYL;
#if H_3D_VSO_COLOR_PLANES
        riU = iUL;
        riV = iVL;
#endif
    }
    else
    {
      riY = iYR;
#if H_3D_VSO_COLOR_PLANES
      riU = iUR;
      riV = iVR;
#endif
    }
  }
}

template <BlenMod iBM, Bool bBitInc> template< Bool bL, SetMod SM > __inline Void
TRenSingleModelC<iBM,bBitInc>::xGetBlendedValueBM1( Pel& riY, Pel iYL,  Pel iYR,  
#if H_3D_VSO_COLOR_PLANES
                                                    Pel& riU, Pel iUL,  Pel iUR,  Pel& riV, Pel iVL,  Pel iVR,  
#endif
                                                    Int iFilledL,  Int iFilledR, Pel iDepthL,  Pel iDepthR  )
{
  if ( iFilledL == REN_IS_FILLED ||  iFilledR == REN_IS_HOLE )
  {
    riY = iYL;
#if H_3D_VSO_COLOR_PLANES
    riU = iUL;
    riV = iVL;
#endif
  }
  else if ( iFilledL == REN_IS_HOLE  )
  {
    riY = iYR;
#if H_3D_VSO_COLOR_PLANES
    riU = iUR;
    riV = iVR;
#endif
  }
  else
  {
    riY = xBlend( iYR, iYL, iFilledL );
#if H_3D_VSO_COLOR_PLANES
    riU = xBlend( iUR, iUL, iFilledL );
    riV = xBlend( iVR, iUL, iFilledL );
#endif
  }
}

template <BlenMod iBM, Bool bBitInc> template< Bool bL, SetMod SM > __inline Void
  TRenSingleModelC<iBM,bBitInc>::xGetBlendedValueBM2( Pel& riY, Pel iYL,  Pel iYR,  
#if H_3D_VSO_COLOR_PLANES
                                                      Pel& riU, Pel iUL,  Pel iUR,  Pel& riV, Pel iVL,  Pel iVR,  
#endif
                                                      Int iFilledL,  Int iFilledR, Pel iDepthL,  Pel iDepthR  )
{
  if      ( iFilledR == REN_IS_FILLED ||  iFilledL == REN_IS_HOLE )
  {
    riY = iYR;
#if H_3D_VSO_COLOR_PLANES
    riU = iUR;
    riV = iVR;
#endif
  }
  else if ( iFilledR == REN_IS_HOLE  )
  {
    riY = iYL;
#if H_3D_VSO_COLOR_PLANES
    riU = iUL;
    riV = iVL;
#endif
  }
  else
  {
    riY = xBlend( iYL, iYR, iFilledR );
#if H_3D_VSO_COLOR_PLANES
    riU = xBlend( iUL, iUR, iFilledR );
    riV = xBlend( iVL, iUR, iFilledR );
#endif
  }
}

template <BlenMod iBM, Bool bBitInc> __inline Pel
TRenSingleModelC<iBM,bBitInc>::xBlend( Pel pVal1, Pel pVal2, Int iWeightVal2 )
{
  return pVal1  +  (Pel) (  ( (Int) ( pVal2 - pVal1) * iWeightVal2 + (1 << (REN_VDWEIGHT_PREC - 1)) ) >> REN_VDWEIGHT_PREC );
}

template <BlenMod iBM, Bool bBitInc> Void
TRenSingleModelC<iBM,bBitInc>::xCopy2PicYuv( Pel** ppiSrcVideoPel, Int* piStrides, TComPicYuv* rpcPicYuvTarget )
{
  TRenFilter<REN_BIT_DEPTH>::copy            ( ppiSrcVideoPel[0], piStrides[0], m_iWidth, m_iUsedHeight, rpcPicYuvTarget->getAddr( COMPONENT_Y  ) +  m_iHorOffset       * rpcPicYuvTarget->getStride( COMPONENT_Y  ), rpcPicYuvTarget->getStride( COMPONENT_Y ) );
  TRenFilter<REN_BIT_DEPTH>::sampleDown2Tap13( ppiSrcVideoPel[1], piStrides[1], m_iWidth, m_iUsedHeight, rpcPicYuvTarget->getAddr( COMPONENT_Cb ) + (m_iHorOffset >> 1) * rpcPicYuvTarget->getStride( COMPONENT_Cb ), rpcPicYuvTarget->getStride( COMPONENT_Cb) );
  TRenFilter<REN_BIT_DEPTH>::sampleDown2Tap13( ppiSrcVideoPel[2], piStrides[2], m_iWidth, m_iUsedHeight, rpcPicYuvTarget->getAddr( COMPONENT_Cr ) + (m_iHorOffset >> 1) * rpcPicYuvTarget->getStride( COMPONENT_Cr ), rpcPicYuvTarget->getStride( COMPONENT_Cr) );
}

template class TRenSingleModelC<BLEND_NONE ,true>;
template class TRenSingleModelC<BLEND_AVRG ,true>;
template class TRenSingleModelC<BLEND_LEFT ,true>;
template class TRenSingleModelC<BLEND_RIGHT,true>;

template class TRenSingleModelC<BLEND_NONE ,false>;
template class TRenSingleModelC<BLEND_AVRG ,false>;
template class TRenSingleModelC<BLEND_LEFT ,false>;
template class TRenSingleModelC<BLEND_RIGHT,false>;

#if H_3D_VSO_EARLY_SKIP
template <BlenMod iBM, Bool bBitInc> template <Bool bL > __inline Bool
TRenSingleModelC<iBM,bBitInc>::xDetectEarlySkip( Int iStartPosX, Int iStartPosY, Int iWidth, Int iHeight, Int iStride, const Pel* piNewData, const Pel* piOrgData, Int iOrgStride)
{
  RM_AOF( m_bEarlySkip ); 
  const Int iCurViewPos = bL ? 0 : 1;
  Int** ppiCurLUT       = m_appiShiftLut   [ iCurViewPos ];
  
  Bool bNoDiff          = true;   
  
  for (Int iPosY=0; iPosY < iHeight; iPosY++)
  {
    m_pbHorSkip[iPosY] = true;

    for (Int iPosX = 0; iPosX < iWidth; iPosX++)
    {
      Int iDisparityRec = abs(ppiCurLUT[0][ RenModRemoveBitInc(piNewData[iPosX])]);
      Int iDispartyOrg  = abs(ppiCurLUT[0][ RenModRemoveBitInc(piOrgData[iPosX])]);

      if( iDispartyOrg != iDisparityRec)
      {
        m_pbHorSkip[iPosY] = false;
        bNoDiff            = false;
        break;
      }
    }
    piNewData += iStride;
    piOrgData += iOrgStride;
  }
  return bNoDiff;
}
#endif
#endif // NH_3D


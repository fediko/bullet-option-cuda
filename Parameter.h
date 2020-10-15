/**********************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code 
the name of the author above.
**********************************************************************/

#ifndef __CONSTANTS_TYPES__
#define __CONSTANTS_TYPES__

////////////////Algorithm Parameters///////////////
#define NbOuter (512)	              //Number of MC trajectories for outer simulation
#define NbInner (1024)                //Number of MC trajectories for inner simulation
#define NbInTimes (1)                 //Serial increase of inner trajectories
#define TotInner (NbInner*NbInTimes)  //Total number of inner trajectories 
#define Total (NbOuter*NbInner)       //Total number of trajectories
////////////////////////////////////////////////////

////// Management proposal of blocks and threads////////
#define BlockOuter (NbOuter<(256) ? NbOuter:256)  //256
#define GridOuter (NbOuter/BlockOuter) //2
#define MinNbT 256
#define GridInnerY ( (NbInner) < (MinNbT) ? (1):(NbInner/MinNbT)) //4
#define BlockInnerX ( (NbInner) < (MinNbT) ? (NbInner):(MinNbT)) //256
#define GridInnerX ( (NbInner) < (MinNbT) ? (Total/MinNbT):(NbOuter) ) //512
#define BlockInnerY (NbOuter/GridInnerX) //1
#define BlockInner (128)
///////////////////////////////////////////////////////


// Pi approximation needed in some kernels
#define MoPI (3.1415927f)
////////////////////////////////////////////////////////////////
// L'Eucuyer CMRG Matrix Values
////////////////////////////////////////////////////////////////
// First MRG
#define a12 63308
#define a13 -183326
#define q12 33921
#define q13 11714
#define r12 12979
#define r13 2883

// Second MRG
#define a21 86098
#define a23 -539608
#define q21 24919
#define q23 3976
#define r21 7417
#define r23 2071

// Normalization variables
#define Invmp 4.6566129e-10f
#define two17 131072.0
#define two53 9007199254740992.0

typedef int TabSeedCMRG_t[NbInner][NbOuter][6];
typedef float Tab2RNG_t[NbInner][NbOuter][2];

#endif
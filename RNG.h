/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include "Parameter.h"
//#include <stdio.h>


////////////////////////////////////////////////////////////////
// Memory for RNG use 
////////////////////////////////////////////////////////////////
// The state variables of CMRG on GPU 
extern TabSeedCMRG_t *pt_CMRG;
// The combination of CMRGs, the used RNGs
extern Tab2RNG_t *pt_2RNG;
// The state variables of CMRG on CPU
extern TabSeedCMRG_t *pt_CMRGCPU;
// Matrices associated to the post treatment of the CMRG
// - First MRG
extern double A1[3][3];
// - Second MRG
extern double A2[3][3];

////////////////////////////////////////////////////////////////
// Post initialization of CMRG
////////////////////////////////////////////////////////////////
void PostInitDataCMRG(void);


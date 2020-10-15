/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/

#include "RNG.h"



void testCUDA(cudaError_t error, const char *file, int line){
   if (error != cudaSuccess){
      printf("There is an error in file %s at line %d\n", file ,line);
      printf("GPUassert: %s %s %d\n", cudaGetErrorString(error), file, line);
      exit(EXIT_FAILURE);
   }
}

#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

////////////////////////////////////////////////////////////////
// Memory for RNG use 
////////////////////////////////////////////////////////////////
// The state variables of CMRG on GPU 
 TabSeedCMRG_t *pt_CMRG;
// The combination of CMRGs, the used RNGs
 Tab2RNG_t *pt_2RNG;
// The state variables of CMRG on CPU
TabSeedCMRG_t CMRGCPU;
TabSeedCMRG_t *pt_CMRGCPU = &CMRGCPU;
// Matrices associated to the post treatment of the CMRG
// - First MRG
 double A1[3][3];
// - Second MRG
 double A2[3][3];

// Functions from http://www.iro.umontreal.ca/~lecuyer/myftp/streams00/c/


/*-------------------------------------------------------------------------*/

static double MultModM (double a, double s, double c, double m)
   /* Compute (a*s + c) % m. m must be < 2^35.  Works also for s, c < 0 */
{
   double v;
   long a1;
   v = a * s + c;
   if ((v >= two53) || (v <= -two53)) {
      a1 = (long) (a / two17);
      a -= a1 * two17;
      v = a1 * s;
      a1 = (long) (v / m);
      v -= a1 * m;
      v = v * two17 + a * s + c;
   }
   a1 = (long) (v / m);
   if ((v -= a1 * m) < 0.0)
      return v += m;
   else
      return v;
}


/*-------------------------------------------------------------------------*/

static void MatVecModM (double A[3][3], double s[3], double v[3], double m)
   /* Returns v = A*s % m.  Assumes that -m < s[i] < m. */
   /* Works even if v = s. */
{
   int i;
   double x[3];
   for (i = 0; i < 3; ++i) {
      x[i] = MultModM (A[i][0], s[0], 0.0, m);
      x[i] = MultModM (A[i][1], s[1], x[i], m);
      x[i] = MultModM (A[i][2], s[2], x[i], m);
   }
   for (i = 0; i < 3; ++i)
      v[i] = x[i];
}


/*-------------------------------------------------------------------------*/

static void MatMatModM (double A[3][3], double B[3][3], double C[3][3],
                        double m)
   /* Returns C = A*B % m. Work even if A = C or B = C or A = B = C. */
{
   int i, j;
   double V[3], W[3][3];
   for (i = 0; i < 3; ++i) {
      for (j = 0; j < 3; ++j)
         V[j] = B[j][i];
      MatVecModM (A, V, V, m);
      for (j = 0; j < 3; ++j)
         W[j][i] = V[j];
   }
   for (i = 0; i < 3; ++i) {
      for (j = 0; j < 3; ++j)
         C[i][j] = W[i][j];
   }
}


/*-------------------------------------------------------------------------*/

static void MatPowModM (double A[3][3], double B[3][3], double m, long n)
   /* Compute matrix B = A^n % m ;  works even if A = B */
{
   int i, j;
   double W[3][3];

   // initialize: W = A; B = I 
   for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; ++j) {
         W[i][j] = A[i][j];
         B[i][j] = 0.0;
      }
   }
   for (j = 0; j < 3; ++j)
      B[j][j] = 1.0;

   // Compute B = A^n % m using the binary decomposition of n 
   while (n > 0) {
      if (n % 2)
         MatMatModM (W, B, B, m);
      MatMatModM (W, W, W, m);
      n /= 2;
   }
}

////////////////////////////////////////////////////////////////
// Post initialization of CMRG
////////////////////////////////////////////////////////////////
void PostInitDataCMRG()
{
 const int m1 = 2147483647; // Requested for the simulation
 const int m2 = 2145483479; // Requested  for the simulation
 int i, j;					// loop indices
 // + Init of the posttreatment table for CMRG rng (should
 //   be turned into a file reading in the future)
   A1[0][0] = 2119409629.0; A1[0][1] =  302707381.0; A1[0][2] =  655487731.0;
   A1[1][0] =  946145520.0; A1[1][1] = 1762689149.0; A1[1][2] =  302707381.0;
   A1[2][0] = 1139076568.0; A1[2][1] =  600956040.0; A1[2][2] = 1762689149.0;

   A2[0][0] = 1705536521.0; A2[0][1] = 1409357255.0; A2[0][2] = 1489714515.0;
   A2[1][0] = 1443451163.0; A2[1][1] = 1705536521.0; A2[1][2] = 1556328147.0;
   A2[2][0] = 1624922073.0; A2[2][1] = 1443451163.0; A2[2][2] =  130172503.0;

 MatPowModM (A1, A1, m1, 1073741824);//utiliser des puissances de 2
 MatPowModM (A2, A2, m2, 1073741824);//utiliser des puissances de 2
 
////////////////////////////////////////////////////////////////
// Local variables used in the initialization 
////////////////////////////////////////////////////////////////
 double s1[3] = {1.0, 1.0, 1.0};
 double s2[3] = {1.0, 1.0, 1.0};

 for (i = 0; i < NbInner; i++) {
    
	for (j = 0; j < NbOuter; j++) {
	   pt_CMRGCPU[0][i][j][0] = (int) s1[0]; 
      pt_CMRGCPU[0][i][j][1] = (int) s1[1];
	   pt_CMRGCPU[0][i][j][2] = (int) s1[2];
	   pt_CMRGCPU[0][i][j][3] = (int) s2[0];
	   pt_CMRGCPU[0][i][j][4] = (int) s2[1];
      pt_CMRGCPU[0][i][j][5] = (int) s2[2];
      
	   MatVecModM (A1, s1, s1, m1);
      MatVecModM (A2, s2, s2, m2);
	}
 }
 testCUDA(cudaMalloc((void**)&pt_CMRG, NbInner*NbOuter*6*sizeof(int)));

 // - Copy CMRG data on the GPU
 testCUDA(cudaMemcpy(pt_CMRG, pt_CMRGCPU, sizeof(TabSeedCMRG_t), cudaMemcpyHostToDevice));
}

/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code 
the name of the author above.
***************************************************************/
#include <iostream>
#include <fstream>
#include "Parameter.h"


#include "RNG.h"
void testCUDA(cudaError_t error, const char *file, int line);
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))



// Generate uniformly distributed random variables
__device__ void CMRG_d(int *a0, int *a1, int *a2, int *a3, int *a4, 
			     int *a5, float *g0, float *g1, int nb){

 const int m1 = 2147483647;// Requested for the simulation
 const int m2 = 2145483479;// Requested for the simulation
 int h, p12, p13, p21, p23, k, loc;// Requested local parameters

 for(k=0; k<nb; k++){
	 // First Component 
	 h = *a0/q13; 
	 p13 = a13*(h*q13-*a0)-h*r13;
	 h = *a1/q12; 
	 p12 = a12*(*a1-h*q12)-h*r12;

	 if (p13 < 0) {
	   p13 = p13 + m1;
	 }
	 if (p12 < 0) {
	   p12 = p12 + m1;
	 }
	 *a0 = *a1;
	 *a1 = *a2;
	 if( (p12 - p13) < 0){
	   *a2 = p12 - p13 + m1;  
	 } else {
	   *a2 = p12 - p13;
	 }
  
	 // Second Component 
	 h = *a3/q23; 
	 p23 = a23*(h*q23-*a3)-h*r23;
	 h = *a5/q21; 
	 p21 = a21*(*a5-h*q21)-h*r21;

	 if (p23 < 0){
	   p23 = p23 + m2;
	 }
	 if (p12 < 0){
	   p21 = p21 + m2;
	 }
	 *a3 = *a4;
	 *a4 = *a5;
	 if ( (p21 - p23) < 0) {
	   *a5 = p21 - p23 + m2;  
	 } else {
	   *a5 = p21 - p23;
	 }

	 // Combines the two MRGs
	 if(*a2 < *a5){
		loc = *a2 - *a5 + m1;
	 }else{loc = *a2 - *a5;} 

	 if(k){
		if(loc == 0){
			*g1 = Invmp*m1;
		}else{*g1 = Invmp*loc;}
	 }else{
		*g1 = 0.0f; 
		if(loc == 0){
			*g0 = Invmp*m1;
		}else{*g0 = Invmp*loc;}
	 }
  }
}

// Genrates Gaussian distribution from a uniform one (Box-Muller)
__device__ void BoxMuller_d(float *g0, float *g1){

  float loc;
  if (*g1 < 1.45e-6f){
    loc = sqrtf(-2.0f*logf(0.00001f))*cosf(*g0*2.0f*MoPI);
  } else {
    if (*g1 > 0.99999f){
      loc = 0.0f;
    } else {loc = sqrtf(-2.0f*logf(*g1))*cosf(*g0*2.0f*MoPI);}
  }
  *g0 = loc;
}

// Black & Scholes model
__device__ void BS_d(float *S2, float S1, float r0,
					 float sigma, float dt, float e){

  *S2 = S1*expf((r0-0.5f*sigma*sigma)*dt*dt + sigma*dt*e);
}
		
__device__ void CMRG_get_d(int *a0, int *a1, int *a2, 
							int *a3, int *a4, int *a5, 
							int *CMRG_In){
	*a0 = CMRG_In[0];
	*a1 = CMRG_In[1];
	*a2 = CMRG_In[2];
	*a3 = CMRG_In[3];
	*a4 = CMRG_In[4];
	*a5 = CMRG_In[5];
}


__global__ void outer_k(float x_0, float r0, 
						float sigma, float dt, int P1, int P2,
						float K,
						float B, int size, int M,	// M pour le nombre de subdivision
						TabSeedCMRG_t *pt_cmrg, float * Stab, int * Ptab){ // Stab pour la matrice des prix du stock et Ptab pour la matrice des P( le nombre de fois sous la barriere)
		
	int idx = threadIdx.x + threadIdx.y*blockDim.x;// indice pour parcourir les trajectoires exterieures

	int a0, a1, a2, a3, a4, a5;
	float U1, U2;

	CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][blockIdx.x][threadIdx.x]);
	
	//initialisation
	Stab[idx*M] = x_0;
	Ptab[idx*M] = 0;
	
   	for (int k=1; k<M; k++){
	   CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &U1, &U2, 2);

	   BoxMuller_d(&U1, &U2);
	   
	   BS_d(Stab+ idx*M + k, Stab[idx*M + k-1], r0, sigma, dt, U1);
	  
	   Ptab[idx*M + k] = Ptab[idx*M + k-1] + int(Stab[idx*M + k]<B);//mettre � jour Ptab
	}
	
}

__global__ void inner_k(float r0, 
						float sigma, float dt, int P1, int P2,
						float K,
						float B, int size, int M,
						TabSeedCMRG_t *pt_cmrg, float * S, int * Ptab, float *V){// S la matrice des prix du stock, Ptab matrice des P et V la matrice des prix de l'option
	
	int idx_int = threadIdx.x + blockIdx.y*blockDim.x;// indice pour parcourir les trajectoires interieures
	int idx_ext = threadIdx.y + blockIdx.x*blockDim.y;// indice pour parcourir les trajectoires exterieures
	int idx = threadIdx.x + blockDim.x*threadIdx.y; // indice interm�diaire
	int blockSize = blockDim.x*blockDim.y; // taille d'un block
	int a0, a1, a2, a3, a4, a5, P;
	float U1, U2;


	//extern __shared__ float A[(100+2)*256];
	
	float A[(100+2)*256]; // M = 100

	float *Ssh, *R1sh;
	Ssh = A;
	R1sh = Ssh + 2*blockSize;
	//R2sh = R1sh + blockDim.x;

	for (int i = 0; i < M; i++){
		CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][idx_int][idx_ext]);
		Ssh[idx] = S[idx_ext*M + i]; //[idx_ext][idx_int][i]
		
		P = Ptab[idx_ext*M + i];	
		for (int k=1; k<=M-i; k++){
			CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &U1, &U2, 2);

			BoxMuller_d(&U1, &U2);
			BS_d(Ssh+idx+(k%2)*blockSize, Ssh[idx+((k+1)%2)*blockSize], r0, sigma, dt, U1);

			P += (Ssh[idx+(k%2)*blockSize]<B);
		}
		
		R1sh[idx + blockSize *i] = expf(-r0*dt*dt*M)*fmaxf(0.0f, Ssh[idx+((M-i)%2)*blockSize]-K)*((P<=P2)&&(P>=P1))/size;
		//R2sh[threadIdx.x] = R1sh[threadIdx.x]*R1sh[threadIdx.x]*size;
	}
	__syncthreads();
	
	// faire la somme des prix de l'option pour une trajectoire int�rieure donn�e le long de l'axe x du block
	int i = blockDim.x/2;

	while(i != 0){
		if(idx < i){
			for (int j = 0; j < M; j++){
				R1sh[idx + blockSize *j] += R1sh[idx + i + blockSize*j]; 
				//R2sh[threadIdx.x] += R2sh[threadIdx.x + i];
			}
			//__syncthreads();
		}
		__syncthreads();
		i /= 2;
	}
	
	// faire la somme des prix de l'option pour une trajectoire int�rieure donn�e pour les diff�rents blocks
	if(idx==0){
		
		for (int j = 0; j < M; j++){
			atomicAdd(V+ NbOuter*j + idx_ext, R1sh[blockSize*j]);

		}
	}

	
}






int main()
{

	float T = 1.0f;
	float K = 100.0f;
	float x_0 = 100.0f;
	float sigma = 0.2f;
	float r0 = 0.1f;
	float B = 120.0f;
	int M = 100;
	int P1 = 10;
	int P2 = 49;
	float dt = sqrtf(T/M);
	float Tim;							// GPU timer instructions
	cudaEvent_t start, stop;			// GPU timer instructions
	float *S, *V, *Vcpu,*Scpu;
	int *P, *Pcpu;

	//allocation de memoire
	testCUDA(cudaMalloc((void**)&S, NbOuter*M*sizeof(float)));
	testCUDA(cudaMalloc((void**)&P, NbOuter*M*sizeof(int)));
	testCUDA(cudaMalloc((void**)&V, NbOuter*M*sizeof(float)));

	testCUDA(cudaMemset(S, 0.0f, NbOuter*M*sizeof(float)));
	testCUDA(cudaMemset(P, 0, NbOuter*M*sizeof(int)));
	testCUDA(cudaMemset(V, 0.0f, NbOuter*M*sizeof(float)));
	

	PostInitDataCMRG();

	testCUDA(cudaEventCreate(&start));			// GPU timer instructions
	testCUDA(cudaEventCreate(&stop));				// GPU timer instructions
	testCUDA(cudaEventRecord(start,0));			// GPU timer instructions



	// creer les grids et blocks en 2D
	dim3 Blocks(GridInnerX, GridInnerY);
	dim3 threadsPerBlock(BlockInnerX, BlockInnerY);

	//lancer le kernel pour les trajectoires exterieures
	outer_k<<<1, 512>>>(x_0, r0,sigma, dt, P1, P2, K, B, BlockInnerX*BlockInnerY*GridInnerY, M, pt_CMRG, S, P);
	testCUDA(cudaDeviceSynchronize());
	printf("Kernel outer done\n");

	//lancer le kernel pour les trajectoires interieures
	inner_k<<<Blocks, threadsPerBlock>>>(r0, sigma, dt, P1, P2, K, B, BlockInnerX*BlockInnerY*GridInnerY, M, pt_CMRG, S, P, V);
	testCUDA(cudaDeviceSynchronize());
	printf("Kernel inner done\n");
	
	
	testCUDA(cudaEventRecord(stop,0));			// GPU timer instructions
	testCUDA(cudaEventSynchronize(stop));			// GPU timer instructions
	testCUDA(cudaEventElapsedTime(&Tim,			// GPU timer instructions
			 start, stop));				// GPU timer instructions
	testCUDA(cudaEventDestroy(start));			// GPU timer instructions
	testCUDA(cudaEventDestroy(stop));				// GPU timer instructions
	printf("Execution time %f ms\n", Tim);



	// sauvegarder les matrices de prix de stock, P et prix de l'option
	Pcpu = (int*)malloc(NbOuter*M*sizeof(int));
	testCUDA(cudaMemcpy(Pcpu, P, NbOuter*M*sizeof(int), cudaMemcpyDeviceToHost));
	testCUDA(cudaDeviceSynchronize());

	Scpu = (float*)malloc(NbOuter*M*sizeof(float));
	testCUDA(cudaMemcpy(Scpu, S, NbOuter*M*sizeof(float), cudaMemcpyDeviceToHost));
	testCUDA(cudaDeviceSynchronize());

	Vcpu = (float*)malloc(NbOuter*M*sizeof(float));
	testCUDA(cudaMemcpy(Vcpu, V, NbOuter*M*sizeof(float), cudaMemcpyDeviceToHost));
	testCUDA(cudaDeviceSynchronize());
	
 
	std::ofstream myfile;
      	myfile.open ("data.csv");
      	myfile <<"Time"<< ";"<<"S"<< ";"<<"P"<< ";"<<"Price"<<"\n";
      	
      	

    	for (int i = 0; i < 512; i++) {
        	for (int j = 0; j < M; j++) {
			myfile <<j<< ";"<<(float)Scpu[i*M+j]<< ";"<<Pcpu[i*M+j]<< ";"<<(float)Vcpu[i*M+j]<<"\n";
        	}
        	
	}
	
	myfile.close();

	
        printf("Done");



	return 0;
}

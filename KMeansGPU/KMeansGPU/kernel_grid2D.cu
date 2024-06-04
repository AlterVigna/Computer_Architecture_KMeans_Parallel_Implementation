/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda.h>


#define _CRT_SECURE_NO_WARNINGS


#include "WINDOWS.h"
#include "conio.h"


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>


#include "KMeans.h"


#include <algorithm>
#include <cfloat>
#include <chrono>
#include <random>
#include <vector>


#include <cooperative_groups.h>



typedef struct LoaderInfo {


    float** pointsMatrix;
    char* csvFile;
    char csvSplitBy;
    int startingRow;
    int endingRow;


}LoaderInfo;



// Algorithm parameters used inside the device.
__constant__ int K;
__constant__ int DIM;
__constant__ int DATASET_SIZE;


// Function prototypes


DWORD WINAPI ThreadFunctionLoadPartialDataset(LPVOID lpParam);


void initializeCentroidsOptim(int K, int DATASET_SIZE, int DIM, Point* points, Point* centroids);
void initPartialVariablesOptim(int K, int DATASET_SIZE, int DIM, Point* points, Point* centroids);
void printPointsOptim(Point* points, int size, int DIM);
void updateCentroidsOptim(int K, int DIM, Point* centroids, Point* sums, int* counts);


// Function for initialization of the centroids.
void initializeCentroidsOptim(int K, int DATASET_SIZE, int DIM, float** pointsMatrix, Point* centroids) {


    // Initialize the random number generator with current time as seed
    srand(123);


    int* number_chosen = (int*)malloc(K * sizeof(int));
    int list_size = 0;


    // Generate a random number between 0 and DATASET_SIZE
    int randomNumber = -1;


    for (int i = 0; i < K; i++) {
        do {
            randomNumber = rand() % (DATASET_SIZE);
        } while (containsNumber(number_chosen, list_size, randomNumber));
        number_chosen[list_size] = randomNumber;
        list_size++;


        for (int j = 0; j < DIM; j++) {
            centroids[i].coordinate[j] = pointsMatrix[j][randomNumber];
        }
    }
}


// Function for initialization to 0 of the temp. variables : sums and counts for updating the centroids.
void initPartialVariablesOptim(int DIM, int K, Point* sums, int* counts) {


    for (int i = 0; i < K; i++) {
        for (int j = 0; j < DIM; j++) {
            sums[i].coordinate[j] = 0;
        }
        counts[i] = 0;
    }
}


// Function to print a list of points of DIM dimension.
void printPointsOptim(Point* points, int size, int DIM) {


    for (int i = 0; i < size; i++) {


        printf("Point %zu: (", i);


        for (int j = 0; j < DIM; j++) {
            printf("%f", points[i].coordinate[j]);
            if (j != DIM - 1) {
                printf(",");
            }
        }
        printf(")\n");
    }
}


//Function to choose new representative centroids for new clusters.
void updateCentroidsOptim(int K, int DIM, Point* centroids, Point* sums, int* counts) {


    // Compute the mean for each cluster to discover new centroid point.
    for (int j = 0; j < K; j++) {


        for (int dim = 0; dim < DIM; dim++) {
            centroids[j].coordinate[dim] = sums[j].coordinate[dim] / counts[j];
        }
    }
    //printList(counts, K, "COUNTS");
    //printPoints(centroids, DIM);


    // reset distance and sum
    for (int j = 0; j < K; j++) {
        for (int dim = 0; dim < DIM; dim++) {
            sums[j].coordinate[dim] = 0;
        }
        counts[j] = 0;
    }
}



DWORD WINAPI ThreadFunctionLoadPartialDataset(LPVOID lpParam) {


    LoaderInfo* arguments = (LoaderInfo*)lpParam;
    float** pointsMatrix = arguments->pointsMatrix;
    char* csvFile = arguments->csvFile;
    int startingRow = arguments->startingRow;
    int endingRow = arguments->endingRow;
    const char csvSplitBy = arguments->csvSplitBy;


    FILE* file = fopen(csvFile, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
    }


    //initDynamicList(list, endingRow - startingRow);
    char line[MAX_LINE_LENGTH];
    int countLine = 0;
    int indexPoint = startingRow;


    while (fgets(line, sizeof(line), file)) {


        //Skip the first header line
        if (countLine < startingRow + 1) {
            countLine++;
            continue;
        }
        if (countLine == endingRow + 1)
            break;


        char* token;
        int i = 0;
        char* nextToken = NULL;


        int numTokens;
        char** tokens = splitString(line, csvSplitBy, &numTokens);


        for (int i = 0; i < numTokens; i++) {
            pointsMatrix[i][indexPoint] = atof(tokens[i]);
            free(tokens[i]);
        }
        free(tokens);


        memset(line, 0, sizeof(line));
        countLine++;
        indexPoint++;
    }


    // Close the file
    fclose(file);


    return 0;
}


__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case
        // of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}



// CUDA kernel for assign memership
__global__ void assignPointsToClusterKernel(float** pointsMatrix, Point* centroids, int* membership, Point* sums, int* counts) {
    int idx = threadIdx.x + threadIdx.y * blockDim.x + blockDim.x * blockDim.y * blockIdx.x + blockDim.x * blockDim.y * blockIdx.y * gridDim.x;


    if (idx < DATASET_SIZE) {
        //code block used for debugging the indexes

        extern __shared__ double shared_data[]; // Dynamic shared memory for centroids


        // Load centroids into shared memory for the first k and wait the others
        if (threadIdx.x < K) {
            for (int dim = 0; dim < DIM; dim++) {
                shared_data[threadIdx.x + dim * K] = centroids[threadIdx.x].coordinate[dim];
            }
        }
        // Wait for those k threads.
        __syncthreads();



        float minDistance = INFINITY;
        int minIndex = -1;
        for (int i = 0; i < K; ++i) {
            float distance = 0.0f;
            for (int j = 0; j < DIM; ++j) {
                //distance += (pointsMatrix[j][idx] - centroids[i].coordinate[j]) * (pointsMatrix[j][idx] - centroids[i].coordinate[j]);
                distance += (pointsMatrix[j][idx] - shared_data[i + j * K]) * (pointsMatrix[j][idx] - shared_data[i + j * K]);
            }
            if (distance < minDistance) {
                minDistance = distance;
                minIndex = i;
            }
        }
        membership[idx] = minIndex;



        // Update sums and counts
        //Updated for change in the indexing
        for (int dim = 0; dim < DIM; dim++) {
            atomicAdd(&sums[minIndex].coordinate[dim], pointsMatrix[dim][idx]);
        }
        atomicAdd(&counts[minIndex], 1);


    }
}


// CUDA kernel for updating centroids
__global__ void updateCentroidsKernel(Point* centroids, Point* sums, int* counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        for (int dim = 0; dim < DIM; dim++) {
            centroids[idx].coordinate[dim] = sums[idx].coordinate[dim] / counts[idx];
            sums[idx].coordinate[dim] = 0;
        }
        counts[idx] = 0;
    }
}





int main() {


    clock_t start_exe, end_exe, start_load, end_load, start_init_centr, end_init_centr, start_alg, end_alg, start_first, end_first;
    double tot_first = 0;


    // Dataset file location
    char csvFile[100];
    char csvSplitBy;


    // Algorithm parameters.


    // Number of clusters to discover
    int K_HOST;
    // Stopping condition
    int MAX_ITERATIONS;


    // Number of point dimension 2D-3D
    int DIM_HOST;


    // Number of points in the dataset
    int DATASET_SIZE_HOST;


    int NR_THREADS;


    printf("----- GPU KMeans 2D Grid  ----- \n");


    loadProperties(csvFile, &csvSplitBy, &K_HOST, &MAX_ITERATIONS, &NR_THREADS, &DATASET_SIZE_HOST, &DIM_HOST);
    printParameters(csvFile, csvSplitBy, K_HOST, MAX_ITERATIONS, NR_THREADS, DATASET_SIZE_HOST, DIM_HOST, 1);


    //************************************************************************
    //Matrix used to load the data
    float** pointsMatrix;
    //using this declaration the data structure is 
    //    x | y
    //0 | n | n
    //1 | n | n
    //...
    pointsMatrix = (float**)malloc(sizeof(float**) * DIM_HOST);
    for (int i = 0; i < DIM_HOST; i++)
        pointsMatrix[i] = (float*)malloc((sizeof(float*) * DATASET_SIZE_HOST));
    //***********************************************************************


    Point* centroids = (Point*)malloc(K_HOST * sizeof(Point));
    Point* sums = (Point*)malloc(K_HOST * sizeof(Point));



    int* counts = (int*)malloc(K_HOST * sizeof(int));
    int* membership = (int*)malloc(DATASET_SIZE_HOST * sizeof(int));


    start_exe = clock();


    start_load = clock();


    HANDLE* loadersHandles = (HANDLE*)malloc(sizeof(HANDLE) * NR_THREADS);
    DWORD* loadersThreadIds = (DWORD*)malloc(sizeof(DWORD) * NR_THREADS);
    LoaderInfo* loaderArgs = (LoaderInfo*)malloc(sizeof(LoaderInfo) * NR_THREADS);


    int STEP = DATASET_SIZE_HOST / NR_THREADS;
    int MOD = DATASET_SIZE_HOST % NR_THREADS;
    int currentDatasetRow = 0;
    for (int i = 0; i < NR_THREADS; i++) {
        loaderArgs[i].csvFile = csvFile;
        loaderArgs[i].csvSplitBy = csvSplitBy;
        loaderArgs[i].pointsMatrix = pointsMatrix;
        loaderArgs[i].startingRow = currentDatasetRow;
        currentDatasetRow += STEP;
        loaderArgs[i].endingRow = (currentDatasetRow + MOD == DATASET_SIZE_HOST) ? DATASET_SIZE_HOST : currentDatasetRow;
        loadersHandles[i] = CreateThread(NULL, 0, ThreadFunctionLoadPartialDataset, &loaderArgs[i], 0, &loadersThreadIds[i]);
    }
    WaitForMultipleObjects(NR_THREADS, loadersHandles, TRUE, INFINITE);


    for (int i = 0; i < NR_THREADS; i++) {
        CloseHandle(loadersHandles[i]);
    }


    end_load = clock();
    start_init_centr = clock();
    initializeCentroidsOptim(K_HOST, DATASET_SIZE_HOST, DIM_HOST, pointsMatrix, centroids);
    initPartialVariablesOptim(DIM_HOST, K_HOST, sums, counts);


    //function declared inside KMeans.h, used to verify the correct loading of the data
    //printMatrix(pointsMatrix, DIM, DATASET_SIZE);


    end_init_centr = clock();


    //printf("Dataset: \n");
    //printPoints(&points, DIM);


    printf("\nStarting Centroids: \n");
    printPointsOptim(centroids, K_HOST, DIM_HOST);



    // Prepare the environment for the GPU execution of the algorithm.


    // Allocate memory for points, centroids, and membership on the device
    //*********************NEW VARIABLES USED**********
    float** d_pointsMatrix;
    float** d_pointsR;
    //***********************************************
    size_t d_Pitch;
    Point* d_centroids, * d_sums;
    int* d_counts = (int*)malloc(K_HOST * sizeof(int));


    int* d_membership;


    //**********************************************************
    d_pointsR = (float**)malloc(DIM_HOST * sizeof(float*));


    cudaMalloc((void**)&d_pointsMatrix, DIM_HOST * sizeof(float*));


    // allocate arrays on the device


    for (int i = 0; i < DIM_HOST; i++)
        cudaMalloc((void**)&d_pointsR[i], DATASET_SIZE_HOST * sizeof(float));


    // copy data to the the device


    cudaMemcpy(d_pointsMatrix, d_pointsR, DIM_HOST * sizeof(float*), cudaMemcpyHostToDevice);


    for (int i = 0; i < DIM_HOST; i++)
        cudaMemcpy(d_pointsR[i], pointsMatrix[i], sizeof(float) * DATASET_SIZE_HOST, cudaMemcpyHostToDevice);


    //****************************************************************
    cudaMalloc((void**)&d_centroids, K_HOST * sizeof(Point));
    cudaMalloc((void**)&d_membership, DATASET_SIZE_HOST * sizeof(int));


    cudaMalloc((void**)&d_sums, K_HOST * sizeof(Point));
    cudaMalloc((void**)&d_counts, K_HOST * sizeof(int));



    // Set constant values
    cudaMemcpyToSymbol(K, &K_HOST, sizeof(int));
    cudaMemcpyToSymbol(DIM, &DIM_HOST, sizeof(int));
    cudaMemcpyToSymbol(DATASET_SIZE, &DATASET_SIZE_HOST, sizeof(int));



    cudaMemcpy(d_centroids, centroids, K_HOST * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_membership, membership, DATASET_SIZE_HOST * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sums, sums, K_HOST * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, counts, K_HOST * sizeof(Point), cudaMemcpyHostToDevice);



   //****************************************************************************************************
  //Davide's architecture support maximum size of blocks of 1024
  //1024*10=10240 threads per block we must exceed maximum number of thread per multiprocessor, so
  //it will have enough warps to switch
   //with this value it's possible to modify blocks size
    int dimBlockX = 32;
    int dimBlockY = 32;
    // this value are calculated in order to use the minimum number of threads to cover the entire dataset
    // and to use matrix of N X N
    dim3 blockSize(dimBlockX, dimBlockY);
    int dimGrid = ceil(sqrt(((DATASET_SIZE_HOST - (dimBlockX * dimBlockY) - 1)) / (dimBlockX * dimBlockY)));
    printf("%d\n", dimGrid);
    int dimGridX = dimGrid;
    int dimGridY = dimGrid;
    dim3 gridSize(dimGridX, dimGridY);
    printf("GRID used %d X %d WITH THREAD BLOCKS %d X %d\n", dimGridX, dimGridY, dimBlockX, dimBlockY);
    //******************************************************************************************************
    printf("\nStart of the algorithm :\n");


    start_alg = clock();


    // Main loop of KMeans algorithm
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {



        // Launch kernel to assign points to clusters
        assignPointsToClusterKernel << < gridSize, blockSize, 2 * K_HOST * sizeof(double) >> > (d_pointsMatrix, d_centroids, d_membership, d_sums, d_counts);
        cudaDeviceSynchronize();
        //end_first = clock();
        //tot_first += end_first - start_first;


        // Launch kernel to update centroids
        updateCentroidsKernel << <1, K_HOST >> > (d_centroids, d_sums, d_counts);
        cudaDeviceSynchronize();
    }


    end_alg = clock();
    //printf("Total first time: %f \n", tot_first);
    printf("End of the algorithm: \n");


    // Copy centroids and membership from device to host
    cudaMemcpy(centroids, d_centroids, K_HOST * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(membership, d_membership, DATASET_SIZE_HOST * sizeof(int), cudaMemcpyDeviceToHost);


    printf("Last centroids: \n");
    printPointsOptim(centroids, K_HOST, DIM_HOST);


    end_exe = clock();


    printTimeStatistics((double)(end_exe - start_exe) / CLOCKS_PER_SEC, (double)(end_load - start_load) / CLOCKS_PER_SEC,
        (double)(end_init_centr - start_init_centr) / CLOCKS_PER_SEC, (double)(end_alg - start_alg) / CLOCKS_PER_SEC);


    printf("Press Enter to continue...\n");
    getchar(); // Wait for user to press Enter


    // Free device memory
    for (int i = 0; i < DIM_HOST; i++)
        cudaFree(d_pointsR[i]);
    cudaFree(d_pointsMatrix);
    cudaFree(d_centroids);
    cudaFree(d_membership);
    cudaFree(d_counts);
    cudaFree(d_sums);



    // Free host memory
    for (int i = 0; i < DIM_HOST; i++)
        free(pointsMatrix[i]);
    free(pointsMatrix);
    free(centroids);
    free(membership);
    free(counts);
    free(sums);





    return 0;
}*/
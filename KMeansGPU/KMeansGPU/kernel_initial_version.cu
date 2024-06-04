/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

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

    Point* points;
    char* csvFile;
    char csvSplitBy;
    int startingRow;
    int endingRow;

}LoaderInfo;


// Function prototypes

DWORD WINAPI ThreadFunctionLoadPartialDataset(LPVOID lpParam);

void initializeCentroidsOptim(int K, int DATASET_SIZE, int DIM, Point* points, Point* centroids);
void initPartialVariablesOptim(int K, int DATASET_SIZE, int DIM, Point* points, Point* centroids);
void printPointsOptim(Point* points, int size, int DIM);

// Function for initialization of the centroids.
void initializeCentroidsOptim(int K, int DATASET_SIZE, int DIM, Point* points, Point* centroids) {

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
            centroids[i].coordinate[j] = points[randomNumber].coordinate[j];
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

DWORD WINAPI ThreadFunctionLoadPartialDataset(LPVOID lpParam) {

    LoaderInfo* arguments = (LoaderInfo*)lpParam;
    Point* list = arguments->points;
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
            list[indexPoint].coordinate[i] = atof(tokens[i]);
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


// NOT USED - Handled at hardware level using -arch=sm_61 during compilation.
//__device__ double atomicAddDouble(double* address, double val)
//{
//    unsigned long long int* address_as_ull =
//        (unsigned long long int*) address;
//    unsigned long long int old = *address_as_ull;
//    unsigned long long int assumed;
//    do {
//        assumed = old;
//        old = atomicCAS(address_as_ull, assumed,
//            __double_as_longlong(val + __longlong_as_double(assumed)));
//        // Note: uses integer comparison to avoid hang in case
        // of NaN (since NaN != NaN)
//    } while (assumed != old);
//    return __longlong_as_double(old);
//}


__global__ void assignPointsToClusterKernel_v0(Point* points, Point* centroids, int* membership, Point* sums, int* counts, int DATASET_SIZE, int DIM, int K,int SIZE) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int indexPartial = idx * SIZE; indexPartial < ((idx + 1) * SIZE); indexPartial++) {
        if (indexPartial < DATASET_SIZE) {

            float minDistance = INFINITY;
            int minIndex = -1;
            for (int i = 0; i < K; ++i) {
                float distance = 0.0f;
                for (int j = 0; j < DIM; ++j) {
                    distance += (points[indexPartial].coordinate[j] - centroids[i].coordinate[j]) * (points[indexPartial].coordinate[j] - centroids[i].coordinate[j]);
                }
                if (distance < minDistance) {
                    minDistance = distance;
                    minIndex = i;
                }
            }
            membership[indexPartial] = minIndex;


            // Update sums and counts
            for (int dim = 0; dim < DIM; dim++) {
                atomicAdd(&sums[minIndex].coordinate[dim], points[indexPartial].coordinate[dim]);
            }

            atomicAdd(&counts[minIndex], 1);
        }
    }
}





// CUDA kernel for updating centroids
__global__ void updateCentroidsKernel(Point* centroids, Point* sums, int* counts, int K, int DIM) {
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

    // Stopping condition
    int MAX_ITERATIONS;

    int NR_THREADS;

    // Number of clusters to discover
    int K;

    // Number of point dimension 2D-3D
    int DIM;

    // Number of points in the dataset
    int DATASET_SIZE;


    printf("----- GPU KMeans  ----- \n");

    loadProperties(csvFile, &csvSplitBy, &K, &MAX_ITERATIONS, &NR_THREADS, &DATASET_SIZE, &DIM);
    printParameters(csvFile, csvSplitBy, K, MAX_ITERATIONS, NR_THREADS, DATASET_SIZE, DIM, 1);

    Point* points = (Point*)malloc(DATASET_SIZE * sizeof(Point));
    Point* centroids = (Point*)malloc(K * sizeof(Point));
    Point* sums = (Point*)malloc(K * sizeof(Point));


    int* counts = (int*)malloc(K * sizeof(int));
    int* membership = (int*)malloc(DATASET_SIZE * sizeof(int));

    start_exe = clock();

    start_load = clock();

    HANDLE* loadersHandles = (HANDLE*)malloc(sizeof(HANDLE) * NR_THREADS);
    DWORD* loadersThreadIds = (DWORD*)malloc(sizeof(DWORD) * NR_THREADS);
    LoaderInfo* loaderArgs = (LoaderInfo*)malloc(sizeof(LoaderInfo) * NR_THREADS);

    int STEP = DATASET_SIZE / NR_THREADS;
    int MOD = DATASET_SIZE % NR_THREADS;
    int currentDatasetRow = 0;
    for (int i = 0; i < NR_THREADS; i++) {
        loaderArgs[i].csvFile = csvFile;
        loaderArgs[i].csvSplitBy = csvSplitBy;
        loaderArgs[i].points = points;
        loaderArgs[i].startingRow = currentDatasetRow;
        currentDatasetRow += STEP;
        loaderArgs[i].endingRow = (currentDatasetRow + MOD == DATASET_SIZE) ? DATASET_SIZE : currentDatasetRow;
        loadersHandles[i] = CreateThread(NULL, 0, ThreadFunctionLoadPartialDataset, &loaderArgs[i], 0, &loadersThreadIds[i]);
    }
    WaitForMultipleObjects(NR_THREADS, loadersHandles, TRUE, INFINITE);

    for (int i = 0; i < NR_THREADS; i++) {
        CloseHandle(loadersHandles[i]);
    }

    end_load = clock();
    start_init_centr = clock();
    initializeCentroidsOptim(K, DATASET_SIZE, DIM, points, centroids);
    initPartialVariablesOptim(DIM, K, sums, counts);

    end_init_centr = clock();

    //printf("Dataset: \n");
    //printPoints(&points, DIM);

    printf("\nStarting Centroids: \n");
    printPointsOptim(centroids, K, DIM);


    // Prepare the environment for the GPU execution of the algorithm.

    // Allocate memory for points, centroids, and membership on the device
    Point* d_points, * d_centroids, * d_sums;

    int* d_counts;
    int* d_membership;


    start_first = clock();

    cudaMalloc((void**)&d_points, DATASET_SIZE * sizeof(Point));
    cudaMalloc((void**)&d_centroids, K * sizeof(Point));
    cudaMalloc((void**)&d_membership, DATASET_SIZE * sizeof(int));


    cudaMalloc((void**)&d_sums, K * sizeof(Point));
    cudaMalloc((void**)&d_counts, K * sizeof(int));

    // Copy points from host to device
    cudaMemcpy(d_points, points, DATASET_SIZE * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, K * sizeof(Point), cudaMemcpyHostToDevice);

    cudaMemcpy(d_sums, sums, K * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, counts, K * sizeof(Point), cudaMemcpyHostToDevice);


    const int threads = 1024;
    const int blocks = (DATASET_SIZE + threads - 1) / threads;

    const int SIZE = (DATASET_SIZE + (blocks * threads) - 1) / (blocks * threads);



    printf("\nStart of the algorithm :");

    start_alg = clock();

    // Main loop of KMeans algorithm
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {

        // Launch kernel to assign points to clusters

        assignPointsToClusterKernel_v0 << < blocks, threads>>> (d_points, d_centroids, d_membership, d_sums, d_counts,DATASET_SIZE,DIM,K,SIZE);
        cudaDeviceSynchronize();

        // Launch kernel to update centroids
        updateCentroidsKernel << <1, K >> > (d_centroids, d_sums, d_counts,K,DIM);
        cudaDeviceSynchronize();
    }

    end_alg = clock();

    printf("End of the algorithm: \n");

    // Copy centroids and membership from device to host
    cudaMemcpy(centroids, d_centroids, K * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(membership, d_membership, DATASET_SIZE * sizeof(int), cudaMemcpyDeviceToHost);



    printf("Last centroids: \n");
    printPointsOptim(centroids, K, DIM);

    end_exe = clock();

    end_first = clock();

    printTimeStatistics((double)(end_exe - start_exe) / CLOCKS_PER_SEC, (double)(end_load - start_load) / CLOCKS_PER_SEC,
        (double)(end_init_centr - start_init_centr) / CLOCKS_PER_SEC, (double)(end_alg - start_alg) / CLOCKS_PER_SEC);

    printf("TOT_CUDA_EXE: %f \n", (double)(end_first - start_first) / CLOCKS_PER_SEC);

    printf("Press Enter to continue...\n");
    //getchar(); // Wait for user to press Enter


    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_membership);

    cudaFree(d_counts);
    cudaFree(d_sums);


    // Free host memory
    free(points);
    free(centroids);
    free(membership);
    free(counts);
    free(sums);


    return 0;
}*/
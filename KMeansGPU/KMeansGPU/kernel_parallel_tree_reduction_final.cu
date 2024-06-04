#include "cuda_runtime.h"
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

// Algorithm parameters used inside the device.
__constant__ int K;
__constant__ int DIM;
__constant__ int DATASET_SIZE;


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



__global__ void assignPointsToClusterKernel(Point* points, Point* centroids, int* membership, Point* sums, int* counts, int* d_counter) {

    // Essentially three dimensional: n * x, n * y, n * counts.
    extern __shared__ double shared_data[];

    const int local_index = threadIdx.x;

    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_index >= DATASET_SIZE) return;

    // Load centroids into shared memory
    if (threadIdx.x < K) {
        for (int dim = 0; dim < DIM; dim++) {
            shared_data[threadIdx.x + dim * K] = centroids[threadIdx.x].coordinate[dim];
        }
    }
    // Wait for those k threads.
    __syncthreads();


    int minIndex = -1;
    double minDistance = INFINITY;
    for (int i = 0; i < K; ++i) {
        double distance = 0.0f;

        for (int j = 0; j < DIM; ++j) {
            distance += (points[global_index].coordinate[j] - shared_data[i + j * K]) * (points[global_index].coordinate[j] - shared_data[i + j * K]);
        }
        if (distance < minDistance) {
            minDistance = distance;
            minIndex = i;
        }
    }

    membership[global_index] = minIndex;
    __syncthreads();


    // Reduction step.
    const int count = local_index + blockDim.x + blockDim.x;

    for (int cluster = 0; cluster < K; cluster++) {
        // Zeros if this point (thread) is not assigned to the cluster, else the values of the point.
        for (int dim = 0; dim < DIM; dim++) {
            shared_data[local_index + dim * blockDim.x] = (minIndex == cluster) ? points[global_index].coordinate[dim] : 0;
        }
        shared_data[count] = (minIndex == cluster) ? 1 : 0;
        __syncthreads();

        // Tree-reduction for this cluster.
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (local_index < stride) {
                for (int dim = 0; dim < DIM; dim++) {
                    shared_data[local_index + dim * blockDim.x] += shared_data[local_index + dim * blockDim.x + stride];
                    shared_data[local_index + dim * blockDim.x + stride] = 0;
                }
                shared_data[count] += shared_data[count + stride];
                shared_data[count + stride] = 0;
            }
            __syncthreads();
        }


        // Now shared_data[0] holds the sum for x.
        if (local_index == 0) {
            //atomicAdd(d_counter, 1);
            //const int cluster_index = blockIdx.x * K + cluster;
            for (int dim = 0; dim < DIM; dim++) {
                atomicAdd(&sums[cluster].coordinate[dim], shared_data[local_index + dim * blockDim.x]);
            }
            atomicAdd(&counts[cluster], shared_data[count]);
        }
        __syncthreads();
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

    // Stopping condition
    int MAX_ITERATIONS;

    int NR_THREADS;

    // Number of clusters to discover
    int K_HOST;

    // Number of point dimension 2D-3D
    int DIM_HOST;

    // Number of points in the dataset
    int DATASET_SIZE_HOST;


    printf("----- GPU KMeans  ----- \n");

    loadProperties(csvFile, &csvSplitBy, &K_HOST, &MAX_ITERATIONS, &NR_THREADS, &DATASET_SIZE_HOST, &DIM_HOST);
    printParameters(csvFile, csvSplitBy, K_HOST, MAX_ITERATIONS, NR_THREADS, DATASET_SIZE_HOST, DIM_HOST, 1);


    Point* points = (Point*)malloc(DATASET_SIZE_HOST * sizeof(Point));
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
        loaderArgs[i].points = points;
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
    initializeCentroidsOptim(K_HOST, DATASET_SIZE_HOST, DIM_HOST, points, centroids);
    initPartialVariablesOptim(DIM_HOST, K_HOST, sums, counts);

    end_init_centr = clock();

    //printf("Dataset: \n");
    //printPoints(&points, DIM);

    printf("\nStarting Centroids: \n");
    printPointsOptim(centroids, K_HOST, DIM_HOST);



    /// DOPO





    // Prepare the environment for the GPU execution of the algorithm.

    // Allocate memory for points, centroids, and membership on the device
    Point* d_points, * d_centroids, * d_sums;

    int* d_counts;
    int* d_membership;


    cudaMalloc((void**)&d_points, DATASET_SIZE_HOST * sizeof(Point));
    cudaMalloc((void**)&d_centroids, K_HOST * sizeof(Point));
    cudaMalloc((void**)&d_membership, DATASET_SIZE_HOST * sizeof(int));

    cudaMalloc((void**)&d_sums, K_HOST * sizeof(Point));
    cudaMalloc((void**)&d_counts, K_HOST * sizeof(int));

    // Copy points from host to device
    cudaMemcpy(d_points, points, DATASET_SIZE_HOST * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, K_HOST * sizeof(Point), cudaMemcpyHostToDevice);

    cudaMemcpy(d_sums, sums, K_HOST * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, counts, K_HOST * sizeof(Point), cudaMemcpyHostToDevice);

    // Set constant values
    cudaMemcpyToSymbol(K, &K_HOST, sizeof(int));
    cudaMemcpyToSymbol(DIM, &DIM_HOST, sizeof(int));
    cudaMemcpyToSymbol(DATASET_SIZE, &DATASET_SIZE_HOST, sizeof(int));


    printf("\nStart of the algorithm :");



    const int threads = 1024;
    const int blocks = (DATASET_SIZE_HOST + threads - 1) / threads;
    printf("BLOCKS: %d \n", blocks);

    const int fine_shared_memory = 3 * threads * sizeof(double); // 24.576


    int h_counter = 0;

    // Device counter, used just for counting the new atomic operatons.
    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    // Initialize the device counter to 0
    cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice);

    start_alg = clock();
    // Main loop of KMeans algorithm
    for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {

        // Launch kernel to assign points to clusters
        assignPointsToClusterKernel << <blocks, threads, fine_shared_memory >> > (d_points, d_centroids, d_membership, d_sums, d_counts, d_counter);
        cudaDeviceSynchronize();


        // Launch kernel to update centroids
        updateCentroidsKernel << <1, K_HOST >> > (d_centroids, d_sums, d_counts);
        cudaDeviceSynchronize();

    }

    end_alg = clock();
    printf("End of the algorithm: \n");

    // Copy centroids and membership from device to host
    cudaMemcpy(centroids, d_centroids, K_HOST * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(membership, d_membership, DATASET_SIZE_HOST * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Last centroids: \n");
    // Copy the counter back to host
    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printPointsOptim(centroids, K_HOST, DIM_HOST);


    // Print the result
    printf("Counter value: %d\n", h_counter);

    // Free device memory
    cudaFree(d_counter);

    end_exe = clock();

    printTimeStatistics((double)(end_exe - start_exe) / CLOCKS_PER_SEC, (double)(end_load - start_load) / CLOCKS_PER_SEC,
        (double)(end_init_centr - start_init_centr) / CLOCKS_PER_SEC, (double)(end_alg - start_alg) / CLOCKS_PER_SEC);

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
}
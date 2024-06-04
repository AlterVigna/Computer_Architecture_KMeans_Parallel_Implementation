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



HANDLE *suspendEvents;

typedef struct ThreadInfo {

    int threadId;
    DynamicList* points;
    DynamicList* centroids;
    DynamicList* sums;

    int* counts;
    int* membership;

    int K;
    int DIM;

    int startIndex;
    int endIndex;

}ThreadInfo;

typedef struct LoaderInfo {

    DynamicList* points;
    char* csvFile;
    char csvSplitBy;
    int startingRow;
    int endingRow;

}LoaderInfo;


// Function prototypes

DWORD WINAPI ThreadFunctionAssignPointsToCluster(LPVOID lpParam);
DWORD WINAPI ThreadFunctionLoadPartialDataset(LPVOID lpParam);


DWORD WINAPI ThreadFunctionLoadPartialDataset(LPVOID lpParam) {

    LoaderInfo* arguments = (LoaderInfo*)lpParam;
    DynamicList* list = arguments->points;
    char* csvFile = arguments->csvFile;
    int startingRow = arguments->startingRow;
    int endingRow = arguments->endingRow;
    const char csvSplitBy = arguments->csvSplitBy;

    FILE* file = fopen(csvFile, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
    }

    initDynamicList(list, endingRow - startingRow);
    char line[MAX_LINE_LENGTH];
    int countLine = 0;

    while (fgets(line, sizeof(line), file)) {

        //Skip the first header line
        if (countLine < startingRow + 1) {
            countLine++;
            continue;
        }
        if (countLine == endingRow + 1)
            break;

        char* token;
        Point point;
        int i = 0;
        char* nextToken = NULL;

        int numTokens;
        char** tokens = splitString(line, csvSplitBy, &numTokens);

        for (int i = 0; i < numTokens; i++) {
            point.coordinate[i] = atof(tokens[i]);
            free(tokens[i]);
        }
        free(tokens);

        //modify function
        appendToDynamicList(list, point);

        memset(line, 0, sizeof(line));
        countLine++;
    }

    // Close the file
    fclose(file);

    return 0;
}



// Function to assign each point of the dataset to a cluster membership and collect the new clusters informations (sums and counts).
DWORD WINAPI ThreadFunctionAssignPointsToCluster(LPVOID lpParam) {

   
    while (1) {

        ThreadInfo* arguments = (ThreadInfo*)lpParam;

        arguments->sums = (DynamicList*)malloc(sizeof(DynamicList));
        initDynamicList(arguments->sums, arguments->K);

        arguments->counts = (int*)malloc(arguments->K * sizeof(int));

        initPartialVariables(arguments->DIM, arguments->K, arguments->sums, arguments->counts);

        //printf("Thread %d running on CPU %d\n", arguments->threadId, GetCurrentProcessorNumber());

        float* dists = (float*)malloc(arguments->K * sizeof(float));
        for (int indexOfPoint = arguments->startIndex; indexOfPoint < arguments->endIndex; indexOfPoint++) {
            Point* point = accessByIndex(arguments->points, indexOfPoint);

            for (int clusterIndex = 0; clusterIndex < arguments->K; clusterIndex++) {

                float sumPartial = 0;
                for (int dim = 0; dim < arguments->DIM; dim++) {
                    Point* centroidPoint = accessByIndex(arguments->centroids, clusterIndex);
                    sumPartial += pow(centroidPoint->coordinate[dim] - point->coordinate[dim], 2);
                }
                // Compute distance from a point to all centroids
                dists[clusterIndex] = sqrt(sumPartial);
            }

            // Find the minimum distance
            float min = dists[0];

            int minIndex = 0;
            for (int z = 1; z < arguments->K; z++) {

                float currentValue = dists[z];
                if (currentValue < min) {
                    min = currentValue;
                    minIndex = z;
                }
            }

            arguments->membership[indexOfPoint] = minIndex;

            // Save information of the points that belongs to the new cluster in order to update it leater.
            for (int dim = 0; dim < arguments->DIM; dim++) {
                Point* sumPoint = accessByIndex(arguments->sums, minIndex);
                sumPoint->coordinate[dim] += point->coordinate[dim];
            }
            arguments->counts[minIndex] += 1;
        }
        SetEvent(suspendEvents[arguments->threadId]);
        HANDLE hThread = GetCurrentThread();
        SuspendThread(hThread);
    }
    return 0;
}




int main(void) {

    clock_t start_exe, end_exe, start_load, end_load, start_init_centr, end_init_centr, start_alg, end_alg;
    
    // Dataset file location
    char csvFile[100];
    char csvSplitBy;

    // Algorithm parameters.

    // Number of clusters to discover
    int K;
    // Stopping condition
    int MAX_ITERATIONS;

    // Number of point dimension 2D-3D
    int DIM;

    // Number of points in the dataset
    int DATASET_SIZE;

    int NR_THREADS;

    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);

    printf("----- Parallel KMeans ----- \n");

    loadProperties(csvFile, &csvSplitBy, &K, &MAX_ITERATIONS, &NR_THREADS, &DATASET_SIZE, &DIM);
    printParameters(csvFile, csvSplitBy, K, MAX_ITERATIONS, NR_THREADS, DATASET_SIZE, DIM, 1);

    DynamicList points;
    DynamicList* pointsSubset = (DynamicList*)malloc(sizeof(DynamicList) * NR_THREADS);
    DynamicList centroids;
    int* membership = (int*)malloc(DATASET_SIZE * sizeof(int));

    // Temp. variables
    DynamicList sums;
    int* counts = (int*)malloc(K * sizeof(int));

    // Initialize the dynamic lists
    initDynamicList(&points, DATASET_SIZE);
    initDynamicList(&centroids, K);
    initDynamicList(&sums, K);

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
        loaderArgs[i].points = &pointsSubset[i];
        loaderArgs[i].startingRow = currentDatasetRow;
        currentDatasetRow += STEP;
        loaderArgs[i].endingRow = (currentDatasetRow + MOD == DATASET_SIZE) ? DATASET_SIZE : currentDatasetRow;
        loadersHandles[i] = CreateThread(NULL, 0, ThreadFunctionLoadPartialDataset, &loaderArgs[i], 0, &loadersThreadIds[i]);
    }
    WaitForMultipleObjects(NR_THREADS, loadersHandles, TRUE, INFINITE);

    for (int i = 0; i < NR_THREADS; i++) {
        CloseHandle(loadersHandles[i]);
    }
    //link of the lists
    for (int i = 0; i < NR_THREADS; i++) {
        for (int y = 0; y < (loaderArgs[i].endingRow - loaderArgs[i].startingRow); y++) {
            appendToDynamicList(&points, pointsSubset[i].array[y]);
        }
        freeDynamicList(&pointsSubset[i]);
    }
    end_load = clock();

    start_init_centr = clock();
    initializeCentroids(K, DATASET_SIZE, DIM, &points, &centroids);
    initPartialVariables(DIM, K, &sums, counts);
    end_init_centr = clock();

    //printf("Dataset: \n");
    //printPoints(&points, DIM);

    printf("\nStarting Centroids: \n");
    printPoints(&centroids, DIM);

  
    // Prepare the environment for the parallel execution of the algorithm.

    // Common variable to store when a thread finishes his job.
    suspendEvents= (HANDLE*)malloc(sizeof(HANDLE) * NR_THREADS);;

    HANDLE* handles = (HANDLE*)malloc(sizeof(HANDLE) * NR_THREADS);
    DWORD* threadIds = (DWORD*)malloc(sizeof(DWORD) * NR_THREADS);
    ThreadInfo* results = (ThreadInfo*)malloc(sizeof(ThreadInfo) * NR_THREADS);

    STEP = DATASET_SIZE / NR_THREADS;
    int currentDatasetIndex = 0;

    for (int i = 0; i < NR_THREADS; i++) {

        results[i].threadId = i;
        threadIds[i] = i;

        int endSplit = ((currentDatasetIndex + 2 * STEP <= DATASET_SIZE)) ? (currentDatasetIndex + STEP) : DATASET_SIZE;

        results[i].points = &points;
        results[i].membership = membership;
        results[i].centroids = &centroids;

        results[i].startIndex = currentDatasetIndex;
        results[i].endIndex = endSplit;

        results[i].K = K;
        results[i].DIM = DIM;
        
        suspendEvents[i] = CreateEvent(NULL, TRUE, FALSE, NULL);
        handles[i] = CreateThread(NULL, 0, ThreadFunctionAssignPointsToCluster, &results[i], CREATE_SUSPENDED, &threadIds[i]);
        currentDatasetIndex = endSplit;
    }

    //printf("Thread ORIG running on CPU %d\n", GetCurrentProcessorNumber());

    int numMaxCores = sysInfo.wProcessorLevel;      //  6 in our case
    int numMaxThreads = sysInfo.dwNumberOfProcessors; // 12 in our case

    /*if (NR_THREADS <= numMaxThreads) {
        // Set CPU affinity for each thread
        for (int i = 0; i < NR_THREADS; i++) {
            
            // Assign each thread to a different CPU logical core

            int shift = ((i * 2) < numMaxThreads) ? (i * 2) : ((i % numMaxCores) * 2 + 1);

            DWORD_PTR mask = 1 << shift;
            if (!SetThreadAffinityMask(handles[i], mask)) {
                fprintf(stderr, "Error setting affinity for thread %d\n", i);
                return 1;
            }
        }
    }*/
    


    printf("\nStart of the algorithm :");
    
    start_alg = clock();

    for (int nrIteration = 0; nrIteration < MAX_ITERATIONS; nrIteration++) {

        // Wake up all the sospended threads.
        for (int i = 0; i < NR_THREADS; i++) {
            ResumeThread(handles[i]);
        }

        // Wait until all the threads has produced their results and then come to sleep.
        WaitForMultipleObjects(NR_THREADS, suspendEvents, TRUE, INFINITE);

        // Aggregate partial results
        for (int j = 0; j < K; j++) {
            Point* totSum = accessByIndex(&sums, j);
            for (int i = 0; i < NR_THREADS; i++) {

                Point* partialSum = accessByIndex(results[i].sums, j);

                for (int dim = 0; dim < DIM; dim++) {
                    totSum->coordinate[dim] += partialSum->coordinate[dim];
                }
                counts[j] += results[i].counts[j];
            }
        }
        updateCentroids(K, DIM, &centroids, &sums, counts);

        // Reset the variables to track sospension next iterations.
        for (int i = 0; i < NR_THREADS; i++) {
            ResetEvent(suspendEvents[i]);
        }

    }
    end_alg = clock();

    printf("End of the algorithm: \n");

    for (int i = 0; i < NR_THREADS; i++) {
        CloseHandle(handles[i]);
    }

    printf("Last centroids: \n");
    printPoints(&centroids, DIM);

    end_exe = clock();

    printTimeStatistics((double)(end_exe - start_exe) / CLOCKS_PER_SEC, (double)(end_load - start_load) / CLOCKS_PER_SEC,
        (double)(end_init_centr - start_init_centr) / CLOCKS_PER_SEC, (double)(end_alg - start_alg) / CLOCKS_PER_SEC);

    printf("Press Enter to continue...\n");
    //getchar(); // Wait for user to press Enter

    return 0;
}
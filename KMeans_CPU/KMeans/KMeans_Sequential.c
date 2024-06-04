/*#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include "KMeans.h"



// Function prototypes
void loadData(char csvFile[100], const char csvSplitBy, DynamicList* list, int DIM);
void assignPointsToCluster(int K, int DIM, int DATASET_SIZE, DynamicList* points, DynamicList* centroids, DynamicList* sums, int* counts, int* membership);


// Function to load data from file
void loadData(char csvFile[100], const char csvSplitBy, DynamicList* list, int DIM) {

    FILE* file = fopen(csvFile, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
    }

    char line[MAX_LINE_LENGTH];
    int countLine = 0;

    while (fgets(line, sizeof(line), file)) {

        //Skip the first header line
        if (countLine == 0) {
            countLine++;
            continue;
        }

        char* token;
        Point point;

        int numTokens;
        char** tokens = splitString(line, csvSplitBy, &numTokens);

        for (int i = 0; i < numTokens; i++) {
            point.coordinate[i] = atof(tokens[i]);
            free(tokens[i]);
        }
        free(tokens);

        appendToDynamicList(list, point);

        memset(line, 0, sizeof(line));
    }

    // Close the file
    fclose(file);
}




// Function to assign each point of the dataset to a cluster membership and collect the new clusters informations (sums and counts).
void assignPointsToCluster(int K, int DIM, int DATASET_SIZE, DynamicList* points, DynamicList* centroids, DynamicList* sums, int* counts, int* membership) {

    float* dists = (float*)malloc(K * sizeof(float));

    for (int indexOfPoint = 0; indexOfPoint < DATASET_SIZE; indexOfPoint++) {

        Point* point = accessByIndex(points, indexOfPoint);
        for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {

            float sumPartial = 0;
            for (int dim = 0; dim < DIM; dim++) {
                Point* centroidPoint = accessByIndex(centroids, clusterIndex);
                sumPartial += pow(centroidPoint->coordinate[dim] - point->coordinate[dim], 2);
            }
            // Compute distance from a point to all centroids
            dists[clusterIndex] = sqrt(sumPartial);
        }

        // Find the minimum distance
        float min = dists[0];

        int minIndex = 0;
        for (int z = 1; z < K; z++) {

            float currentValue = dists[z];
            if (currentValue < min) {
                min = currentValue;
                minIndex = z;
            }
        }

        membership[indexOfPoint] = minIndex;

        // Save information of the points that belongs to the new cluster in order to update it leater.
        for (int dim = 0; dim < DIM; dim++) {
            Point* sumPoint = accessByIndex(sums, minIndex);
            sumPoint->coordinate[dim] += point->coordinate[dim];
        }

        counts[minIndex] += 1;
    }
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

    printf("----- Sequential KMeans ----- \n");

    loadProperties(csvFile, &csvSplitBy, &K, &MAX_ITERATIONS, &NR_THREADS, &DATASET_SIZE, &DIM);
    printParameters(csvFile, csvSplitBy, K, MAX_ITERATIONS, NR_THREADS, DATASET_SIZE, DIM, 0);

    DynamicList points;
    DynamicList centroids;
    int* membership = (int*)malloc(DATASET_SIZE * sizeof(int));

    // Temp. variables
    DynamicList sums;
    int* counts = (int*)malloc(K * sizeof(int));

    // Initialize the dynamic lists
    initDynamicList(&points, 1);
    initDynamicList(&centroids, K);
    initDynamicList(&sums, K);

    start_exe = clock();

    start_load = clock();

    loadData(csvFile, csvSplitBy, &points, DIM);

    end_load = clock();

    start_init_centr = clock();
    initializeCentroids(K, DATASET_SIZE, DIM, &points, &centroids);
    initPartialVariables(DIM, K, &sums, counts);

    end_init_centr = clock();

    //printf("Dataset: \n");
    //printPoints(&points, DIM);

    printf("\nStarting Centroids: \n");
    printPoints(&centroids, DIM);

    //printList(membership, DATASET_SIZE,"Membership");

    printf("\nStart of the algorithm :");
    start_alg = clock();
    for (int nrIteration = 0; nrIteration < MAX_ITERATIONS; nrIteration++) {
        assignPointsToCluster(K, DIM, DATASET_SIZE, &points, &centroids, &sums, counts, membership);
        //printList(membership, DATASET_SIZE, "Membership");
        //printList(counts, K, "Counts");
        updateCentroids(K, DIM, &centroids, &sums, counts);
    }
    end_alg = clock();

    printf("End of the algorithm: \n");





    printf("Last centroids: \n");
    printPoints(&centroids, DIM);

    end_exe = clock();

    printTimeStatistics((double)(end_exe - start_exe) / CLOCKS_PER_SEC, (double)(end_load - start_load) / CLOCKS_PER_SEC,
        (double)(end_init_centr - start_init_centr) / CLOCKS_PER_SEC, (double)(end_alg - start_alg) / CLOCKS_PER_SEC);

    printf("Press Enter to continue...\n");
    //getchar(); // Wait for user to press Enter

    return 0;

}*/
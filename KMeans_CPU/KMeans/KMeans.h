#include <stdio.h>

#define _CRT_SECURE_NO_WARNINGS
#define MAX_LINE_LENGTH 100

// DATA STRUCTURES USED IN THE ALGORITHM

// Define the structure for a Point
typedef struct {
    double coordinate[3];
} Point;

// Define the structure for the dynamic list
typedef struct {
    Point* array;
    size_t capacity;
    size_t size;
} DynamicList;



// METHODS TO HANDLE DynamicList TYPE
void initDynamicList(DynamicList* list, size_t initial_capacity);
void appendToDynamicList(DynamicList* list, Point point);
Point* accessByIndex(DynamicList* list, size_t index);
void freeDynamicList(DynamicList* list);

// METHODS TO READ config.properties FILE
int loadProperties(char *csvFile, char* csvSplitBy, int* K, int* MAX_ITERATIONS, int* NR_THREADS, int* DATASET_SIZE, int* DIM);
void readDatasetSizeAndNumberOfDimensions(char csvFile[100], char csvSplitBy, int* DATASET_SIZE, int* DIM);

// METHODS SPECIFIC OF KMEANS
void initPartialVariables(int DIM, int K, DynamicList* sums, int* counts);
void initializeCentroids(int K, int DATASET_SIZE, int DIM, DynamicList* points, DynamicList* centroids);
void updateCentroids(int K, int DIM, DynamicList* centroids, DynamicList* sums, int* counts);


// ------------------------------- METHODS OF VARIUOS UTILITY
int containsNumber(int* list, int size, int number);


// METHODS FOR PRINTING OUTPUT 
void printParameters(char csvFile[100], char csvSplitBy, int K, int MAX_ITERATIONS, int NR_THREADS, int DATASET_SIZE, int DIM, int printNrThreads);
void printPoints(DynamicList* points, int DIM);
void printList(int* list, int DATASET_SIZE, const char* descr);
void printTimeStatistics(double tot_exe_time, double tot_loading, double tot_init_variable, double tot_alg_exe);




// ------------------------------- METHODS TO HANDLE DynamicList TYPE

// Function to initialize the dynamic list
void initDynamicList(DynamicList* list, size_t initial_capacity) {
    list->array = (Point*)malloc(initial_capacity * sizeof(Point));
    if (list->array == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
    list->capacity = initial_capacity;
    list->size = 0;
}

// Function to append a new element to the dynamic list
void appendToDynamicList(DynamicList* list, Point point) {
    if (list->size >= list->capacity) {
        // If the list is full, double its capacity
        list->capacity *= 2;
        list->array = (Point*)realloc(list->array, list->capacity * sizeof(Point));
        if (list->array == NULL) {
            fprintf(stderr, "Memory reallocation failed.\n");
            exit(1);
        }
    }
    // Append the new element
    list->array[list->size++] = point;
}

// Function to access an element of the dynamic list by index
Point* accessByIndex(DynamicList* list, size_t index) {
    if (index >= list->size) {
        fprintf(stderr, "Index out of bounds.\n");
        exit(1);
    }
    return &(list->array[index]);
}

// Function to free memory allocated for the dynamic list
void freeDynamicList(DynamicList* list) {
    free(list->array);
    list->array = NULL;
    list->capacity = 0;
    list->size = 0;
}





// ------------------------------- METHODS TO READ config.properties FILE

 
// Functions to read the size of the dataset and the dimension of points.
void readDatasetSizeAndNumberOfDimensions(char csvFile[100], char csvSplitBy, int* DATASET_SIZE, int* DIM) {

    // Assuming maximum length of numeric string is 20
    char numericString[20];

    int index = 0;
    // Remove non-numeric characters from csvFile and store in numericString
    for (int i = 0; csvFile[i] != '\0'; i++) {
        if (isdigit(csvFile[i])) {
            numericString[index++] = csvFile[i];
        }
    }
    numericString[index] = '\0'; // Null-terminate the string

    *DATASET_SIZE = atoi(numericString);

    FILE* file = fopen(csvFile, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        *DATASET_SIZE = 0;
        *DIM = 0;
    }

    char str[MAX_LINE_LENGTH];
    char* line = str;
    fgets(str, sizeof(str), file);

    *DIM = 1;
    while (*line != '\0') {
        if (*line == csvSplitBy) {
            *DIM += 1;
        }
        line++; // Move to the next character
    }
    // Close the file
    fclose(file);

}

// Function to load the properties in the config.properties file
int loadProperties(char *csvFile, char* csvSplitBy, int* K, int* MAX_ITERATIONS, int* NR_THREADS, int* DATASET_SIZE, int* DIM) {

    FILE* file = fopen("config.properties", "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {

        if (strncmp(line, "#", 1) == 0) continue;

        // Remove trailing newline character
        line[strcspn(line, "\n")] = 0;

        // Parse line to extract key and value
        char* key = strtok(line, "=");
        char* value = strtok(NULL, "=");

        if (strcmp(key, "datasetPath") == 0) {
            strcpy(csvFile, value);
        }
        else if (strcmp(key, "csvCharSplit") == 0) {
            *csvSplitBy = value[0];
        }
        else if (strcmp(key, "numberOfClusters_K") == 0) {
            *K = atoi(value);
        }
        else if (strcmp(key, "maxIterations") == 0) {
            *MAX_ITERATIONS = atoi(value);
        }
        else if (strcmp(key, "nrThreads") == 0) {
            *NR_THREADS = atoi(value);
        }
    }
    // Close the file
    fclose(file);

    readDatasetSizeAndNumberOfDimensions(csvFile, *csvSplitBy, DATASET_SIZE, DIM);

    return 0;
}





// ------------------------------- METHODS SPECIFIC OF KMEANS

// Function for initialization to 0 of the temp. variables : sums and counts for updating the centroids.
void initPartialVariables(int DIM, int K, DynamicList* sums, int *counts) {

    for (int i = 0; i < K; i++) {
        Point point;

        for (int j = 0; j < DIM; j++) {
            point.coordinate[j] = 0;
        }
        appendToDynamicList(sums, point);
        counts[i] = 0;
    }
}




// Function for initialization of the centroids.
void initializeCentroids(int K, int DATASET_SIZE, int DIM, DynamicList* points, DynamicList* centroids) {

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

        Point point;
        Point* dataset = accessByIndex(points, randomNumber);

        for (int j = 0; j < DIM; j++) {
            point.coordinate[j] = dataset->coordinate[j];
        }
        appendToDynamicList(centroids, point);
    }
}


//Function to choose new representative centroids for new clusters.
void updateCentroids(int K, int DIM, DynamicList* centroids, DynamicList* sums, int* counts) {

    // Compute the mean for each cluster to discover new centroid point.
    for (int j = 0; j < K; j++) {

        Point* centroidPoint = accessByIndex(centroids, j);
        Point* sumPoint = accessByIndex(sums, j);
        int count = counts[j];

        for (int dim = 0; dim < DIM; dim++) {
            centroidPoint->coordinate[dim] = sumPoint->coordinate[dim] / count;
        }
    }
    //printList(counts, K, "COUNTS");
    //printPoints(centroids, DIM);

    // reset distance and sum
    for (int j = 0; j < K; j++) {
        Point* sumPoint = accessByIndex(sums, j);
        for (int dim = 0; dim < DIM; dim++) {
            sumPoint->coordinate[dim] = 0;
        }
        counts[j] = 0;
    }
}









// ------------------------------- METHODS OF VARIUOS UTILITY


// Utility function to check if an array already contains a number.
int containsNumber(int* list, int size, int number) {
    for (int i = 0; i < size; i++) {
        if (list[i] == number) {
            return 1; // Number found
        }
    }
    return 0; // Number not found
}





// ------------------------------- METHODS FOR PRINTING OUTPUT 


// Print the parameters read from config.properties file.
void printParameters(char csvFile[100], char csvSplitBy, int K, int MAX_ITERATIONS, int NR_THREADS, int DATASET_SIZE, int DIM, int printNrThreads) {

    printf("Properties Read: \n");
    printf("Csv File %s \n", csvFile);
    printf("CsvSplitBy: %c \n", csvSplitBy);
    printf("K: %i \n", K);
    printf("MAX_ITERATIONS: %i \n", MAX_ITERATIONS);

    printf("DIMENSION SPACE: %i \n", DIM);
    printf("DATASET_SIZE: %i \n", DATASET_SIZE);

    if (printNrThreads == 1) {
        printf("NR_THREADS: %i \n", NR_THREADS);
    }
}

// Function to print a list of points of DIM dimension.
void printPoints(DynamicList* points, int DIM) {

    for (int i = 0; i < points->size; i++) {

        Point* p = accessByIndex(points, i);
        printf("Point %zu: (", i);

        for (int j = 0; j < DIM; j++) {
            printf("%f", p->coordinate[j]);
            if (j != DIM - 1) {
                printf(",");
            }
        }
        printf(")\n");
    }

}

// Function to print a list of integers of N length.
void printList(int* list, int N, const char* descr) {
    printf("\n%s: \n", descr);
    for (int i = 0; i < N; i++) {
        printf("Elem %zu: %i \n", i, list[i]);
    }
}

// Function to print the time statistic of execution time of various part of the code.
void printTimeStatistics(double tot_exe_time, double tot_loading, double tot_init_variable, double tot_alg_exe) {

    printf("\n\nTIME STATISTICS:  \n");
    printf("Total execution time: %f \n", tot_exe_time);
    printf("\nDETAILS:\n");
    printf("Loading dataset time: %f \n", tot_loading);
    printf("Init. Variables time: %f \n", tot_init_variable);
    printf("Algorithm execution: %f \n", tot_alg_exe);
}


char** splitString(const char* str, char delimiter, int* numTokens) {
    // Count the number of tokens
    *numTokens = 1;
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] == delimiter) {
            (*numTokens)++;
        }
    }

    // Allocate memory for array of pointers
    char** tokens = (char**)malloc((*numTokens) * sizeof(char*));
    if (tokens == NULL) {
        printf("Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    // Split the string into tokens
    int tokenIndex = 0;
    const char* tokenStart = str;
    const char* tokenEnd;
    while ((tokenEnd = strchr(tokenStart, delimiter)) != NULL) {
        // Allocate memory for token
        tokens[tokenIndex] = (char*)malloc((tokenEnd - tokenStart + 1) * sizeof(char));
        if (tokens[tokenIndex] == NULL) {
            printf("Memory allocation failed.\n");
            exit(EXIT_FAILURE);
        }

        // Copy token to array
        strncpy(tokens[tokenIndex], tokenStart, tokenEnd - tokenStart);
        tokens[tokenIndex][tokenEnd - tokenStart] = '\0';
        tokenIndex++;
        tokenStart = tokenEnd + 1;
    }

    // Allocate memory for the last token
    tokens[tokenIndex] = (char*)malloc((strlen(tokenStart) + 1) * sizeof(char));
    if (tokens[tokenIndex] == NULL) {
        printf("Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }
    strcpy(tokens[tokenIndex], tokenStart);

    return tokens;
}
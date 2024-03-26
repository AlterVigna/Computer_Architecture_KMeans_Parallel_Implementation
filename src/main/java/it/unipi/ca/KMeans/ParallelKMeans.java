package it.unipi.ca.KMeans;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;



/**
 * Parallel version of KMeans.
 * 
 */
public class ParallelKMeans {

	public static void main(String[] args) throws InterruptedException, ExecutionException {

		// External dataset file info
		String csvFile;
		String csvSplitBy;

		// Algorithm parameters
		int K; // Number of clusters to discover
		int MAX_ITERATIONS; // Stopping condition
		int NR_THREADS;

		Properties properties = new Properties();
		String fileProperties = "config.properties";

		FileInputStream inputConfig = null;
		try {

			// Load properties from config.properties file.
			inputConfig = new FileInputStream(fileProperties);
			properties.load(inputConfig);

			csvFile = properties.getProperty("datasetPath");
			csvSplitBy = properties.getProperty("csvCharSplit");
			K = Integer.parseInt(properties.getProperty("numberOfClusters_K"));
			MAX_ITERATIONS = Integer.parseInt(properties.getProperty("maxIterations"));
			NR_THREADS = Integer.parseInt(properties.getProperty("nrThreads"));

		} catch (Exception e) {

			e.printStackTrace();
			System.err.println("Some of the properties are not configured correctly. The program will be quit..");
			return;
		}

		int DIM = getNumberOfDimensions(csvFile, csvSplitBy); // Dimension of the points in the dataset
		int DATASET_SIZE = Integer.parseInt(Pattern.compile("[^0-9]").matcher(csvFile).replaceAll("").toString());

		// Dataset points a list of points in n-dimensions.
		List<List<Float>> points = new ArrayList<List<Float>>();

		// Result to be achieved: about Cluster informations and composition

		// Centroid points
		List<List<Float>> centroids = new ArrayList<List<Float>>();
		// Membership of each point in the cluster
		List<Integer> membership = new ArrayList<Integer>();
		
		

//----- START MEASUREMENT

		long startMain = System.currentTimeMillis();
		
		ExecutorService executor = Executors.newFixedThreadPool(NR_THREADS);

		System.out.println("Loading the dataset");

		long startLoadDataset = System.currentTimeMillis();

		loadData(executor,csvFile, csvSplitBy, DATASET_SIZE, DIM, NR_THREADS, points, membership);

		long endLoadDataset = System.currentTimeMillis();
		System.out.println("Dataset loaded");

		long startVariableInit = System.currentTimeMillis();

		initializeCentroids(K, points, centroids);

		System.out.println("\nStarting centroids: ");
		printCentroids(centroids);

		long endVariableInit = System.currentTimeMillis();

		System.out.println("\nStart of the algorithm");
		
		long startAlgorithmExecution = System.currentTimeMillis();
		
		int STEP = DATASET_SIZE / NR_THREADS;
		
		for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {

			List<Future<ThreadReturns>> returnList = new ArrayList<Future<ThreadReturns>>();

			int currentDatasetIndex = 0;
			while (currentDatasetIndex < DATASET_SIZE) {

				int startSplit = currentDatasetIndex;
				int endSplit = ((currentDatasetIndex + 2 * STEP <= DATASET_SIZE)) ? (currentDatasetIndex + STEP)
						: DATASET_SIZE;

				Future<ThreadReturns> feature = executor.submit(
						new AssignPointsToClusters(startSplit, endSplit, K, DIM, points, centroids, membership));
				returnList.add(feature);
				currentDatasetIndex = endSplit;

			}

			updateCentroids(K, DIM, returnList, centroids);
		}
		System.out.println("End of the algorithm");

		long endAlgorithmExecution = System.currentTimeMillis();

		System.out.println("\nLast centroids: ");
		printCentroids(centroids);
		
		executor.shutdown();
		try {
			executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		long endMain = System.currentTimeMillis();

//----- END MEASUREMENT

		printTimeElapsed(endMain - startMain, endLoadDataset - startLoadDataset, endVariableInit - startVariableInit,
				endAlgorithmExecution - startAlgorithmExecution);

	}

	/**
	 * Method for loading the dataset into a proper data structure and
	 * initialization of default membership.
	 * @param loaders 
	 * 
	 * @throws ExecutionException
	 * @throws InterruptedException
	 */
	public static void loadData(ExecutorService loaders, String csvFile, String csvSplitBy, int DATASET_SIZE, int N_DIM, int NR_THREADS,
			List<List<Float>> points, List<Integer> membership) throws InterruptedException, ExecutionException {

		int STEP = DATASET_SIZE / NR_THREADS;

		List<Future<List<List<Float>>>> futureData = new ArrayList<Future<List<List<Float>>>>();

		// Split the dataset to be loaded in parts as equal as possible
		int currentDatasetIndex = 0;
		while (currentDatasetIndex < DATASET_SIZE) {

			int startSplit = currentDatasetIndex;
			int endSplit = ((currentDatasetIndex + 2 * STEP <= DATASET_SIZE)) ? (currentDatasetIndex + STEP)
					: DATASET_SIZE;

			Future<List<List<Float>>> loads = loaders
					.submit(new LoadPartialDataset(startSplit, endSplit, csvFile, csvSplitBy, N_DIM));

			futureData.add(loads);
			currentDatasetIndex = endSplit;
		}

		for (int i = 0; i < NR_THREADS; i++) {
			points.addAll(futureData.get(i).get());
		}
		
		// Dataset fully loaded

		//long startInitMemb = System.currentTimeMillis();
		for (int i = 0; i < points.size(); i++) {
			membership.add(0);
		}

		//long endInitMemb = System.currentTimeMillis();
		//System.out.println("Tempo ForkDati: " + (endInitMemb - startInitMemb));
	}

	/**
	 * Function to choose and initialize random centroid from a dataset of points.
	 * 
	 * @param K         - the number of clusters
	 * @param points    - the dataset
	 * @param centroids - the list of central points representing each cluster
	 */
	public static void initializeCentroids(int K, List<List<Float>> points, List<List<Float>> centroids) {
		Random random = new Random(0);
		Set<Integer> randomChosen = new HashSet<Integer>();
		for (int i = 0; i < K; i++) {
			int randomNumber = -1;
			do {
				randomNumber = random.nextInt(points.size());
			} while (randomChosen.contains(randomNumber));
			randomChosen.add(randomNumber);
			centroids.add(new ArrayList<Float>(points.get(randomNumber)));
		}
	}

	/**
	 * Methods to aggregate and compute the new centroids coordinates.
	 * 
	 * @param returnList - the list of the results obtained from each different
	 *                   threads
	 * @throws InterruptedException
	 * @throws ExecutionException
	 */

	/**
	 * Methods to aggregate and compute the new centroids coordinates.
	 * 
	 * @param K          - the number of clusters
	 * @param DIM        - the number of dimension of each point
	 * @param returnList - the list of the results obtained from each different
	 *                   threads
	 * @param centroids  - the list of central points representing each cluster
	 * 
	 * @throws InterruptedException
	 * @throws ExecutionException
	 */
	public static void updateCentroids(int K, int DIM, List<Future<ThreadReturns>> returnList,
			List<List<Float>> centroids) throws InterruptedException, ExecutionException {

		List<List<Float>> sums = new ArrayList<List<Float>>();
		List<Integer> counts = new ArrayList<Integer>();

		for (int j = 0; j < K; j++) {
			List<Float> sum = new ArrayList<Float>();
			for (int dim = 0; dim < DIM; dim++) {
				sum.add(0.0f);
			}
			sums.add(sum);
			counts.add(0);
		}

		// Reduce operation: take all the sums and counts from all the threads and
		// compute a global mean.
		for (Iterator<Future<ThreadReturns>> iterator = returnList.iterator(); iterator.hasNext();) {
			Future<ThreadReturns> future = (Future<ThreadReturns>) iterator.next();
			ThreadReturns ris = future.get();

			for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
				for (int dim = 0; dim < DIM; dim++) {
					sums.get(clusterIndex).set(dim,
							ris.getSums().get(clusterIndex).get(dim) + sums.get(clusterIndex).get(dim));
				}
				counts.set(clusterIndex, ris.getCounts().get(clusterIndex) + counts.get(clusterIndex));
			}
		}

		for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
			for (int dim = 0; dim < centroids.get(clusterIndex).size(); dim++) {
				centroids.get(clusterIndex).set(dim, sums.get(clusterIndex).get(dim) / counts.get(clusterIndex));
			}
		}
	}

	private static class AssignPointsToClusters implements Callable<ThreadReturns> {

		int indexStart;
		int indexEnd;
		int K;
		int DIM;

		List<List<Float>> points = new ArrayList<List<Float>>();
		List<List<Float>> centroids = new ArrayList<List<Float>>();
		List<Integer> membership = new ArrayList<Integer>();

		public AssignPointsToClusters(int start, int end, int K, int DIM, List<List<Float>> points,
				List<List<Float>> centroids, List<Integer> membership) {

			this.indexStart = start;
			this.indexEnd = end;
			this.K = K;
			this.DIM = DIM;

			this.points = points;
			this.centroids = centroids;
			this.membership = membership;

		}

		@Override
		public ThreadReturns call() throws Exception {

			List<Float> dists = new ArrayList<Float>();
			ThreadReturns result = new ThreadReturns(K, DIM);

			for (int j = 0; j < K; j++) {
				dists.add(0.0f);
			}

			for (int i = indexStart; i < indexEnd; i++) {

				List<Float> point = points.get(i);

				for (int j = 0; j < K; j++) {
					dists.set(j, 0.0f);
				}

				for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
					float sumPartial = 0.0f;
					for (int dim = 0; dim < DIM; dim++) {
						sumPartial += Math.pow(centroids.get(clusterIndex).get(dim) - point.get(dim), 2);
					}
					dists.set(clusterIndex, (float) Math.sqrt(sumPartial));
				}

				float min = dists.get(0);
				int minIndex = 0;
				for (int j = 1; j < dists.size(); j++) {
					float currentValue = dists.get(j);
					if (currentValue < min) {
						min = currentValue;
						minIndex = j;
					}
				}

				this.membership.set(i, minIndex);

				for (int dim = 0; dim < DIM; dim++) {
					result.getSums().get(minIndex).set(dim, result.getSums().get(minIndex).get(dim) + point.get(dim));
				}
				result.getCounts().set(minIndex, result.getCounts().get(minIndex) + 1);
			}

			return result;
		}
	}

	private static class LoadPartialDataset implements Callable<List<List<Float>>> {

		int firstRow;
		int finalRow;
		int N_DIM;
		String csvFile;
		String csvSplitBy;
		// List<Integer> membership= new ArrayList<List<Float>>();
		// List<List<Float>> points = new ArrayList<List<Float>>();

		public LoadPartialDataset(int startingRow, int finalRow, String csvFile, String csvSplitBy, int N_DIM) {

			this.firstRow = startingRow;
			this.finalRow = finalRow;

			this.csvFile = csvFile;
			this.csvSplitBy = csvSplitBy;
			this.N_DIM = N_DIM;

		}

		@Override
		public List<List<Float>> call() throws Exception {

			List<List<Float>> points = new ArrayList<List<Float>>();

			// long start=System.currentTimeMillis();
			int row = this.firstRow;
			try {
				Iterator<String> lineIterator = Files.lines(Paths.get(this.csvFile)).skip(this.firstRow + 1).iterator();
				for (row = this.firstRow; row < this.finalRow; row++) {
					String line = lineIterator.next();
					String[] data = line.split(this.csvSplitBy);
					List<Float> point = new ArrayList<Float>();
					for (int dim = 0; dim < N_DIM; dim++) {
						point.add(Float.parseFloat(data[dim]));
					}
					points.add(point);

				}
			} catch (Exception e) {
				PrintStream errorStream = System.err;
				errorStream.println("An arithmetic exception occurred: " + e.getMessage());
				e.printStackTrace();
			}

			// long end=System.currentTimeMillis();
			// System.out.println("Tempo impiegato thread: "+(end-start));

			return points;
		}

	}

	/**
	 * Utility Class to keep the results computed by a thread.
	 */
	public static class ThreadReturns {

		private List<List<Float>> sums = new ArrayList<List<Float>>();
		private List<Integer> counts = new ArrayList<Integer>();

		public ThreadReturns(int K, int DIM) {

			for (int j = 0; j < K; j++) {
				List<Float> sum = new ArrayList<Float>();
				for (int dim = 0; dim < DIM; dim++) {
					sum.add(0.0f);
				}
				this.sums.add(sum);
				this.counts.add(0);
			}
		}

		public List<List<Float>> getSums() {
			return sums;
		}

		public List<Integer> getCounts() {
			return counts;
		}
	}

// Utility functions    

	/**
	 * Utility method to print the centroids coordinates.
	 * 
	 * @param centroids - the list of central points representing each cluster
	 */
	public static void printCentroids(List<List<Float>> centroids) {
		for (int i = 0; i < centroids.size(); i++) {
			System.out.print("Centroid " + i + ": ");
			for (Float val : centroids.get(i)) {
				System.out.print(val + " ");
			}
			System.out.println();
		}
	}

	/**
	 * Utility method to discover the number of space dimension of the cluster
	 * coordinates points.
	 * 
	 * @param csvFile    - the file location of the dataset
	 * @param csvSplitBy - the splitting character to analize a row in the csv file
	 * @return - the number of dimensions of each point
	 */
	public static int getNumberOfDimensions(String csvFile, String csvSplitBy) {

		try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
			String line;
			line = br.readLine();
			String[] ris = line.split(csvSplitBy);
			return ris.length;

		} catch (IOException e) {
			e.printStackTrace();
		}
		return 0;
	}

	/**
	 * Utility method to track execution time among various part of the program.
	 * 
	 * @param totalElapsedTime            - total time between start and end of the
	 *                                    program
	 * @param totalLoadingDatasetTime     - total time needed to load the dataset in
	 *                                    memory
	 * @param totalInitCentroidTime       - total time needed to initialize
	 *                                    centroids
	 * @param totalAlgorithmExecutionTime - total time needed execute the main
	 *                                    algorithm
	 */
	public static void printTimeElapsed(long totalElapsedTime, long totalLoadingDatasetTime, long totalInitCentroidTime,
			long totalAlgorithmExecutionTime) {

		System.out.println("\nTotal execution time (ms): " + totalElapsedTime);

		System.out.println("DETAILS:");
		System.out.println("Loading dataset (ms): " + totalLoadingDatasetTime);
		System.out.println("Init. Variables time (ms): " + totalInitCentroidTime);
		System.out.println("Alg. execution time (ms): " + totalAlgorithmExecutionTime);

	}

}
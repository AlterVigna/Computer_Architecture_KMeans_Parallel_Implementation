package it.unipi.ca.KMeans;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Future;

import it.unipi.ca.KMeans.ParallelKMeans.ThreadReturns;

/**
 * Test Hello world!
 *
 */
public class App {
	public static void main(String[] args) {

		// External file info
		String csvFile = "clustering_dataset.csv";
		String csvSplitBy = ",";

		// Algorithm parameters
		int K = 5; // Number of clusters to discover
		int MAX_ITERATIONS = 1; // Stopping condition

		int DIM = getNumberOfDimensions(csvFile, csvSplitBy); // Dimension of the points in the dataset

		// Dataset points a list of points in n-dimensions.
		List<List<Float>> points = new ArrayList<List<Float>>();

		// Result to be achieved: about Cluster informations and composition

		// Centroid points
		List<List<Float>> centroids = new ArrayList<List<Float>>();
		// Membership of each point in the cluster
		List<Integer> membership = new ArrayList<Integer>();

		System.out.println("Loading the dataset...");

		loadData(csvFile, csvSplitBy, DIM, points, membership);

		System.out.println("Dataset loaded");

		initializeCentroids(K, points, centroids);

		System.out.println("Starting centroids: ");
		printCentroids(centroids);

		// Temp. variables
		List<List<Float>> sums = new ArrayList<List<Float>>();
		List<Integer> counts = new ArrayList<Integer>();

		// Initialization temp. variables.
		for (int j = 0; j < K; j++) {
			List<Float> partialSum = new ArrayList<Float>();
			for (int k = 0; k < DIM; k++) {
				partialSum.add(0.0f);
			}
			sums.add(partialSum);
			counts.add(0);
		}

		for (int nrIteration = 0; nrIteration < MAX_ITERATIONS; nrIteration++) {
			
			// For each point in the cluster, assign its nearest centroid and compute (sums and count) information of each cluster
			assignPointsToCluster(K,points,centroids,sums,counts,membership);

			// Update new Centroids
			updateCentroids(K, DIM, centroids, sums, counts);

		}

		System.out.println("Last centroids: ");
		printCentroids(centroids);

		long fine = System.currentTimeMillis();

	}

	/**
	 * Method for loading the dataset into a proper data structure and
	 * initialization of default membership.
	 */
	public static void loadData(String csvFile, String csvSplitBy, int N_DIM, List<List<Float>> points,
			List<Integer> membership) {
		try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
			String line = br.readLine(); // Skip the first header line
			while ((line = br.readLine()) != null) {
				String[] data = line.split(csvSplitBy);
				List<Float> point = new ArrayList<Float>();
				for (int dim = 0; dim < N_DIM; dim++) {
					point.add(Float.parseFloat(data[dim]));
				}
				points.add(point);
				membership.add(0);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void assignPointsToCluster(int K, List<List<Float>> points, List<List<Float>> centroids,
			List<List<Float>> sums, List<Integer> counts, List<Integer> membership) {

		List<Float> dists = new ArrayList<Float>();
		for (int j = 0; j < K; j++) {
			dists.add(0.0f);
		}

		for (int indexOfPoint = 0; indexOfPoint < points.size(); indexOfPoint++) {

			for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {

				float sumPartial = 0.0f;
				for (int dim = 0; dim < centroids.get(clusterIndex).size(); dim++) {
					sumPartial += Math.pow(centroids.get(clusterIndex).get(dim) - points.get(indexOfPoint).get(dim), 2);
				}

				// Compute distance
				dists.set(clusterIndex, (float) Math.sqrt(sumPartial));
			}

			float min = dists.get(0);
			int minIndex = 0;

			for (int z = 1; z < dists.size(); z++) {

				float currentValue = dists.get(z);
				if (currentValue < min) {
					min = currentValue;
					minIndex = z;
				}
			}
			// Assign to the point, the nearest cluster. 
			membership.set(indexOfPoint, minIndex);
			
			// Save information of the points that belongs to the new cluster in order to update it leater.
			for (int dim = 0; dim < centroids.get(minIndex).size(); dim++) {
				sums.get(minIndex).set(dim, sums.get(minIndex).get(dim) + points.get(indexOfPoint).get(dim));
			}
			counts.set(minIndex, counts.get(minIndex) + 1);
		}

	}

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

	public static void updateCentroids(int K, int DIM, List<List<Float>> centroids, List<List<Float>> sums,
			List<Integer> counts) {

		for (int j = 0; j < K; j++) {
			for (int dim = 0; dim < DIM; dim++) {
				centroids.get(j).set(dim, sums.get(j).get(dim) / counts.get(j));
			}
		}

		// reset distance and sum
		for (int j = 0; j < K; j++) {
			for (int k = 0; k < DIM; k++) {
				sums.get(j).set(k, 0.0f);
			}
			counts.set(j, 0);
		}
	}

	/**
	 * Utility method to print the centroids coordinates.
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

	public static int getNumberOfDimensions(String csvFile, String csvSplitBy) {

		try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
			String line;
			line = br.readLine();
			String[] ris = line.split(csvSplitBy);
			return ris.length; // Initialization of the dimension of points.

		} catch (IOException e) {
			e.printStackTrace();
		}
		return 0;
	}

}

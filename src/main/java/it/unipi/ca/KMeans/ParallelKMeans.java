package it.unipi.ca.KMeans;
import java.util.concurrent.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class ParallelKMeans {

	// External file info
    static String csvFile = "clustering_dataset_10000000.csv";
    static String csvSplitBy = ",";
	

    // Algorithm parameters
    static int K = 5; // Number of clusters to discover
    static int MAX_ITERATIONS = 1; // Stopping condition
    static int DIM = 0; // Dimension of the points in the dataset
    static int DATASET_SIZE = 0; // Number of points in the dataset

    static int NR_THREAD=12;
    // Dataset points a list of points in n-dimensions.
    static List<List<Float>> points = new ArrayList<List<Float>>();

    // Result to be achieved: about Cluster informations and composition
   
    // Centroid points
    static List<List<Float>> centroids = new ArrayList<List<Float>>();
    // Membership of each point in the cluster
    static List<Integer> membership = new ArrayList<Integer>();

    
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        System.out.println("Loading the dataset...");
        
        loadData();

        System.out.println("Dataset loaded");

        initializeCentroids();
       
        System.out.println("Starting centroids: ");
        printCentroids(centroids);
       
        System.out.println("Start of the algorithm");

        long startTime = System.currentTimeMillis();
        
        
    	int STEP=DATASET_SIZE/NR_THREAD;
        
        // Split the dataset in parts as equal as possible
    	List<Integer> splits=new ArrayList<Integer>();
    	
    	int currentDatasetIndex=0;
    	while (currentDatasetIndex<DATASET_SIZE) {
    		currentDatasetIndex=((currentDatasetIndex+STEP<DATASET_SIZE))?(currentDatasetIndex+STEP):DATASET_SIZE;
    		splits.add(currentDatasetIndex);
    	}
    	
    	
    	
    	

        
        ExecutorService executor = Executors.newFixedThreadPool(NR_THREAD);
        List<Future<ThreadReturns>> returnList=new ArrayList<Future<ThreadReturns>>();
        
        
        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        	
        	returnList.clear();
        	currentDatasetIndex=0;
        	
        	for (Iterator<Integer> iterator = splits.iterator(); iterator.hasNext();) {
        		final int startSplit=currentDatasetIndex;
        		Integer endSplit = (Integer) iterator.next();
        		
        		//Future<ThreadReturns> feature = executor.submit(() -> assignPointsToCluster(startSplit,endSplit));
        		Future<ThreadReturns> feature = executor.submit(new MyInfoCallable(startSplit, endSplit, K, DIM, points, centroids, membership));
        				
        		returnList.add(feature);
        		currentDatasetIndex=endSplit;
			}

          executor.shutdown();
          try {
              executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
          } catch (InterruptedException e) {
              e.printStackTrace();
          }
        
          updateCentroids(returnList);
        }

        long end = System.currentTimeMillis();

        System.out.println("Last centroids: ");
        printCentroids(centroids);

        System.out.println("Total Time: " + (end - startTime));
    }

    
    /**
     * Method for loading the dataset into a proper data structure and initialization of default membership.
     */
    public static void loadData() {
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            line = br.readLine(); // Skip the first header line
            String[] ris = line.split(csvSplitBy);
            DIM = ris.length; // Initialization of the dimension of points.
            while ((line = br.readLine()) != null) {
                String[] data = line.split(csvSplitBy);
                List<Float> point = new ArrayList<Float>();
                for (int dim = 0; dim < DIM; dim++) {
                    point.add(Float.parseFloat(data[dim]));
                }
                points.add(point);
                membership.add(0);
            }
            DATASET_SIZE = points.size();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    
    public static void initializeCentroids() {
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

    
    
   public static ThreadReturns assignPointsToCluster(int inizio, int fine) {
    	
    	// Temp. variables
        List<Float> dists = new ArrayList<Float>();
        ThreadReturns results= new ThreadReturns(K);
        
        for (int j=0;j<K;j++) {
        	dists.add(0.0f);
        }
    	
    	for (int i = inizio; i < fine; i++) {
			
    		List<Float> point = points.get(i);
    		
    		for (int j=0;j<K;j++) {
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
    		
            membership.set(i, minIndex);
    		
            for (int dim = 0; dim < DIM; dim++) {
            	results.getSums().get(minIndex).set(dim, results.getSums().get(minIndex).get(dim) + point.get(dim));
            }
            results.getCounts().set(minIndex, results.getCounts().get(minIndex) + 1); 
		}
		return results;
    }

   
    /**
     * Methods to aggregate and compute the new centroids coordinates.
     * 
     * @param returnList
     * @throws InterruptedException
     * @throws ExecutionException
     */
    public static void updateCentroids(List<Future<ThreadReturns>> returnList) throws InterruptedException, ExecutionException {
        
        List<List<Float>> sums = new ArrayList<List<Float>>();
        List<Integer> counts = new ArrayList<Integer>();
        
        for (int j=0;j<K;j++) {
         	List<Float> sum= new ArrayList<Float>();
         	for (int dim=0;dim<DIM;dim++) {
         		sum.add(0.0f);
         	}
         	sums.add(sum);
         	counts.add(0);
         }
    	// Reduce operation: take all the sums and counts from all the threads.
    	for (Iterator<Future<ThreadReturns>> iterator = returnList.iterator(); iterator.hasNext();) {
			Future<ThreadReturns> future = (Future<ThreadReturns>) iterator.next();
			ThreadReturns ris = future.get();
			
			for (int clusterIndex = 0; clusterIndex < K; clusterIndex++) {
	            for (int dim = 0; dim < DIM; dim++) {
	            	
	            	sums.get(clusterIndex).set(dim, ris.getSums().get(clusterIndex).get(dim)+sums.get(clusterIndex).get(dim));
	            	counts.set(clusterIndex,ris.getCounts().get(clusterIndex)+counts.get(clusterIndex));
	            }
	        }
		}
    	
    	for (int clusterIndex=0; clusterIndex<K;clusterIndex++) {
	   		for (int dim=0;dim<centroids.get(clusterIndex).size();dim++) {
	   			centroids.get(clusterIndex).set(dim, sums.get(clusterIndex).get(dim)/counts.get(clusterIndex));
	   		}
	   	}
    	
    }

    /**
     * Utility Class to keep the results computed by a thread. 
    */
    public static class ThreadReturns{
    	
    	private List<List<Float>> sums = new ArrayList<List<Float>>();
        private List<Integer> counts = new ArrayList<Integer>();
        
        public ThreadReturns(int K) {
        	
        	 for (int j=0;j<K;j++) {
             	List<Float> sum= new ArrayList<Float>();
             	for (int dim=0;dim<DIM;dim++) {
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
    
    
    
    private static class MyInfoCallable implements Callable<ThreadReturns> {
    	
    	int inizio;
    	int fine;
    	int K;
    	int DIM;
    	
    	List<List<Float>> points = new ArrayList<List<Float>>();
    	List<List<Float>> centroids= new ArrayList<List<Float>>();
    	List<Integer> membership = new ArrayList<Integer>();
    	
    	
    	public MyInfoCallable(int inizio, int fine,int K, int DIM,
    			List<List<Float>> points,List<List<Float>> centroids,List<Integer> membership) {
    		
    		this.inizio=inizio;
    		this.fine=fine;
    		this.K=K;
    		this.DIM=DIM;
    		
    		this.points=points;
    		this.centroids=centroids;
    		this.membership=membership;

        }
    	
    
		@Override
		public ThreadReturns call() throws Exception {
			
			List<Float> dists = new ArrayList<Float>();
	        ThreadReturns result= new ThreadReturns(K);
	        
	        for (int j=0;j<K;j++) {
	        	dists.add(0.0f);
	        }
	    	
	    	for (int i = inizio; i < fine; i++) {
				
	    		List<Float> point = points.get(i);
	    		
	    		for (int j=0;j<K;j++) {
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
	    		
	            membership.set(i, minIndex);
	    		
	            for (int dim = 0; dim < DIM; dim++) {
	            	result.getSums().get(minIndex).set(dim, result.getSums().get(minIndex).get(dim) + point.get(dim));
	            }
	            result.getCounts().set(minIndex, result.getCounts().get(minIndex) + 1); 
			}

			return result;
		}
    	
    	
    	
    }
    
    
    
    
    
    
    
}
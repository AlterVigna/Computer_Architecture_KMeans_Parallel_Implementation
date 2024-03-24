package it.unipi.ca.KMeans;
import java.util.concurrent.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.io.File;
import java.io.LineNumberReader;


public class ParallelKMeans {

	
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        
    	
    	// External file info
        String csvFile = "clustering_dataset.csv";
        String csvSplitBy = ",";
    	
        int DATASET_SIZE=0;
        int NR_THREAD=10;	// Number of threads to use
        
        // Algorithm parameters
        int K = 5; // Number of clusters to discover
        int MAX_ITERATIONS = 5; // Stopping condition
        
        int DIM = getNumberOfDimensions(csvFile, csvSplitBy);  // Dimension of the points in the dataset
        
        // Dataset points a list of points in n-dimensions.
        List<List<Float>> points = new ArrayList<List<Float>>();

        // Result to be achieved: about Cluster informations and composition
       
        // Centroid points
        List<List<Float>> centroids = new ArrayList<List<Float>>();
        // Membership of each point in the cluster
        List<Integer> membership = new ArrayList<Integer>();

        //evaluation of DATASET_SIZE
        try(LineNumberReader lineNumberReader = new LineNumberReader(new FileReader(new File("clustering_dataset.csv")))) {
                     lineNumberReader.skip(Long.MAX_VALUE);
                     DATASET_SIZE = lineNumberReader.getLineNumber()-1;
        }catch(Exception e){
            System.err.println(e);
        }
        
        System.out.println("Start of the algorithm");

        long startTime = System.currentTimeMillis();
        
        
    	int STEP=DATASET_SIZE/NR_THREAD;
        
        // Split the dataset in parts as equal as possible
    	List<Integer> splits=new ArrayList<Integer>();
    	
        System.out.println("Loading the dataset...");
        	
        ExecutorService loaders = Executors.newFixedThreadPool(NR_THREAD);
        	
        currentDatasetIndex=0;

        /* Sistemare il ritorno del sottoinsieme dei dati*/
        Future<LoaderReturns> futurePoints= new ArrayList<Future<LoaderReturns>>();	
        for (Iterator<Integer> iterator = splits.iterator(); iterator.hasNext();) {
            final int startSplit=currentDatasetIndex;
            Integer endSplit = (Integer) iterator.next();
        		
            Future<LoaderReturns> loads  = executor.submit(new loadPartialDataset(startSplit, endSplit, K, DIM, points, centroids, membership));
        				
            futurepoints.add(loads);
            currentDatasetIndex=endSplit;
        }

        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
                e.printStackTrace();
        }
        //Dataset fully loaded 
        points=futurePoints;
        

        System.out.println("Dataset loaded");

        initializeCentroids(K, points, centroids);
       
        System.out.println("Starting centroids: ");
        printCentroids(centroids);
    	
    	currentDatasetIndex=0;
    	while (currentDatasetIndex<DATASET_SIZE) {
    		currentDatasetIndex=((currentDatasetIndex+STEP<DATASET_SIZE))?(currentDatasetIndex+STEP):DATASET_SIZE;
    		splits.add(currentDatasetIndex);
    	}
    	
        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        	
        	ExecutorService executor = Executors.newFixedThreadPool(NR_THREAD);
            List<Future<ThreadReturns>> returnList=new ArrayList<Future<ThreadReturns>>();
        	
        	currentDatasetIndex=0;
        	
        	for (Iterator<Integer> iterator = splits.iterator(); iterator.hasNext();) {
        		final int startSplit=currentDatasetIndex;
        		Integer endSplit = (Integer) iterator.next();
        		
        		Future<ThreadReturns> feature = executor.submit(new AssignPointsToClusters(startSplit, endSplit, K, DIM, points, centroids, membership));
        				
        		returnList.add(feature);
        		currentDatasetIndex=endSplit;
			}

          executor.shutdown();
          try {
              executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
          } catch (InterruptedException e) {
              e.printStackTrace();
          }
        
          updateCentroids(K,DIM, returnList,centroids);
        }

        long end = System.currentTimeMillis();

        System.out.println("Last centroids: ");
        printCentroids(centroids);

        System.out.println("Total Time: " + (end - startTime));
    }

    
    
    public static void initializeCentroids(int K,List<List<Float>> points,List<List<Float>> centroids) {
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
     * @param returnList
     * @throws InterruptedException
     * @throws ExecutionException
     */
    public static void updateCentroids(int K,int DIM,List<Future<ThreadReturns>> returnList,List<List<Float>> centroids) throws InterruptedException, ExecutionException {
        
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
	            }
	            counts.set(clusterIndex,ris.getCounts().get(clusterIndex)+counts.get(clusterIndex));
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
        
        public ThreadReturns(int K,int DIM) {
        	
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
    
    /*
        Utilily class to keep the result computed by loader Threads
    */
    public static class LoaderReturns{
    	
    	private List<List<Float>> points;
        
        public LoaderReturns () {
        	this.points= new ArrayList<List<Float>>();       	
             }
        }

	public List<List<Float>> getPointsSubset() {
			return points;
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
    
    
    private static class loadPartialDataset implements Callable<LoaderReturns> {
    	
    	int firstRow;
    	int finalRow;
        int N_DIM;
        String csvFile;
        String csvSplitBy;
        List<Integer> membership= new ArrayList<List<Float>>();
       	List<List<Float>> points = new ArrayList<List<Float>>();
    	
    	
    	public loadPartialDataset(int startingRow, int finalRow,List<List<Float>> points,String csvFile,String csvSplitBy,int N_DIM,List<Integer> membership) {
    		
    		this.firstRow=startingRow;
    		this.finalRow=finalRow;
    		this.points=points;
                this.csvFile=csvFile;
                this.csvSplitBy=csvSplitBy;
                this.N_DIM=N_DIM;
                this.membership=membership;
        }
    	
    
	@Override
	public LoaderReturns call() throws Exception {
            
            try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
                Stream<String> lines = Files.lines(Paths.get(this.csvFile));
                Iterator<String> line = lines.skip(this.firstRow-1).findFirst().get();
                for (int row=this.firstRow;row<this.finalRow;i++) {
                    String[] data = line.split(this.csvSplitBy);
                    List<Float> point = new ArrayList<Float>();
                    for (int dim = 0; dim < N_DIM; dim++) {
                        point.add(Float.parseFloat(data[dim]));
                    }
                    points.add(point);
                    membership.add(0);
                    line.next();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            return this.points;
	}
    	
    }
    
    private static class AssignPointsToClusters implements Callable<ThreadReturns> {
    	
    	int indexStart;
    	int indexEnd;
    	int K;
    	int DIM;
    	
    	List<List<Float>> points = new ArrayList<List<Float>>();
    	List<List<Float>> centroids= new ArrayList<List<Float>>();
    	List<Integer> membership = new ArrayList<Integer>();
    	
    	
    	public AssignPointsToClusters(int inizio, int fine,int K, int DIM,
    			List<List<Float>> points,List<List<Float>> centroids,List<Integer> membership) {
    		
    		this.indexStart=inizio;
    		this.indexEnd=fine;
    		this.K=K;
    		this.DIM=DIM;
    		
    		this.points=points;
    		this.centroids=centroids;
    		this.membership=membership;

        }
    	
    
		@Override
		public ThreadReturns call() throws Exception {
			
			List<Float> dists = new ArrayList<Float>();
	        ThreadReturns result= new ThreadReturns(K,DIM);
	        
	        for (int j=0;j<K;j++) {
	        	dists.add(0.0f);
	        }
	    	
	    	for (int i = indexStart; i < indexEnd; i++) {
				
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
	    		
	            this.membership.set(i, minIndex);
	    		
	            for (int dim = 0; dim < DIM; dim++) {
	            	result.getSums().get(minIndex).set(dim, result.getSums().get(minIndex).get(dim) + point.get(dim));
	            }
	            result.getCounts().set(minIndex, result.getCounts().get(minIndex) + 1); 
			}

			return result;
		}
    	
    	
    	
    }
    
    
    public static int getNumberOfDimensions(String csvFile,String csvSplitBy) {
    	
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
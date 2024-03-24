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

/**
 * Test 
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
       
    	// Algorithm parameters
    	int K=5; 	// Number of clusters to discover
    	int MAX_ITERATIONS=1; // Stopping condition
    	int DIM=0; // Dimension of the points in the dataset
    	int DATASET_SIZE=0; // Number of points in the dataset
    	
    	// External file info
    	String csvFile = "clustering_dataset_10000000.csv";
        String csvSplitBy = ",";
        
        
        // Dataset points a list of points in n-dimensions.
        List<List<Float>> points= new ArrayList<List<Float>>();
       
        // Cluster informations
        List<List<Float>> centroids =  new ArrayList<List<Float>>();
        List<List<Float>> sums =  new ArrayList<List<Float>>();
        List<Integer> counts= new ArrayList<Integer>();
        
        
        //Results: membership of each point in the cluster
        List<Integer> membership =  new ArrayList<Integer>();
        
        
        //Temp. variables
        List<Float> dists= new ArrayList<Float>(); 
       
 
        System.out.println("Loading the dataset...");
        long startLoading=System.currentTimeMillis();
        
        
        String line="";
   	 	try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
   	 	line = br.readLine(); 
   	 	String[] ris = line.split(csvSplitBy); // Skip the first header line
   	 
   	 	DIM=ris.length; // Initialization of the dimension of points.
            while ((line = br.readLine()) != null) {
                
            	// Splitting the line by commas
                String[] data = line.split(csvSplitBy);

                //System.out.println("X: " + data[0] + ",Y: " + data[1]);
                List<Float> point= new ArrayList<Float>();
                
                for (int dim=0;dim<DIM;dim++) {
                	point.add(Float.parseFloat(data[dim]));
                }
                
                points.add(point);
                membership.add(0);
            }
            DATASET_SIZE=points.size();
        } catch (IOException e) {
            e.printStackTrace();
        }
   	 	
   	 long endLoading=System.currentTimeMillis();
   	 
   	 System.out.println("Tempo totale Loading:"+(endLoading-startLoading));
   	 	
   	 // Initialization of the centroids	and related informations.
   	for (int j=0;j<K;j++) {
   		List<Float> centr= new ArrayList<Float>();
   		List<Float> partialSum=new ArrayList<Float>();
	   	
   		for(int k=0;k<DIM;k++) {
	     	centr.add(0.0f);
	     	partialSum.add(0.0f);
	   	}
	   	centroids.add(centr);
	   	sums.add(partialSum);
	   	dists.add(0.0f);
	   	counts.add(0);
   	}
   	 		
   	 System.out.println("Dataset loaded");
        
   	 // Centroids initialization
   	 
   	 
   	 // Start of the algorithm
    	
   	 long inizio=System.currentTimeMillis();
   	 
   	 // Select random centroids
   	 Random random = new Random(0);
   	 
   	 Set<Integer> randomChosen= new HashSet<Integer>();
   	 for(int i=0;i<K;i++) {
   		
   		int randomNumber = -1;
   		do {
   			randomNumber=random.nextInt(points.size());
   		}
   		while (randomChosen.contains(randomNumber));
   		randomChosen.add(randomNumber);
   	
   		for (int k=0;k<centroids.get(i).size();k++) {
   			centroids.get(i).set(k, points.get(randomNumber).get(k));
		}
   	 }
   	 
   	 printCentroids(centroids);
   	 
   	 
   	 for (int nrIteration=0;nrIteration<MAX_ITERATIONS;nrIteration++) {
   		 
	   	 for(int indicePunto=0; indicePunto<DATASET_SIZE;indicePunto++) {

	   		 List<Float> point = points.get(indicePunto);
	   		 
	   		 for (int indiceCluster=0; indiceCluster<K;indiceCluster++) {
	   			float sumPartial=0.0f; 
	   			for (int dim=0;dim<centroids.get(indiceCluster).size();dim++) {
	   				sumPartial+=Math.pow(centroids.get(indiceCluster).get(dim)-point.get(dim), 2);
	   			}
	   			 // Compute distance
	   			 dists.set(indiceCluster, (float) Math.sqrt(sumPartial)); 
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
	   		
	        membership.set(indicePunto, minIndex); 
	        	
        	for (int dim=0;dim<centroids.get(minIndex).size();dim++) {
        		sums.get(minIndex).set(dim, sums.get(minIndex).get(dim)+point.get(dim));
        	}
	        	
	        counts.set(minIndex, counts.get(minIndex)+1);
	   	 }

	   	 // Update new Centroids
	   	 
	   	 for (int j=0; j<K;j++) {
	   		for (int dim=0;dim<centroids.get(j).size();dim++) {
	   			centroids.get(j).set(dim, sums.get(j).get(dim)/counts.get(j));
	   		}
	   	 }
	   	 
	   	 
	   	// reset distance and sum
	 	for (int j=0;j<K;j++) {
	   		for(int k=0;k<DIM;k++) {
	   			sums.get(j).set(k,0.0f);
		   	}
		   	dists.set(j,0.0f);
		   	counts.set(j, 0);
	   	}	 
   	 }
   	 long fine=System.currentTimeMillis();
   	 
  
   	System.out.println("Last centroids: ");
   	 
 	printCentroids(centroids);
   	
  	System.out.println("Tempo totale:"+(fine-inizio));
   	   
    }
    
    public static void printCentroids(List<List<Float>> centroids) {
    	
    	System.out.println("Starting centroids: ");
       	int a=1;
       	for (Iterator<List<Float>> iterator = centroids.iterator(); iterator.hasNext();) {
       		List<Float>  centroid = (List<Float>) iterator.next();
       		System.out.print("Centr nr. "+a+" :");
       		for (int k=0;k<centroid.size();k++) {
    			System.out.print(centroid.get(k));
    			if (k!=centroid.size()-1) {
    				System.out.print(",");
    			}
    		}
       		System.out.print("\n");
       		a++;
       	}
    	
    	
    	
    }
    
   
    
    
    
    
    
    
    
}

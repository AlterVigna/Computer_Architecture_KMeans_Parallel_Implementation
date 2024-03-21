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
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
       
    	// Parameters
    	int K=2; 	// Number of clusters to discover
    	int MAX_ITERATIONS=100; // Stopping condition
    	
    	
    	String csvFile = "clustering_dataset_10.csv";
        String csvSplitBy = ",";
        
        
        // Data structures - initialization

        // Dataset points
        List<Float> X = new ArrayList<Float>();
        List<Float> Y = new ArrayList<Float>();
        
     
        // Cluster informations
        List<Float> centroids_X = new ArrayList<Float>(); 
        List<Float> centroids_Y =  new ArrayList<Float>(); 
        List<Integer> membership =  new ArrayList<Integer>();
        
        
        List<Float> dists= new ArrayList<Float>(); 
       	
        List<Float> sums_X= new ArrayList<Float>(); 
        List<Float> sums_Y= new ArrayList<Float>(); 
      	List<Integer> counts= new ArrayList<Integer>();
       	 
        
        for(int i=0;i<K;i++) {
        	
        	centroids_X.add(0.0f);
        	centroids_Y.add(0.0f);
        	membership.add(0);
        	
        	dists.add(0.0f);
        	sums_X.add(0.0f);
        	sums_Y.add(0.0f);
        	counts.add(0);
        	
        }
        	 
        System.out.println("Loading the dataset...");
        
        String line="";
   	 	try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
   	 	 br.readLine(); // Skip the first header line
            while ((line = br.readLine()) != null) {
                
            	// Splitting the line by commas
                String[] data = line.split(csvSplitBy);
                //System.out.println("X: " + data[0] + ",Y: " + data[1]);
                X.add(Float.parseFloat(data[0]));
                Y.add(Float.parseFloat(data[1])); 
                
                membership.add(0);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
      
   	 int DATASET_SIZE=X.size();
   	 
   	 System.out.println("Dataset loaded");
        
   	 // Centroids initialization
   	 
   	 // Select random centroids
   	 Random random = new Random();
   	 
   	 Set<Integer> randomChosen= new HashSet<Integer>();
   	 for(int i=0;i<K;i++) {
   		
   		int randomNumber = -1;
   		do {
   			randomNumber=random.nextInt(X.size());
   			
   		}
   		while (randomChosen.contains(randomNumber));
   		randomChosen.add(randomNumber);
   	
   		centroids_X.set(i, X.get(randomNumber));
   		centroids_Y.set(i, Y.get(randomNumber));
   	 }
   	 
   	 System.out.println("Starting centroids: ");
   	 System.out.println("Centr nr.1 :"+centroids_X.get(0)+","+centroids_Y.get(0));
   	 System.out.println("Centr nr.2 :"+centroids_X.get(1)+","+centroids_Y.get(1));
   	 
   	 // Start of the algorithm
   	
   	 
   	 
   	 for (int nrIteration=0;nrIteration<MAX_ITERATIONS;nrIteration++) {
   		 
	   	 for(int i=0; i<DATASET_SIZE;i++) {
	   		 
	   		 float x=X.get(i);
	   		 float y=Y.get(i);
	   		 
	   		 for (int j=0; j<K;j++) {
	   			 
	   			 float centr_x=centroids_X.get(j);
	   			 float centr_y=centroids_Y.get(j);
	   			 
	   			 // Compute distance
	   			 dists.set(j, (float) Math.sqrt(Math.pow(centr_x-x, 2) + Math.pow(centr_y-y, 2))); 
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
	   		
	        membership.set(i, minIndex); 
	        
	        
	        sums_X.set(minIndex, sums_X.get(minIndex)+x); 
	        sums_Y.set(minIndex, sums_Y.get(minIndex)+y); 
	        counts.set(minIndex, counts.get(minIndex)+1);
	        
	   	 }
	   	 
	   	 
	   	 
	   	 
	   	 
	   	 // Update new Centroids
	   	 
	   	 for (int j=0; j<K;j++) {
	   		 centroids_X.set(j,  sums_X.get(j)/counts.get(j));
	   		 centroids_Y.set(j,  sums_Y.get(j)/counts.get(j));
	   	 }
   	

   	 }
   	 
   	 System.out.println("Last centroids: ");
  	 System.out.println("Centr nr.1 :"+centroids_X.get(0)+","+centroids_Y.get(0));
  	 System.out.println("Centr nr.2 :"+centroids_X.get(1)+","+centroids_Y.get(1));
   	 
   	 
   	 
   	 
   	 
    
        
        
    }
    
    
    
   
    
    
    
    
    
    
    
}

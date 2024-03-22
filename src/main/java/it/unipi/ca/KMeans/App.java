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
    	int K=5; 	// Number of clusters to discover
    	int MAX_ITERATIONS=100; // Stopping condition
    	int DIM=0; // 
    	
    	String csvFile = "clustering_dataset.csv";
        String csvSplitBy = ",";
        
        
        // Data structures - initialization

        // Dataset points
        //List<Float> X = new ArrayList<Float>();
        //List<Float> Y = new ArrayList<Float>();
       
        List<List<Float>> points= new ArrayList<List<Float>>();
       
        // Cluster informations
        List<List<Float>> centroids =  new ArrayList<List<Float>>();
        List<Integer> membership =  new ArrayList<Integer>();
        
        
        List<Float> dists= new ArrayList<Float>(); 
       	
        List<List<Float>> sums =  new ArrayList<List<Float>>();
        List<Integer> counts= new ArrayList<Integer>();
       	 	 
        System.out.println("Loading the dataset...");
        
        String line="";
   	 	try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
   	 	String[] ris = line.split(br.readLine()); // Skip the first header line
   	 	DIM=ris.length+1;
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
        } catch (IOException e) {
            e.printStackTrace();
        }
   	 // Initialization of centroids	
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
   	 		
      
   	 int DATASET_SIZE=points.size();
   	 
   	 System.out.println("Dataset loaded");
        
   	 // Centroids initialization
   	 
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
   	 
   	 // Start of the algorithm
   	
   	 long inizio=System.currentTimeMillis();
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
   	 
 	a=1;
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
   	
  	System.out.println("Tempo totale:"+(fine-inizio));
   	   
    }
    
    
    
   
    
    
    
    
    
    
    
}

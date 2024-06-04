This is a project for Computer Architecture course in Computer Engineering curriculum, free choice activity for Artificial Intelligence and Data Engineering Master Degree at the University of Pisa.
It contains the source code and the presentation about the implementation of parallel solutions of the KMeans unsupervised learning algorithm, first on CPU and then on GPU.
We started from a simple sequential implementation on CPU in C language moving to different parallel versions exploiting threads and introducing various optimizations like (thread-affinity and new data structcures).
We exploit the Intel V Tune profiler to guide our decisions and improving our first version code to try to exploit at maxiumum our hardware architecture.
Finally, we develop several GPU versions using Cuda C and we used the Nsight Compute to understand and improve our bottlenecks according to our GPU architecture. 
All the details are illustrated inside the file CA_Presentation.pdf

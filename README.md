## Operating Instructions
**Initial setup**
```
pip install -r requirements.txt
source venv/bin/activate 
```

**To run the dash visualizations**
```
python visualizations.py 
```

**To run the graph.py source code**
```
python graph.py 
```

**To compile the dynamic connectivity file** 
*Using pybind and cmake to compile c++ into modules python can import. Modified from Tom Tseng's implementation and added pybind modules in **dynamic_connectivity.cpp**. Original implementation linked in resources references. 
```
#create a build directory 
cd compile 
mkdir build 
cd build 

#compile then build 
cmake .. \ -DCMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
cmake --build .
```
The **dynamic_connectivity.so** file can now be imported by python. 


## Lectures/Papers/Resources Referenced: 
1. Online Graph Connectivity Function:
    - MIT Lecture: https://courses.csail.mit.edu/6.851/spring12/scribe/L20.pdf
    - Wikipedia Dynamic Connectivity: https://en.wikipedia.org/wiki/Dynamic_connectivity
    - Stanford Euler Tour Trees: https://web.stanford.edu/class/archive/cs/cs166/cs166.1146/lectures/04/Small04.pdf 
    - Stanford Euler Tour Trees: https://web.stanford.edu/class/archive/cs/cs166/cs166.1146/lectures/17/Slides17.pdf
    - Codeforces Online Dynamic Connectivity Blog: https://codeforces.com/blog/entry/128556
    - Dynamic Connectivity Implementation: https://github.com/tomtseng/dynamic-connectivity-hdt **Implementation is in C but I created modules using pybind and compiled it into the dynamic_connectivity file included here**

2. Optimal Satisficing And OR Tree: https://www.sciencedirect.com/science/article/pii/S0004370205001438



## Progress Log

### 9/5:
Tasks: N/A

Meeting:
1. Meet with Evan + Haya 
2. Hellerstein introduced the concept of graph connectivity problem  

### 9/12:
Tasks:
1. Created Boolean class + algorithm  
2. Started Graph class 

Meeting: 
1. Finished Graph class
2. Issue with efficient disconnect function 

### 9/19:
Tasks: 
1. Proved checking endpoints determine graph connectivity 
2. Started implementing Graph algorithm 
3. Working monte carlo simulation 

Meeting: 
1. Discussed further options 

### 9/26: 
Tasks: 
1. Implemented Graph c++ within python 
2. Generate some preliminary webapp simulations 
3. Investigate 2 connectivity 

Meeting:
1. Discussed issue with depth first search as an optimal algorithm 
2. Discussed the approximation algo ideas 

### 10/3:
Tasks: 
1. Simulation
2. Meet with Haya 

Meeting:
1. Demoed the webapp
2. Discussed the possiblity of 2 approx to 1 certificate. Largest bundle is the mole. 0 certificate is the problem. 

### 10/17: 
Tasks: 
1. Implement 1 verification algo + show optimality 
2. Implement 0 verification heuristic dfs and compare empirically 
3. Webapp can model much more algos 

Meeting: 
1. Discussed the matroids 
2. Discussed the greedy algorithm choosing largest pi as optimal 1 certificate verification 


### 10/28 
Tasks: 
1. Create an input method 
2. Create an algorithm selector 

Meeting: 
1. Discussed the possibility of implementing the approximation algo 
2. Approximation of approximation SODA paper: approx optimal 


### 10/31
Tasks: 
1. Writeup to disprove the f(x)~1 vs f(x)~2 
2. Writeup for the smallest bin to test for 0 certificates RR with 1 certificate

### 11/21 
Tasks:
1. dp exponential time find the optimal algorithm 
2. prove that ci/pi DFA is optimal for verification algorithm when x has one 0 bundle 

Meeting: 
1. What is a local search? 
2. Simulated annealing? 
3. Genetic programming?

### 12/ 19
Tasks:
1. Complete the writeup 

Meeting: 
1. Potential dp algorithm for the f(x) = 1 case 
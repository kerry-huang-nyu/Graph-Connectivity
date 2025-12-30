#include "dynamic_connectivity.hpp"
#include <iostream>
using namespace std;

int main(){
    DynamicConnectivity graph(6);

    // Graph is two triangles:
    //   0          5
    //   |\        /|
    //   | \      / |
    //   2--1    4--3
    graph.AddEdge({0, 1});
    graph.AddEdge({1, 2});
    graph.AddEdge({2, 0});
    graph.AddEdge({3, 4});
    graph.AddEdge({4, 5});
    graph.AddEdge({5, 3});

    cout << "Expects True" << graph.IsConnected(0, 2) << "\n";
    cout << "Expects False" << graph.IsConnected(0, 5) << "\n";

    return 0;
}
'''
API Key: b28f0f373c33c72b563bdb1267874ea1d86d474254a8febf144fc2ee90d67577
'''
#USD/Yen/GBP/KRW/CAD
from math import log
import csv



class Arbitrage:
    def __init__(self, currencies, rates):
        # currencies = ["USD", "GBP", "YEN", ...]
        # rates = [ ["USD", "GBP", 1.2], ["GBP", "USD", 5/6], ["USD", "YEN", 5.0], ...]
        self.graph = Graph(currencies)
        for rate in rates:
            self.graph.addEdge(rate[0], rate[1], rate[2])
        self.graph.modifyEdges()

    def find_arbitrage(self):
        self.graph.bellmanford(self.graph.V[0])

class Graph:
    def __init__(self, vertices):
        self.V = vertices # no. of verts
        self.graph = []

    def addEdge(self, u, v, w):
        self.graph.append([u,v,w])

    def modifyEdges(self):
        for edge in self.graph:
            edge[2] = (-1) * log(edge[2])

    def bellmanford(self, src):

        dist = {}
        for v in self.V:
            dist[v] = float("Inf")
        dist[src] = 0

        n = len(self.V)
        pre = {}
        for vert in self.V:
            pre[vert] = None

        for i in range(len(self.V)-1):
            for edge in self.graph:
                v1 = edge[0]
                v2 = edge[1]
                w = edge[2]
                if dist[v1] != float("Inf") and dist[v1] + w < dist[v2]:  
                        dist[v2] = dist[v1] + w  
                        pre[v2] = v1

        for edge in self.graph:
            v1 = edge[0]
            v2 = edge[1]
            w = edge[2]
            if dist[v1] != float("Inf") and dist[v1] + w < dist[v2]:
                print_cycle = [v2, v1]
                # Start from the source and go backwards until you see the source vertex again or any vertex that already exists in print_cycle array
                while pre[v1] not in  print_cycle:
                    print_cycle.append(pre[v1])
                    v1 = pre[v1]
                print_cycle.append(pre[v1])
                print("Arbitrage Opportunity: \n")
                print(" --> ".join([p for p in print_cycle[:-1]]))
                print("Negative weight cycle")
        return dist


 


if __name__ == "__main__":

    data = []
    curr = []
    with open('math260/project/data5.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row[2] = float(row[2])
            data.append(row)
            if row[0] not in curr:
                curr.append(row[0])

    b = Arbitrage(curr, data)
    b.find_arbitrage()
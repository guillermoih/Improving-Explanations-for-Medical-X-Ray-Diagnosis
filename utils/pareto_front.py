import matplotlib.pyplot as plt
import numpy as np


# # Get dominates

# +
def dominates(row, candidateRow):
    return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row) 

def simple_cull(inputPoints, get_idx=False):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    
    if get_idx:
        dominates_idx = []
        
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints = [arr for arr in inputPoints if not np.array_equal(arr, row)]
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
                
                if get_idx:
                    dominates_idx.append(rowNr)
                    
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
            
    if get_idx:
        return paretoPoints, dominatedPoints, dominates_idx
    else:
        return paretoPoints, dominatedPoints


# -

# # Plot pareto frontier

def plot_pareto_frontier(fits, class_chs, maxX=False, maxY=False):
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.set_xlabel('Classification')
    ax.set_ylabel('Mutations magnitude')
    ax.set_zlabel('Mutations number')
    
    pareto_pts, non_pareto_pts = simple_cull(list(fits))
    pareto_pts = np.asarray(list(pareto_pts))
    non_pareto_pts = np.asarray(list(non_pareto_pts))
    if len(pareto_pts) > 0:
        if len(pareto_pts) > 2:
            ax.plot_trisurf(-pareto_pts[:,0], pareto_pts[:,1], pareto_pts[:,2],
                        alpha=0.2, cmap='cool')
    
    scatter = ax.scatter3D(-fits[:,0], fits[:,1], fits[:,2],
                           cmap='cool', c=class_chs)
    legend = ax.legend(*scatter.legend_elements(alpha=0.4),
                       loc="lower left")
    legend.get_texts()[0].set_text('Original classification')
    if len(legend.get_texts()) > 1:
        legend.get_texts()[1].set_text('Missclassified')
    ax.add_artist(legend)
    
    plt.show()

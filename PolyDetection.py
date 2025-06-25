import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import networkx as nx



# Matplotlib setup
fig, ax = plt.subplots(1,4, figsize=(20,5))
square = patches.Rectangle((0.0, 0.0), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
ax[0].add_patch(square)
ax[0].set_xlim(0, 1)
ax[0].set_ylim(0, 1)
ax[0].set_aspect('equal')
ax[1].set_aspect('auto')
ax[2].set_aspect('auto')
ax[3].set_aspect('auto')
plt.bone()



# --------------------------------- Vertex class ---------------------------------
class Vertex:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    def __eq__(self, other):
        return np.isclose(self.x, other.x) and np.isclose(self.y, other.y)
    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6)))
    def __repr__(self):
        return f"Vertex({self.x:.4f}, {self.y:.4f})"

#---------------------------------------------------------------------------

# Corners of the square
corners = [
    Vertex(0.0, 0.0),
    Vertex(1.0, 0.0),
    Vertex(1.0, 1.0),
    Vertex(0.0, 1.0)
]



# --------------------------------- Line Class ---------------------------------

digits = 4

class Line:
    _id_counter = 0  # class variable for unique IDs

    def __init__(self, v1=None, v2=None, id=None):
        if v1 is not None and v2 is not None:
            self.v1 = v1
            self.v2 = v2
        else:
            rand=random.randint(0,3)
            val1=np.round(random.random(), digits)
            val2=np.round(random.random(), digits)
            int1=random.randint(0,1)
            int2=abs(int1-1)
            if rand == 0:
                self.v1 = Vertex(int1, val1)
                self.v2 = Vertex(int2, val2)
            elif rand==1:
                self.v1 = Vertex(val1, int1)
                self.v2 = Vertex(val2, int2)
            elif rand==2:
                self.v1 = Vertex(int1, val1)
                self.v2 = Vertex(val2, int2)
            elif rand==3:
                self.v1 = Vertex(val1, int1)
                self.v2 = Vertex(int2, val2)
        if id is not None:
            self.id = id
        else:
            self.id = Line._id_counter
            Line._id_counter += 1
        self.intersections = []
        self.intersected_lines = set()
        self.x = [self.v1.x, self.v2.x]
        self.y = [self.v1.y, self.v2.y]
    def __str__(self):
        return f"Line {self.id}: ({self.v1.x},{self.v1.y}) ({self.v2.x},{self.v2.y})"
    def as_tuple(self):
        return (self.v1, self.v2)
    
    def intersect(self, other):
        # Returns intersection point as Vertex if segments intersect, else None
        x1, y1 = self.v1.x, self.v1.y
        x2, y2 = self.v2.x, self.v2.y
        x3, y3 = other.v1.x, other.v1.y
        x4, y4 = other.v2.x, other.v2.y

        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if np.isclose(denom, 0):
            return None  # Parallel lines

        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

        # Check if intersection is within both segments
        def between(a, b, c):
            return min(a, b) - 1e-8 <= c <= max(a, b) + 1e-8

        if (between(x1, x2, px) and between(y1, y2, py) and
            between(x3, x4, px) and between(y3, y4, py)):
            def add_intersected_line(self, other_line):
                self.intersected_lines.add(other_line)
            add_intersected_line(self, other)
            add_intersected_line(other, self)
            return Vertex(np.round(px, digits), np.round(py, digits))
        return None

# ------------------------------------------------------------------------------

# -------------------------------Polygon Class----------------------------------
class Polygon:
    _id_counter = 0  # class variable for unique IDs
    lines=[]
    def init(self, vertices, id=None):
        self.vertices=vertices
        if id is not None:
            self.id = id
        else:
            self.id = Polygon._id_counter
            Polygon._id_counter += 1





#=============================== HELPER FUNCTIONS ======================================


def create_lines(n):
    lines = []
    for idx in range(n):
        line = Line()
        print(f"{line}")
        lines.append(line)
    return lines

# Find all intersections
print("Finding intersections and splitting lines...")
def find_intersections(lines):
    intersections = {i: set([lines[i].v1, lines[i].v2]) for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = lines[i].intersect(lines[j])
            if pt:
                intersections[i].add(pt)
                intersections[j].add(pt)
    return intersections

# Split lines at intersection points
def split_line_at_points(line, points):
    # Sort points along the line
    if abs(line.v1.x - line.v2.x) > abs(line.v1.y - line.v2.y):
        points = sorted(points, key=lambda v: v.x)
    else:
        points = sorted(points, key=lambda v: v.y)
    segments = []
    # Remove duplicate points (using set), then sort again
    unique_points = list(set(points))
    if abs(line.v1.x - line.v2.x) > abs(line.v1.y - line.v2.y):
        unique_points.sort(key=lambda v: v.x)
    else:
        unique_points.sort(key=lambda v: v.y)
    # Replace -0.0 with 0.0 for x and y
    for v in unique_points:
        if np.isclose(v.x, 0.0):
            v.x = 0.0
        if np.isclose(v.y, 0.0):
            v.y = 0.0
    for i in range(len(unique_points)-1):
        if unique_points[i] != unique_points[i+1]:
            seg = Line(unique_points[i], unique_points[i+1])
            segments.append(seg)
    return segments


   
#shoelace formula for area of polygon
#x[:-1] is [x1,x2,...,xn] and x[1:] is [x2,x3,...xn+1] so the terms pair up
def areaOfPoly(poly):
    poly_x = []
    poly_y = []
    for vert in poly:
        poly_x.append(vert[0])
        poly_y.append(vert[1]) 
    if poly_x[0] != poly_x[-1] or poly_y[0] != poly_y[-1]:
        poly_x.append(poly_x[0])
        poly_y.append(poly_y[0])
    poly_x = np.array(poly_x)
    poly_y = np.array(poly_y)
    A = 0.5 * np.abs(np.sum(poly_x[:-1] * poly_y[1:] - poly_x[1:] * poly_y[:-1]))
    return A


# Pick a random polygon from a list of polygons
def pick_a_poly(polys):
    print(len(polys))
    poly=random.randint(0, len(polys)-1)
    print(poly)
    return polys[poly]

# Pick a random point along a line
def pick_a_point(v1, v2):
    print(v1)
    print(v2)
    x = round(random.uniform(v1[0], v2[0]), digits)
    y = round(random.uniform(v1[1], v2[1]), digits)
    vert=Vertex(x, y)
    print(vert)
    return vert
    
# TODO NEED TO FIND SOME WAY TO GET THE LINES THEMSELVES INTO THE CUT_A_POLY METHOD
'''
Possibly make a polygon class with a list of vertices and lines?
'''

#cut a polygon by choosing two points on two sides and connecting them with a line, removing the original polygon and forming two new polygons
def cut_a_poly(poly):
    s1=random.randint(1, len(poly))
    s2=random.randint(1, len(poly))
    while(s1==s2):
        s2=random.randint(1, len(poly))
    p1=pick_a_point(poly[s1-1], poly[s1])
    p2=pick_a_point(poly[s2-1], poly[s2])
    points=[p1, p2]
    lines=[]
    line= Line(p1, p2)
    lines.append(line)
    split_l1=split_line_at_points(s1, points)
    split_l2=split_line_at_points(s2, points)
    lines.extend(split_l1)
    lines.extend(split_l2)
    cut_edges = [((round(line.v1.x, digits), round(line.v1.y, digits)), (round(line.v2.x, digits), round(line.v2.y, digits))) for line in lines]
    graph=nx.graph()
    graph.add_edges_from(cut_edges)
    new_polys=nx.minimum_cycle_basis(graph)
    for p in new_polys:print(f"Cut Poly:{p}") 

#=============================== END HELPER FUNCTIONS ======================================

#==================================== BODY OF CODE =========================================



# Add random lines and border lines
lines = create_lines(3)
border_lines = [
    Line(corners[0], corners[1]),
    Line(corners[1], corners[2]),
    Line(corners[2], corners[3]),
    Line(corners[3], corners[0])
]
lines += border_lines


intersections=find_intersections(lines)
final_lines = []
for idx, line in enumerate(lines):
    pts = list(intersections[idx])
    segments = split_line_at_points(line, pts)
    final_lines.extend(segments)


# Plot all segments
for line in final_lines:
    ax[0].plot(line.x, line.y, color='b')

# Find Polygons
edges = [((round(line.v1.x, digits), round(line.v1.y, digits)), (round(line.v2.x, digits), round(line.v2.y, digits))) for line in final_lines]
print("Finding Polygons...")
G = nx.Graph()
G.add_edges_from(edges)
polys = nx.minimum_cycle_basis(G)
print("Detected polygons:", polys)
print("Number of Polygons:", len(polys))
totalverts=0

# Data
listPolySizes=[]
listPolyAreas=[]
vBarLines=[]
for poly in polys:
    totalverts+=len(poly)
    listPolySizes.append(len(poly))
    listPolyAreas.append(areaOfPoly(poly))
    vBarLines.append(totalverts/len(listPolySizes))    
print("Average number of vertices:", totalverts/(len(polys))) 
indeces=[]
for i in range(0,len(listPolyAreas)):
    indeces.append(i)

p=pick_a_poly(polys)
cut_a_poly(p)


# Plotting data
ax[1].hist(listPolySizes)
ax[1].set_xlabel("Number of Sides")
ax[1].set_ylabel("Number of Polygons")
ax[2].hist(listPolyAreas)
ax[2].set_xlabel("Area of Polygons")
ax[2].set_ylabel("Number of Polygons")
ax[3].plot(indeces,vBarLines)
ax[3].set_xlabel("Number of Polygons")
ax[3].set_ylabel("Avg. Number of Sides")
plt.subplot_tool()
plt.show()


'''
Pick a random polygon
Pick two random points (on different sides)
Make a line connecting those two points
Add the two new polygons to the list of polygons
Remove the initial polygon
Take the number of each n-gon after every cut. (DO THIS IN A SMART WAY. DONT JUST TAKE A RANDOM )

'''


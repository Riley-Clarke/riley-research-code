import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import networkx as nx
from collections import Counter
import json



# Matplotlib setup
plt.figure(figsize=(10,10))
plt.bone()



digits = 4
# --------------------------------- Vertex class ---------------------------------
class Vertex:
    reg = 0
    irr = 0

    def __init__(self, x, y):
        self.x = round(float(x), digits)
        self.y = round(float(y), digits)
        self.tp = (self.x, self.y)
    def __eq__(self, other):
        return np.isclose(self.x, other.x) and np.isclose(self.y, other.y)
    def __eqtp__(self, other):
        return np.isclose(self[0], other[0]) and np.isclose(self[1], other[1])
    def __hash__(self):
        return hash((round(self.x, digits), round(self.y, digits)))
    def __repr__(self):
        return f"Vertex({self.x:4f}, {self.y:4f})"
    def as_tuple(self):
        return (self.x, self.y)

#---------------------------------------------------------------------------

# Corners of the square
corners = [
    Vertex(0.0, 0.0),
    Vertex(1.0, 0.0),
    Vertex(1.0, 1.0),
    Vertex(0.0, 1.0)
]



# --------------------------------- Line Class ---------------------------------



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
        # TODO CHANGED SELF.X AND Y TO TUPLES FOR TESTING PURPOSES AND ADDED P1 AND P2 AS TUPLES
        self.x = (self.v1.x, self.v2.x)
        self.y = (self.v1.y, self.v2.y)
        self.p1= (self.v1.x, self.v1.y)
        self.p2= (self.v2.x, self.v2.y)
    def __str__(self):
        return f"Line {self.id}: ({self.v1.x},{self.v1.y}) ({self.v2.x},{self.v2.y})"
    def as_tuple(self):
        return (self.v1, self.v2)
    def intersect(self, other, tol=0.0001):
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

        # Check if intersection is within both segments (with tolerance)
        def between(a, b, c):
            return (min(a, b) - tol <= c <= max(a, b) + tol) or np.isclose(c, a, atol=tol) or np.isclose(c, b, atol=tol)

        if (between(x1, x2, px) and between(y1, y2, py) and
            between(x3, x4, px) and between(y3, y4, py)):
            def add_intersected_line(self, other_line):
                self.intersected_lines.add(other_line)
            add_intersected_line(self, other)
            add_intersected_line(other, self)
            return Vertex(np.round(px, digits), np.round(py, digits))
        return None

# ------------------------------------------------------------------------------


#=============================== HELPER FUNCTIONS ======================================


def create_lines(n):
    lines = []
    for idx in range(n):
        line = Line()
        print(f"{line}")
        lines.append(line)
    return lines

def snap_to_existing_vertices(pt, existing_vertices, tol=0.001):
    # pt is a Vertex, existing_vertices is a list of Vertex
    for v in existing_vertices:
        if np.isclose(pt.x, v.x, atol=tol) and np.isclose(pt.y, v.y, atol=tol):
            return Vertex(v.x, v.y)  # Snap to existing vertex
    return pt  # No match, keep as is



# Find all intersections
print("Finding intersections and splitting lines...")
def find_intersections(lines):
    # Collect all existing vertices
    all_vertices = []
    for line in lines:
        all_vertices.extend([line.v1, line.v2])
    intersections = {i: set([lines[i].v1, lines[i].v2]) for i in range(len(lines))}
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = lines[i].intersect(lines[j])
            if pt:
                pt = snap_to_existing_vertices(pt, all_vertices)
                intersections[i].add(pt)
                intersections[j].add(pt)
    return intersections

# Split lines at intersection points
def split_line_at_points(line, points):
    # Collect all existing vertices for snapping
    existing_vertices = [line.v1, line.v2]
    points = [snap_to_existing_vertices(pt, existing_vertices) for pt in points]
    # Sort points along the line
    if abs(line.p1[0] - line.p2[0]) > abs(line.p1[1] - line.p2[1]):
        points = sorted(points, key=lambda v: v.x)
    else:
        points = sorted(points, key=lambda v: v.y)
    segments = []
    unique_points = list({(v.x, v.y): v for v in points}.values())  # Remove duplicates by coordinates
    if abs(line.p1[0] - line.p2[0]) > abs(line.p1[1] - line.p2[1]):
        unique_points.sort(key=lambda v: v.x)
    else:
        unique_points.sort(key=lambda v: v.y)
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
    areas = [areaOfPoly(poly) for poly in polys]
    chosen_poly = random.choices(polys, weights=areas, k=1)[0]
    return chosen_poly

# Pick a random point along a line
def pick_a_point(v1, v2):
    # v1 and v2 are Vertex objects
    t = random.uniform(0, 1)
    # Ensure t is not 0 or 1 (not endpoints)
    while np.isclose(t, 0) or np.isclose(t, 1):
        t = random.uniform(0, 1)
    x = v1[0] + t * (v2[0] - v1[0])
    y = v1[1] + t * (v2[1] - v1[1])
    
    return Vertex(round(x, digits), round(y, digits))

def update_poly(poly, points):
    # Checking if the given polygon has any of the given points on any of its lines and updating it if so.
    updated_poly = poly.copy()
    for point in points:
        for i in range(len(poly)):
            v1 = poly[i]
            v2 = poly[(i + 1) % len(poly)]
            # Check if point is on the line segment v1-v2
            x1, y1 = v1[0], v1[1]
            x2, y2 = v2[0], v2[1]
            px, py = point.x, point.y
            # Compute cross product to check colinearity
            cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
            if np.isclose(cross, 0, atol=0.0005):
                # Check if point is between v1 and v2
                if min(x1, x2) - 0.0005 <= px <= max(x1, x2) + 0.0005 and min(y1, y2) - 0.0005 <= py <= max(y1, y2) + 0.0005:
                    # Insert point between v1 and v2 if not already present
                    idx = i + 1
                    if (round(px, digits), round(py, digits)) not in [(round(v[0], digits), round(v[1], digits)) for v in updated_poly]:
                        updated_poly.insert(idx, Vertex(px, py).as_tuple())
    return updated_poly

    
#cut a polygon by choosing two points on two sides and connecting them with a line, removing the original polygon and forming two new polygons
def cut_a_poly(poly, poly_list, max_attempts=10):
    for attempt in range(max_attempts):
        print(poly)
        s1=random.randint(0, len(poly)-1)
        s2=random.randint(0, len(poly)-1)
        v_of_s1=None
        v_of_s2=None
        while(s1==s2):
            s2=random.randint(0, len(poly)-1)
        p1=None
        p2=None
        if(s1==len(poly)-1):
            p1=pick_a_point(poly[s1], poly[0])
            v_of_s1=[poly[s1], poly[0]]
            p2=pick_a_point(poly[s2], poly[s2+1])
            v_of_s2=[poly[s2], poly[s2+1]]
        elif(s2==len(poly)-1):
            p1=pick_a_point(poly[s1], poly[s1+1])
            v_of_s1=[poly[s1], poly[s1+1]]
            p2=pick_a_point(poly[s2], poly[0])
            v_of_s2=[poly[s2], poly[0]]
        else:
            p1=pick_a_point(poly[s1], poly[s1+1])
            v_of_s1=[poly[s1], poly[s1+1]]
            p2=pick_a_point(poly[s2], poly[s2+1])
            v_of_s2=[poly[s2], poly[s2+1]]
        points=[p1, p2]
        lines=[]
        for v in range(len(poly)):
            if(v==len(poly)-1):
                v1=Vertex(poly[v][0], poly[v][1])
                v2=Vertex(poly[0][0], poly[0][1])
                lines.append(Line(v1, v2))
            else:
                v1=Vertex(poly[v][0], poly[v][1])
                v2=Vertex(poly[v+1][0], poly[v+1][1])
                lines.append(Line(v1, v2))
        n_line= Line(p1, p2)
        print(f"New Line: {n_line}\n")
        lines.append(n_line)

        intersections=find_intersections(lines)
        print(intersections)
        final_lines = []
        for idx, line in enumerate(lines):
            pts = list(intersections[idx])
            segments = split_line_at_points(line, pts)
            final_lines.extend(segments)
            
        cut_edges = [((round(line.v1.x, digits), round(line.v1.y, digits)), (round(line.v2.x, digits), round(line.v2.y, digits))) for line in final_lines]
        print(f"Cut edges {cut_edges}\n")
        graph=nx.Graph()
        graph.add_edges_from(cut_edges)
        new_polys=nx.minimum_cycle_basis(graph)
        #Possibly hardcode it to make sure the length of new polys is 2 after every cut?
        if(len(new_polys)==2):
            poly_list.remove(poly)
            touching_polys = []
            poly_set = set(tuple([round(v[0], digits), round(v[1], digits)]) for v in poly)
            for other_poly in poly_list:
                other_set = set(tuple([round(v[0], digits), round(v[1], digits)]) for v in other_poly)
                if poly_set & other_set:
                    touching_polys.append(other_poly)
            for tp in touching_polys:
                if tp in poly_list:
                    poly_list.remove(tp)
                    updated_tp = update_poly(tp, points)
                    print(updated_tp)
                    poly_list.append(updated_tp)

            for p in new_polys:
                print(f"Cut Poly:{p}\n") 
                poly_list.append(p)
            plt.plot(n_line.x, n_line.y, color='r')
            
            return poly_list
    #print("Max attempts reached, skipping this cut.")
    return poly_list
    



#=============================== END HELPER FUNCTIONS ======================================

#==================================== BODY OF CODE =========================================


for k in range(1):
    # Add random lines and border lines
    #lines = create_lines(15)
    border_lines = [
        Line(corners[0], corners[1]),
        Line(corners[1], corners[2]),
        Line(corners[2], corners[3]),
        Line(corners[3], corners[0])
    ]
    #lines += border_lines
    lines = border_lines

    intersections=find_intersections(lines)
    final_lines = []
    for idx, line in enumerate(lines):
        pts = list(intersections[idx])
        segments = split_line_at_points(line, pts)
        final_lines.extend(segments)


    # Plot all segments
    for line in final_lines:
        plt.plot(line.x, line.y, color='b')

    # Find Polygons
    edges = [((round(line.v1.x, digits), round(line.v1.y, digits)), (round(line.v2.x, digits), round(line.v2.y, digits))) for line in final_lines]
    print("Finding Polygons...")
    G = nx.Graph()
    G.add_edges_from(edges)
    polys = nx.minimum_cycle_basis(G)
    #print("Detected polygons:", polys)
    #print("Number of Polygons:", len(polys))
    totalverts=0

    # Data

    num_ngons=[]

    num_cuts=5
    for i in range(num_cuts):
        p=pick_a_poly(polys)
        polys=cut_a_poly(p, polys)
        side_counts=Counter(len(poly) for poly in polys)
        num_ngons.append(dict(side_counts))
        #print(len(polys), "\n")
    print(f"Polygons after cut:{polys}")
    


    # Save num_ngons to a file
    with open(f"./neural_data/ngon_counts_TESTING.json", "w") as f:
        json.dump(num_ngons, f, indent=2)

    # Build vertex-to-polygons mapping with colinearity check
    irregular_count=0
    regular_count=0
    vertex_to_polys = {}
    for poly_idx, poly in enumerate(polys):
        n = len(poly)
        for i, vert in enumerate(poly):
            vert_tuple = Vertex(vert[0], vert[1]).as_tuple()
            vert_key = str(vert_tuple)  # Convert tuple to string for JSON

            # Check colinearity with previous and next vertex
            prev_vert = poly[i - 1]
            next_vert = poly[(i + 1) % n]
            x0, y0 = prev_vert
            x1, y1 = vert
            x2, y2 = next_vert
            # Compute cross product to check colinearity
            cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            print(f"Cross for {prev_vert, vert, next_vert}: {cross}")
            is_irregular = 1 if np.isclose(cross, 0, atol=1e-4) else 0

            if is_irregular:
                irregular_count=irregular_count+1
            else:
                regular_count=regular_count+1
            if vert_key not in vertex_to_polys:
                vertex_to_polys[vert_key] = []
            vertex_to_polys[vert_key].append((poly_idx, is_irregular))

    '''
        N_F= summation from i=1 to F of v_i
        N_V= summation from j=1 to V of n_j
        N= (N_F + N_V)/2
        N*= (N*_F + N*_V)/2
        in F, v_i is the number of total vertices (combinatorial degree)
        v*_i is the number of regular vertices (corner degree)
        so N_F = the total number of nodes
        and N_V= the total combinatorial degree of all vertices
        N*_F= the total number of regular nodes
        N*_V= the corner degree of all vertices
        so N=(nodes + combinatorial degree of vertices)/2
           N*=(regular nodes + corner degree of all vertices)/2
        
           x=V(number of nodes)/N*
           y=F(number of polys)/N*
    '''


    print(f"Avg. Number of sides: {regular_count/len(polys)}")
    print(f"Number of irreguar vertices: {irregular_count}")
    print(f"Number of regular vertices: {regular_count}")
    print(f"Recip. of Average nodal corner degree (x): {(irregular_count+regular_count)/regular_count}")
    print(f"Recip. of Average cell corner degree (y): {(len(polys))/regular_count}")
    # Save to JSON
    with open(f"./vertex_to_polys_{k}.json", "w") as f:
        json.dump(vertex_to_polys, f, indent=2)

    # These statics don't work currently after switching to having both regular and irregular vertices, will fix soon
    listPolySizes=[]
    listPolyAreas=[]
    vBarLines=[]
    for poly in polys:
        totalverts+=len(poly)
        listPolySizes.append(len(poly))
        listPolyAreas.append(areaOfPoly(poly))
        vBarLines.append(totalverts/len(listPolySizes))    
    print("Total number of polygons: ", len(polys))
    indeces=[]
    for i in range(0,len(listPolyAreas)):
        indeces.append(i)

    poly_vertex_stats = []
    for poly in polys:
        n = len(poly)
        regular = 0
        irregular = 0
        for i, vert in enumerate(poly):
            prev_vert = poly[i - 1]
            next_vert = poly[(i + 1) % n]
            x0, y0 = prev_vert
            x1, y1 = vert
            x2, y2 = next_vert
            cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            is_irregular = 1 if np.isclose(cross, 0, atol=1e-4) else 0
            if is_irregular:
                irregular += 1
            else:
                regular += 1
        poly_vertex_stats.append({'regular': regular, 'irregular': irregular})

    print("Regular/Irregular vertex counts per polygon:", poly_vertex_stats)



    plt.savefig('Current_Cut.png')
    fig, ax = plt.subplots(1,3, figsize=(20,5))
    ax[0].set_aspect('auto')
    ax[1].set_aspect('auto')
    ax[2].set_aspect('auto')
    # Plotting data
    ax[0].hist(listPolySizes)
    ax[0].set_xlabel("Number of Sides")
    ax[0].set_ylabel("Number of Polygons")
    ax[1].hist(listPolyAreas)
    ax[1].set_xlabel("Area of Polygons")
    ax[1].set_ylabel("Number of Polygons")
    ax[2].plot(indeces,vBarLines)
    ax[2].set_xlabel("Number of Polygons")
    ax[2].set_ylabel("Avg. Number of Sides")
    plt.savefig('Data_For_Current_Cut.png')
    plt.show()




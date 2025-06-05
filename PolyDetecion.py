import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# Matplotlib setup
fig, ax = plt.subplots(1)
square = patches.Rectangle((0.0, 0.0), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(square)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

minPointDiff=0.001
minPointDiffSq=minPointDiff*minPointDiff

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
    def sub(self, other):
        self.x-=other.x
        self.y-=other.y
        return self
    def add(self, other):
        self.x+=other.x
        self.y+=other.y
        return self
    def mul(self, other):
        self.x*=other.x
        self.y*=other.y
        return self
    def div(self, other):
        self.x/=other.x
        self.y/=other.y
        return self
    def getSquaredLength(self):
        return self.x*self.x+self.y*self.y
    def getSquaredDistance(self, other):
        return self.sub(other).getSquaredLength
    @staticmethod
    def getLineFromPoints(p, q):
        a = p.y - q.y
        b = q.x - p.x
        c = (p.x - q.x) * p.y + (q.y - p.y) * p.x
        return [a, b, c]
    @staticmethod
    def lineDist(a, b, c, p):
        if (a + b == 0.0):
            return float('inf')
        return abs(a * p.x + b * p.y + c) / np.sqrt(a * a + b * b)
    @staticmethod
    def lineDistPoints(a,b,p):
        params=Vertex.getLineFromPoints(a, b)
        return Vertex.lineDist(params[0],params[1],params[2], p)

#---------------------------------------------------------------------------

# Corners of the square
corners = [
    Vertex(0.0, 0.0),
    Vertex(1.0, 0.0),
    Vertex(1.0, 1.0),
    Vertex(0.0, 1.0)
]


# ------------------------------- Helper Functions -----------------------------

def doIntersect(p1, q1, p2, q2):
    o1=orientation(p1, q1, p2)
    o2=orientation(p1, q1, q2)
    o3=orientation(p2, q2, p1)
    o4=orientation(p2, q2, q1)

    #General case
    if(o1!=o2 and o3 != o4):
        return True
    
    #Special cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 and onSegment(p1, p2, q1)): return True

    # p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 and onSegment(p1, q2, q1)): return True

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 and onSegment(p2, p1, q2)): return True

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 and onSegment(p2, q1, q2)): return True

    return False # Doesn't fall in any of the above cases

''' 
 To find orientation of ordered triplet (p, q, r).
 The function returns following values
 0 --> p, q and r are colinear
 1 --> Clockwise
 2 --> Counterclockwise
'''

def orientation(p, q, r):
    val= (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if(val==0.0): return 0
    if(val>0):return 1
    return 2

'''
 Given three colinear points p, q, r, the function checks if
 point q lies on line segment 'pr'
'''

def onSegment(p, q, r):
    if (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
        q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
        return True

    return False

def collinearVecs(p, q, r):
    return orientation(p, q, r) == 0

'''
* @return true is this point is betwen a and b
* @note c must be collinear with a and b
* @see O'Rourke, Joseph, "Computational Geometry in C, 2nd Ed.", pp.32
*/
'''

def between(p, a, b):
    if (not collinearVecs(p, a, b)):
        return False
    _x = p.x
    _y = p.y
    return ((a.x <= _x and _x <= b.x) and (a.y <= _y and _y <= b.y)) or ((b.x <= _x and _x <= a.x) and (b.y <= _y and _y <= a.y))

def pointsDiffer(a, b, approx):
    if (approx):
        return a.getSquaredDistance(b) >= minPointDiffSq
    return a.x != b.x or a.y != b.y

def overlap(l1, l2):
     return (collinearVecs(l1.a, l2.a, l2.b) and collinearVecs(l1.b, l2.a, l2.b)) and ((l1.contains(l2.a) or l1.contains(l2.b)) or (l2.contains(l1.a) or l2.contains(l1.b)))

def simplifiedLine(line_1, line_2):
    if (overlap(line_1, line_2)):
        if (line_1.contains(line_2)):
            ret = line_1
            return 1
        if (line_2.contains(line_1)):
            ret = line_2
            return 2
        new_line_start_point=None
        new_line_end_point=None

        # detects which point of <line_1> must be removed
        if (between(line_1.a, line_2.a, line_2.b)):
            new_line_start_point = line_1.b
        else:
            new_line_start_point = line_1.a
        # detects which point of <line_2> must be removed
        if (between(line_2.a, line_1.a, line_1.b)):
            new_line_end_point = line_2.b
        else:
            new_line_end_point = line_2.a
        # create a new line
        ret = Line(new_line_start_point, new_line_end_point)
        return 3
    return 0


def iComparePointOrder(p1, p2):
    if (p1.y < p2.y):
        return -1
    elif (p1.y == p2.y):
        if (p1.x < p2.x):
            return -1
        elif (p1.x == p2.x):
            return 0
    # p1 is greater than p2
    return 1

def bComparePointOrder(p1, p2): return iComparePointOrder(p1, p2) < 0

#---------------------------------------------------------------------------




# --------------------------------- Line Class ---------------------------------

digits = 4

class Line:
    _id_counter = 0  # class variable for unique IDs

    def __init__(self, v1=None, v2=None, id=None):
        if v1 is not None and v2 is not None:
            self.v1 = v1
            self.v2 = v2
        else:
            if random.randint(1, 10) % 2 == 0:
                y1 = np.round(random.random(), digits)
                y2 = np.round(random.random(), digits)
                self.v1 = Vertex(0, y1)
                self.v2 = Vertex(1, y2)
            else:
                x1 = np.round(random.random(), digits)
                x2 = np.round(random.random(), digits)
                self.v1 = Vertex(x1, 0)
                self.v2 = Vertex(x2, 1)
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
            print("Intersection added")
            return Vertex(np.round(px, digits), np.round(py, digits))
        return None

# ------------------------------------------------------------------------------



class Poly_Cycle:
    def __init__(self, set, start, end, closed, fine):
        self.Cycle_Set=set
        self.start_idx=start
        self.end_idx=end
        self.isClosed=closed
        self.fine=fine

PolyCycles=[]

class Polygon:
    def __init__(self):
        pass

class Polygon_Detector:
    def __init__(self):
        pass


def create_lines(n):
    lines = []
    for idx in range(n):
        line = Line()
        print(f"Line {idx+1}: {line}")
        lines.append(line)
    return lines

# Add random lines and border lines
lines = create_lines(2)
border_lines = [
    Line(corners[0], corners[1]),
    Line(corners[1], corners[2]),
    Line(corners[2], corners[3]),
    Line(corners[3], corners[0])
]
lines += border_lines

# Find all intersections
intersections = {i: set([lines[i].v1, lines[i].v2]) for i in range(len(lines))}
for i in range(len(lines)):
    for j in range(i+1, len(lines)):
        pt = lines[i].intersect(lines[j])
        if pt:
            intersections[i].add(pt)
            intersections[j].add(pt)

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

new_lines = []
for idx, line in enumerate(lines):
    pts = list(intersections[idx])
    segments = split_line_at_points(line, pts)
    new_lines.extend(segments)

# Plot all segments
for line in new_lines:
    plt.plot(line.x, line.y, color='b')



# Print all vertices (optional)
all_vertices = set()
for line in new_lines:
    all_vertices.add(line.v1)
    all_vertices.add(line.v2)
for vertex in all_vertices:
    print(f"Vertex: ({vertex.x},{vertex.y})")

for line in new_lines:
    print(line)
    

plt.show()





# NEXT-

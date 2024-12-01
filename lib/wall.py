import math
import matplotlib.patches as mpatches
import matplotlib.path as mpath

codes = [
    mpath.Path.MOVETO,  # Move to the start point
    mpath.Path.CURVE3,  # Cubic Bezier curve
    mpath.Path.CURVE3,
]


class wall:
    def __init__(self, Start: list, End: list, Curve: list) -> None:
        self.A = Start
        self.B = End
        self.D = Curve
        # Bezier Curve Points (in absolute positions)
    
    def createPatch(self, ax):
        path = mpath.Path([self.A, self.D, self.B], codes)
        patch = mpatches.PathPatch(path, facecolor='none', lw=2)
        ax.add_patch(patch)

    def getPoint(self, u: float) -> list:
        newX = ((u*u) - 2*u + 1) * self.A[0] + (2*(1-u)*u) * self.D[0] + (u**2)*self.B[0]
        newY = ((u*u) - 2*u + 1) * self.A[1] + (2*(1-u)*u) * self.D[1] + (u**2)*self.B[1]
        return [newX, newY]
    
    def getTangent(self, u: float) -> list:
        newX = (2*u-2) * self.A[0] + (2-4*u) * self.D[0] + (2*u)*self.B[0]
        newY = (2*u-2) * self.A[1] + (2-4*u) * self.D[1] + (2*u)*self.B[1]
        return [newX, newY]
    #May be very incorrect.

    def findU(self, P: list, V: list) -> float:
        
        g = V[0] / V[1]
	
        a = (self.A[0]) - (2 * self.D[0]) + (self.B[0]) - (g * self.A[1]) + (2 * g * self.D[1]) - (g * self.B[1])
        b = (-2 * self.A[0]) + (2 * self.D[0]) + (2 * g * self.A[1]) - (2 * g * self.D[1])
        c = self.A[0] - P[0] - (g * self.A[1]) + (g * P[1])
		
        discriminant = (b*b) - (4 * a * c)
        
        if (discriminant < 0):
           return -1

        u1 = (-b - math.sqrt(discriminant))/(2 * a)
        if (u1 > 1 or u1 < 0):
            return (-b + math.sqrt(discriminant))/(2 * a)
        
        return u1
    
    def isValidIntersect(self, u, p, v):
        x, y = self.getPoint(u)
        dot =  (x - p[0]) * v[0] + (y - p[1]) * v[1]

        return (u <= 1 and u >= 0 and dot > 0)
    
    def getIntersect(self, P: list, V: list) -> tuple:
        #returns the actual position of intersection
        u1 = self.findU(P,V)

        if self.isValidIntersect(u1, P, V):
            return self.getPoint(u1), self.getTangentVelocity(P, V, u1)
        
        return None, None

    def getTangentVelocity(self, P: list, V: list, u : float):
        #no clue
        return V

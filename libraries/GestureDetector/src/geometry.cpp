#include "Arduino.h"
#include "GestureDetector.h"

// Geometry primitives used in angle calculations.

// Length of vector betwen two points
float length(int x1, int x2, int y1, int y2)
{
	return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1 ) * (y2 - y1));
}

// Dot product of two vectors
float dot(int u1, int u2, int v1, int v2)
{
	return (u1 * u2) + (v1 * v2);
}

// Perp product of two vectors
float perp(int u1, int u2, int v1, int v2)
{
	return (-v1 * u2) + (u1 * v2);

}

// Helpers for point-in-polygon test.
// From Sunday, "Inclusion of a point in a polygon" http://geomalgorithms.com/a03-_inclusion.html
// isLeft(): tests if a point is Left|On|Right of an infinite line.
//    Input:  three points P0, P1, and "this"
//    Return: >0 for P2 left of the line through P0 and P1
//            =0 for P2  on the line
//            <0 for P2  right of the line
//    See: Algorithm 1 "Area of Triangles and Polygons"
int Point::is_left(Point P0, Point P1)
{
    return ((P1.x - P0.x) * (this->y - P0.y)
            - (this->x - P0.x) * (P1.y - P0.y));
}

// Find if a point is in a polygon.
// From Sunday, "Inclusion of a point in a polygon" http://geomalgorithms.com/a03-_inclusion.html
// Winding number test for a point in a polygon
//      Input:   a point,
//               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
//      Return:  wn = the winding number (=0 only when P is outside)
int Point::in_polygon(Point* V, int n)
{
    int    wn = 0;    // the  winding number counter
    int i;

    // loop through all edges of the polygon
    for (i = 0; i < n; i++)
    {   // edge from V[i] to  V[i+1]
        if (V[i].y <= this->y)
        {          // start y <= P.y
            if (V[i + 1].y  > this->y)      // an upward crossing
            if (this->is_left(V[i], V[i + 1]) > 0)  // P left of  edge
                ++wn;            // have  a valid up intersect
        }
        else
        {                        // start y > P.y (no test needed)
            if (V[i + 1].y <= this->y)     // a downward crossing
            if (this->is_left(V[i], V[i + 1]) < 0)  // P right of  edge
                --wn;            // have  a valid down intersect
        }
    }
    return wn;
}

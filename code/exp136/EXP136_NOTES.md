# Minimum enclosing triangle uniqueness

The exact convex hull, in counterclockwise order, is \((0,1/8),(3/28,0),(7/8,0),(25/28,3/28),(1/8,7/8),(0,25/28)\).

Use the midpoint-contact and flush-side characterization of minimum enclosing triangles. For each hull edge, use an area-preserving affine frame with that edge on \(y=0\) and the hull above it. The other two side midpoints have the same height \(h>0\). Their abscissae \(l,r\) are the endpoints of the horizontal hull section. Indeed, the enclosure's section at half the apex height runs between precisely these two midpoints, both in the hull.

If \(h\) is not a hull vertex height, each midpoint lies strictly inside a hull edge. Its supporting line must be that edge's line. Enumerate every triple of hull edge lines containing the chosen base. Retain nondegenerate enclosing triangles with all three midpoints in the hull.

At each positive hull vertex height, compute \(l,r\) by rational linear interpolation. For \(l<r\), the triangle with base midpoint \((z,0)\) has vertices \((z+l-r,0),(z+r-l,0),(l+r-z,2h)\). The base contact confines \(z\) to the base edge. At a hull vertex \((x,y)\), require \(0\le y\le 2h\) and constrain \(x\) between the left and right sides evaluated at \(y\). For fixed \(h,l,r\) these are linear inequalities in \(z\), so intersect their exact rational intervals. Every triangle in a nonempty interval has area \(2h(r-l)\). Keep track of intervals of positive length so a continuum at the minimum cannot pass unnoticed.

The nine retained entries, counted with repeated representations, have areas \(1/2\) and \(43/32\). Every entry of area \(1/2\) is the reference triangle, with singleton parameter intervals where applicable. Thus the unique minimum enclosure is the reference triangle. This enumeration uses the classical characterization, which is cited in the paper.

If an optimal configuration in the reference triangle is sent to the certified configuration by the normalization map \(A\), the common optimum area and positivity give \(|\det A|=1\). The triangle \(A(\Omega)\) encloses the certified configuration and has area \(1/2\), so uniqueness gives \(A(\Omega)=\Omega\). Consequently \(A\) is a vertex permutation of the reference triangle. The same argument applies after composing with a symmetry to identify the normalized image with the listed configuration.

Reproduce with `python -B experiments/136-tri9-enclosure-uniqueness/check.py`. The enumeration establishes enclosure uniqueness independent of the global search. The maximizer classification additionally uses the global covered-search claim.

"""Rebuild and check the triangle n=9 first-order basin using exact fractions."""

from fractions import Fraction as F
from itertools import combinations, permutations, product
import json
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/"127-triangle-n9/cert/local_certs_tri_n9.json"
sys.path.insert(0,str(HERE.parent/"152-geometry-trace"))
from verify_trace import determinant, require, vertices


def inverse(rows):
    n = len(rows)
    aug = [list(map(F,row))+[F(i==j) for j in range(n)] for i,row in enumerate(rows)]
    for j in range(n):
        pivot = next((i for i in range(j,n) if aug[i][j]),None)
        require(pivot is not None,"loaded Jacobian is singular")
        aug[j],aug[pivot] = aug[pivot],aug[j]
        aug[j] = [v/aug[j][j] for v in aug[j]]
        for i in range(n):
            if i!=j and aug[i][j]:
                factor = aug[i][j]
                aug[i] = [a-factor*b for a,b in zip(aug[i],aug[j])]
    result = [row[n:] for row in aug]
    require(all(sum(rows[i][k]*result[k][j] for k in range(n))==int(i==j)
                for i,j in product(range(n),repeat=2)),"inverse identity failed")
    return result


def verify(points, optimum, radius):
    require(len(points)==9 and optimum>0 and radius>0,"invalid basin parameters")
    require(all(x>=0 and y>=0 and x+y<=1 for x,y in points),"infeasible centre")
    loaded = []
    signed_areas = {}
    tie_rows = []
    for i,j,k in combinations(range(9),3):
        area = determinant(points[i],points[j],points[k])
        signed_areas[i,j,k] = area
        if abs(area)!=2*optimum:
            continue
        sign = 1 if area>0 else -1
        row = [F(0)]*18
        for a,b,c in ((i,j,k),(j,k,i),(k,i,j)):
            row[2*a] = sign*(points[b][1]-points[c][1])
            row[2*a+1] = sign*(points[c][0]-points[b][0])
        tie_rows.append(row)
        loaded.append(row)
    require(min(map(abs,signed_areas.values()))==2*optimum,"centre objective mismatch")
    wall_count = 0
    for i,(x,y) in enumerate(points):
        for slack, normal in ((x,(1,0)),(y,(0,1)),(1-x-y,(-1,-1))):
            if slack==0:
                row = [F(0)]*18
                row[2*i:2*i+2] = normal
                wall_count += 1
                loaded.append(row)
    require(len(tie_rows)==13 and wall_count==6,"unexpected active constraints")
    bordered = inverse([row+[1] for row in loaded])
    stress = bordered[-1]
    require(min(stress)>0 and sum(stress)==1,"no positive normalized stress")
    require(all(sum(w*row[j] for w,row in zip(stress,loaded))==0 for j in range(18)),
            "stationarity failed")

    # The vertices of {d: Jd >= -1} have d = B[:, j] / stress[j],
    # where B comprises the first 18 rows of the bordered inverse.
    max_displacement = max(abs(row[j])/stress[j] for row in bordered[:-1] for j in range(19))
    alpha = 1/max_displacement
    # A triangle inside the displacement square has doubled area at most 4r².
    require(4*radius<alpha,"quadratic remainder can exceed coercivity")
    max_gradient = max(sum(map(abs,row)) for row in tie_rows)
    require(max_gradient*radius+4*radius**2<2*optimum,"tie orientation may change")
    return dict(alpha=alpha,max_gradient=max_gradient,
                min_stress=min(stress),quadratic_radius_limit=alpha/4)


def verify_cover(box, q, points, radius, barycentric_permutation, point_assignment):
    require(len(box)==len(points),"cover requires a full local box")
    require(sorted(barycentric_permutation)==[0,1,2],"invalid simplex symmetry")
    require(sorted(point_assignment)==list(range(len(points))),"invalid point assignment")
    for domain, target in zip(box,point_assignment):
        tx,ty = points[target]
        for x,y in vertices(domain,q):
            barycentric = (q-x-y,x,y)
            px,py = (F(barycentric[i],q) for i in barycentric_permutation[1:])
            require(max(abs(px-tx),abs(py-ty))<=radius,"point outside the basin")


def load():
    data = json.loads(SOURCE.read_text())["9"][0]
    points = tuple(tuple(map(F,row)) for row in data["points_exact"])
    optimum = F(data["t_exact"])
    production = F(data["extra"]["basin_radius_exact"])*F(999999,1000000)-F(data["extra"]["box_rho_exact"])
    return points,optimum,production


def main():
    points,optimum,production = load()
    start = perf_counter()
    result = verify(points,optimum,production)
    radius = result["quadratic_radius_limit"]*F(9999,10000)
    try:
        verify(points,optimum,radius)
    except ValueError:
        # Retain a simple rational radius with strictly fixed orientations.
        radius = min(radius, optimum/(2*result["max_gradient"]))
        verify(points,optimum,radius)
    elapsed = perf_counter()-start
    q = 37*2**14
    # Genuine grid coordinates in a permuted candidate image, with nonzero widths.
    perm = (1,0,2)
    order = list(reversed(range(9)))
    transformed = [(1-points[i][0]-points[i][1],points[i][1]) for i in order]
    def interval(value):
        lo = int(q*value)
        return max(0,lo-1),min(q,lo+2)
    box = [("R",*interval(x),*interval(y)) for x,y in transformed]
    verify_cover(box,q,points,production,perm,order)
    print(json.dumps(dict(optimum=str(optimum),production_radius=str(production),
                          improved_radius=str(radius),verification_seconds=elapsed,
                          **{k:str(v) for k,v in result.items()}),indent=2))


if __name__=="__main__":
    main()

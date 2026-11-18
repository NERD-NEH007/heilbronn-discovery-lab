"""Replay cover records against a verified basin and a pinned statement."""

from hashlib import sha256
import json
from pathlib import Path
import sys

from verify_basin import F, SOURCE, load, verify as verify_basin, verify_cover
from verify_trace import keys, read_record, require, verify as verify_tree


def verify(lines, expected):
    stream = iter(lines)
    header = read_record(next(stream,""))
    keys(header,"op version n Q cutoff_raw_t root optimum_raw_t basin_sha256 cover_radius")
    require(header==expected,"trace statement mismatch")
    require(header["op"]=="HEADER" and header["version"]==2 and header["n"]==9,"invalid cover header")
    require(sha256(SOURCE.read_bytes()).hexdigest()==header["basin_sha256"],"basin artifact mismatch")
    points,optimum,_ = load()
    radius = F(header["cover_radius"])
    require(F(header["optimum_raw_t"])==optimum and 0<F(header["cutoff_raw_t"])<optimum,
            "invalid pair of proof thresholds")
    verify_basin(points,optimum,radius)
    base = {key:header[key] for key in ("op","n","Q","cutoff_raw_t","root")}
    events = []

    def translated():
        yield json.dumps(dict(base,version=1))
        for line in stream:
            record = read_record(line)
            if record["op"]=="CLOSE_COVER":
                keys(record,"op barycentric_permutation point_assignment")
                events.append(record)
                yield '{"op":"FRONTIER_YIELD"}'
            else:
                if record["op"]=="FRONTIER_YIELD":
                    events.append(None)
                yield line

    result = verify_tree(translated())
    unresolved = []
    covers = 0
    for event, box in zip(events,result["frontier"],strict=True):
        if event is None:
            unresolved.append(box)
            continue
        for key in ("barycentric_permutation","point_assignment"):
            value = event[key]
            require(type(value) is list and all(type(i) is int for i in value),"invalid cover map")
        verify_cover(box,header["Q"],points,radius,event["barycentric_permutation"],
                     event["point_assignment"])
        covers += 1
    return dict(complete=not unresolved,frontier=unresolved,cover_closures=covers,
                ordinary_closures=result["closures"],root_upper_bound=None if unresolved else str(optimum),
                equality_requires_certified_orbit=not unresolved)


if __name__=="__main__":
    statement = read_record(Path(sys.argv[2]).read_text(encoding="utf-8"))
    with Path(sys.argv[1]).open(encoding="utf-8") as trace:
        print(json.dumps(verify(trace,statement)))

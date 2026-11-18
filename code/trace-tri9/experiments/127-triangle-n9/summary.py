"""summarise results/*.json:  python summary.py [prefix]"""
import json, glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
pre = sys.argv[1] if len(sys.argv) > 1 else ''
for f in sorted(glob.glob(os.path.join(HERE, 'results', pre + '*.json'))):
    d = json.load(open(f)); s = d['stats']
    print(f"{os.path.basename(f)[:-5]:28s} unres={len(d['unresolved']):3d} leaves={s['outer_leaves']:7d} cliques={s['cliques']:10d} "
          f"local={s['local_nodes']:11d} (+fail {s['fail_nodes']}) P2={s['P2']:9d} cover={s['cover']:5d} "
          f"core-h={s['secs'] / 3600:7.3f} wall={s['wall'] / 60:6.1f}min")

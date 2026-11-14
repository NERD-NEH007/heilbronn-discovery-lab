"""Write MANIFEST-n9.sha256: SHA-256 of every input of the n = 9 proof (code, certificate, run logs, paper).
Run from the repository root:  python code/make_manifest.py      Check:  sha256sum -c MANIFEST-n9.sha256"""
import hashlib, glob, os
FILES = (sorted(glob.glob('code/exp110/*.py')) + sorted(glob.glob('code/exp110/clean_room/*.py'))
         + sorted(glob.glob('certificates/exp110/p2dump/*')) + ['code/exp110/README.md', 'code/exp103/bb.py',
         'code/exp44/certify_interval.py', 'certificates/exp103/local_certs.json']
         + sorted(glob.glob('certificates/exp110/results/run_n9/*'))
         + ['papers/heilbronn_n9.tex', 'papers/heilbronn_n9.pdf', 'requirements.txt'])
with open('MANIFEST-n9.sha256', 'w', newline='\n') as f:
    for p in FILES:
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        f.write(f'{h}  {p.replace(os.sep, "/")}\n')
print(f'{len(FILES)} files hashed -> MANIFEST-n9.sha256')

# triangle n = 8 (exp 113)
import glob as _g
TRI = (sorted(_g.glob('code/exp113/cert/*')) + sorted(_g.glob('code/exp113/bb/*.py')) + sorted(_g.glob('certificates/exp113/run_n8/*'))
       + sorted(_g.glob('configurations/triangle/*')) + ['papers/heilbronn_tri8.tex', 'papers/heilbronn_tri8.pdf', 'requirements.txt'])
with open('MANIFEST-tri8.sha256', 'w', newline='
') as f:
    for p in TRI:
        f.write(f'{hashlib.sha256(open(p, "rb").read()).hexdigest()}  {p.replace(os.sep, "/")}
')
print(f'{len(TRI)} files hashed -> MANIFEST-tri8.sha256')

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
TRI = (sorted(glob.glob('code/exp113/cert/*')) + sorted(glob.glob('code/exp113/bb/*.py')) + sorted(glob.glob('certificates/exp113/run_n8/*'))
       + sorted(glob.glob('configurations/triangle/*')) + ['papers/heilbronn_tri8.tex', 'papers/heilbronn_tri8.pdf', 'requirements.txt'])
with open('MANIFEST-tri8.sha256', 'w', newline=chr(10)) as f:
    for p in TRI:
        f.write(hashlib.sha256(open(p, 'rb').read()).hexdigest() + '  ' + p.replace(os.sep, '/') + chr(10))
print(len(TRI), 'files hashed -> MANIFEST-tri8.sha256')

# triangle n = 9 (exp 127, 136)
TRI9 = (sorted(glob.glob('code/exp127/*.py')) + ['code/exp127/batch.sh'] + sorted(glob.glob('code/exp127/cert/*'))
        + sorted(glob.glob('code/exp136/*.py')) + sorted(glob.glob('certificates/exp127/results/*'))
        + ['configurations/triangle/pool_tri_n9.json', 'papers/heilbronn_tri9.tex', 'papers/heilbronn_tri9.pdf', 'requirements.txt'])
with open('MANIFEST-tri9.sha256', 'w', newline=chr(10)) as f:
    for p in TRI9:
        f.write(hashlib.sha256(open(p, 'rb').read()).hexdigest() + '  ' + p.replace(os.sep, '/') + chr(10))
print(len(TRI9), 'files hashed -> MANIFEST-tri9.sha256')

# exp 110: faster branch and bound and the n = 9 run

Run every command from the repository root. `bb2.py` imports the unchanged exp 103 `code/exp103/bb.py` and reads
`certificates/exp103/local_certs.json`. Run state and logs are written to `output/exp110/run_<tag>/`
(the logs of our runs are in `certificates/exp110/results/`).

```
python code/exp110/test_sound2.py 7                      # soundness tests (against the n = 7 optimum)
python code/exp110/check_closed_form.py                  # n = 9: closed form (9 sqrt 65 - 55)/320 (paper 3)
python code/exp110/run_par2.py 7 exact --tag n7          # about 10 minutes
python code/exp110/run_par2.py 8 exact --tag n8          # about 9 minutes
python code/exp110/run_par2.py 9 exact --skip 1.2 --audit 0.0002 --tag n9 --est-nodes 3e8   # about 16 hours on 14 workers
```

Use `--workers W` to set the number of worker processes. A stopped run resumes from `state.pkl` when the same command is
repeated. `probe2.py` and `run_small2.py` are the size probes and small validation runs described in
`certificates/exp110/EXP110_NOTES.md`. Requires numpy, numba, highspy, scipy and sympy.

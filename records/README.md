# records/

Put published record configurations here. They are not redistributed in this repository.

1. Open https://math.tejstead.com/heilbronn, choose the unit square and size n (19 <= n <= 36), and download the
   coordinate file.
2. Save it as `records/n{n}.txt`, for example `records/n22.txt`. Any text file works if its data rows end with `x y`
   (separated by spaces, tabs or commas). Lines starting with `#` are comments.
3. Check it: `python code/verify_record_file.py records/n22.txt`, or `--all` for every file here.

The papers use the records as published on 12 November 2026. `configurations/record_values.json` has their values. A
record improved since then shows a different value. The scripts also read the directory named in the environment
variable `HEILBRONN_RECORDS`, if it is set.

Credit: T. Stead (maintainer) and the contributors listed on the site.

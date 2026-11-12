# Test fixtures

Small made-up files in the real formats of the persona sources (decision
0020), laid out as in the data folder, for the importers' tests. None of it
is real data from either source: the Neuron Persona Library states no
licence, so none of its data goes into git, fixtures included. Every name,
tag, text and neuron in them is invented, none taken from the library's
files.

`library/qwen/` holds the library's three files for Qwen3-8B, `library.json`,
`combos.json` and `sensitive.json`, with invented rows (personas such as
Teapot and Snowfield Clerk) in exactly the library's format, and the
`FETCHED.json` record that `pushlab fetch library` writes beside them. Two
things follow the files, and `tests/unit/test_library_import.py` checks
both: the `"from"` stamp of `sensitive.json` gives the sizes in bytes of
`library.json` and `combos.json` (`import_library` refuses a stamp that
disagrees with the files), and `FETCHED.json` gives each file's `bytes` and
`sha256`. So if you change `library.json` or `combos.json`, update the stamp
first, then the record.

Mayner's data, parquet tables and tensor files, is made up in code instead,
by the `mayner_data` fixture in `tests/unit/conftest.py`. Its bundle has no
`MANIFEST`, so nothing is checked against one; the tests that need one write
it.

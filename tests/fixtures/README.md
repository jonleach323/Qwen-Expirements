# Test fixtures

Small made-up files in the real formats of the persona sources (decision
0020), laid out as in the data folder, for the importers' tests. None of it
is real data from either source: the Neuron Persona Library states no
licence, so none of its data goes into git, fixtures included.
`library/qwen/` holds the library's three files for Qwen3-8B, `library.json`,
`combos.json` and `sensitive.json`, with invented rows (personas such as
Teapot and Glacier) in exactly the library's format, and the `FETCHED.json`
record that `pushlab fetch library` writes beside them; if you change one of
the three, update its `bytes` and `sha256` there, which
`tests/unit/test_library_import.py` checks. Mayner's data, parquet tables and
tensor files, is made up in code instead, by the `mayner_data` fixture in
`tests/unit/conftest.py`.

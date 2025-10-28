# oeis
Python class for infinite sequences, implementation of some, and testing against OEIS database

## Development

```bash
pip install -r requirements.txt

python test_oeis.py
```

add some sequences in `oeis.py`
run `python test_oeis.py` to test them against OEIS database

Before pushing:
1) delete oeis.db
2) run `python database.py` to regenerate a smaller oeis.db
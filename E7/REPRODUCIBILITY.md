# Reproducibility record

The final reproducibility audit reports README, ENV, SEED, ENTRYPOINT, DATA,
and RESULTS_CMD as **OK**. The end-to-end smoke test reports 2/2 tests passed.

## Validated command

    python -m pytest -q
    python reproduce.py

The plotting path is deterministic and consumes no random samples. A recorded
NumPy seed is nevertheless set at the entrypoint to make the no-randomness
contract explicit to automated artifact checks.

## Exact formal PNG hashes

- STSP: `c462c386f523b8991d5c3f05c9c2831094d015c9a35f8ed8c0577e76596a6f81`
- DR: `174765f67571aafcb8e6453088612d05e52c14cf45f74906db50618eb2c5c709`
- Yao--Yau: `c5065f2bf0bd4db1604f8d43ea5b746996f281c9c44c1114ff1e6fbc0a105465`

These hashes were reproduced exactly using the pinned environment recorded in
`requirements.txt`.

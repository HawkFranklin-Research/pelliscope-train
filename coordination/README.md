# Two-agent coordination

`jobs.yaml` is the assignment source of truth. Change a job to `running` only after recording its owner, machine, shared Git commit, configuration hash, case-manifest hash, and split-manifest hash. One job has one writer.

Set `HAWK_DERM_AGENT` before running a job so the global run ledger identifies the producing agent. Existing feature banks can be assigned without changing the shared configuration through `HAWK_DERM_IMPORT_BANK_<ENCODER_KEY>`, for example:

```bash
export HAWK_DERM_AGENT=macbook-agent
export HAWK_DERM_IMPORT_BANK_SIGLIP2_SO400M=/absolute/path/all_cases_siglip2_so400m_features.npz
```

Split work by complete encoder or complete seed. Do not split one feature bank across machines. Upload large outputs to the configured Hugging Face repository, record the immutable revision and checksum in `ARTIFACT_INDEX.csv`, and merge only after a second machine verifies the artifact.

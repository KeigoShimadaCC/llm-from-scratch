# PHASE-06A SFT fixed evaluation

## Summary

- Prompt template version: `kgpt_sft_v1_compact`
- Dataset source: phase06a_repo_authored_instruction_smoke_v1
- License: Repo-authored fixture for this project; no third-party instruction data.
- Response-only loss masking: True
- Base response loss: 13.9152
- SFT response loss: 0.4848
- Narrow improvement: yes
- Held-out SFT validation loss: 6.2837 -> 34.7090

## Prompt Format

```text
Q:{instruction}
A:
```

## Fixed Probe Samples

### `say hi`

- Expected: `hi`
- Base: `Q:say hi
A:alalalalalalalal`
- SFT: `Q:say hi
A:hi`
### `say ok`

- Expected: `ok`
- Base: `Q:say ok
A:alalalalalalalal`
- SFT: `Q:say ok
A:ok`
### `repeat cat`

- Expected: `cat`
- Base: `Q:repeat cat
A:########`
- SFT: `Q:repeat cat
A:cat`
### `answer blue`

- Expected: `blue`
- Base: `Q:answer blue
A:mmmmmmmm`
- SFT: `Q:answer blue
A:s`

## Limitations

- This is a tiny repo-authored instruction fixture, not a safety or alignment dataset.
- The model is only expected to improve on narrow command/response probes.
- Held-out validation loss regressed in this smoke run, so the evidence supports memorized narrow commands only.
- Generated text can still repeat or include tokenization artifacts.

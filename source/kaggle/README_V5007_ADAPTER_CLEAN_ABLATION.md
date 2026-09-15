# v0.5007 clean-adapter A/B

Notebook: `legalqa_dsc2026_v5007_adapter_clean_ablation_t4.ipynb`.

This is a controlled public-inference experiment. It retains the v0.5007
corpus, dense index, selector, retrieval, prompts, generation configuration,
and base checkpoints. The only model identity allowed to differ is the full
QLoRA adapter.

## First Kaggle session

Attach only:

1. this source Dataset;
2. the original organizer Dataset containing `public-official.json`,
   `train.json`, and `selected-contexts/`;
3. v0.5007 dense index (`b5c6...`, 215267 chunks);
4. the four approved base checkpoints;
5. the clean adapter A1 extracted as files, not kept only inside a ZIP;
6. the original v0.5007 `tuned_selection.yaml` (`85a00e...`).

The selector SHA-256 is computed after normalizing UTF-8 BOM and CRLF/LF
line endings. Saving the same weights in Windows does not change its identity.

Do not attach a different dense index, another full adapter, `train_clean.json`,
a v4 selector, or any old inference recovery. Set `AUTO_RESUME = False` and
run all cells.

The run must print these identities before expensive retrieval:

```text
adapter: 590a732d4ea601e20b68d27407206265a52fd5753402ae2f9dc3e7084ccd850a
dense:   b5c6eadb46ee7aceab44b8643fa8a5beefde5420c02cf2146e0fd30e38eccf23
source:  97aaa9cabf0385f9d72abf8359910ece5f3e539cd151a65db7225c7159c7f0a1
run:     00345b636cec2c37
```

Any mismatch is intentional failure: do not submit that run.

## Resume session

If the first session reports `PAUSED SAFELY`, save its
`legalqa_recovery_latest.zip` as a Dataset. Attach exactly the same six inputs
above plus this one recovery Dataset. Set `AUTO_RESUME = True`. The recovery
fingerprint must be `00345b636cec2c37`. The ZIP or Dataset may have a Windows
suffix/custom Kaggle slug; discovery uses its internal fingerprint, not its name.

Never resume with a recovery from the original adapter (`de25...`) or from the
newer full-pipeline run (`a090...`).

## Completion

The final cell accepts only a completed manifest with run fingerprint
`00345b636cec2c37` and a ZIP containing exactly one non-empty 1,000-answer
`submission.json`.

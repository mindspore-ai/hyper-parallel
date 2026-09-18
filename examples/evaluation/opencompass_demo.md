# OpenCompass MMLU demo

Status: **single-process MMLU PPL adapter implemented; distributed and generation paths not yet supported**.

The adapter targets OpenCompass `0.5.4` and the official `mmlu_ppl_ac766d` configuration. Install the Torch build that
matches the target accelerator first, then install the evaluation extra in the same environment:

```bash
pip install -e '.[opencompass]'
```

Set a local checkpoint and run the default 32-sample `college_computer_science` smoke subset:

```bash
export HP_CHECKPOINT=/path/to/base-model
export HP_TOKENIZER=/path/to/base-model
opencompass examples/evaluation/opencompass_hyper_mmlu.py \
  --work-dir outputs/hyper-mmlu-smoke --debug
```

Use the source-checkout entry point instead when OpenCompass was installed from source:

```bash
python run.py /path/to/hyper-parallel/examples/evaluation/opencompass_hyper_mmlu.py \
  -w /path/to/hyper-parallel/outputs/hyper-mmlu-smoke --debug
```

Set `HP_MMLU_SUBJECT=all` and `HP_MMLU_SAMPLE_LIMIT=0` for the complete suite. `HP_EVAL_BATCH_SIZE`, `HP_MAX_SEQ_LEN`,
`HP_EVAL_DTYPE`, `HP_TRUST_REMOTE_CODE`, and `HP_OPENCOMPASS_NUM_GPUS` configure the model task. The current adapter rejects
`world_size > 1`; do not schedule one OpenCompass worker per model-parallel rank.

The first native-vs-Hyper comparison must align expanded prompts, tokenization, candidate score masks, and denominators
before comparing labels and accuracy. The adapter returns the same normalized causal cross entropy used by OpenCompass's
HuggingFace PPL backend; it does not substitute answer-token logits.

See `.agent/skills/benchmark-evaluation/references/opencompass-demo.md` for the full validation flow.

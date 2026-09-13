# Training notebook

`meat_freshness_9class_multirun.ipynb` is the exact resumable notebook used for the reported nine-class experiment. It verifies the dataset manifest before training, records checkpoints after completed epochs, runs four architectures with five prespecified seeds, and blocks paper-ready aggregation until all 20 compatible runs are present.

Run it with streamed terminal progress from the repository root:

```bash
python3 scripts/run_notebook_streaming.py --full
```

The default command without `--full` performs only the safe two-epoch pilot.

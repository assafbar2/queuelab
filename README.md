# QueueLab

Can a small language model learn what to check before it tells a customer what went wrong?

QueueLab is a proposed companion to [The Support Machine](https://assafbar2.github.io/the-support-machine/). The plan is to fine-tune a 1.7B-parameter model on synthetic support investigations and compare it with its unchanged base model in the same simulated product.

**Status: specification only.** No adapter has been trained, no benchmark has been run, and no improvement is claimed.

[Read the v1 specification](docs/SPEC.md) for the opening ticket, the training recipe, the evaluation protocol, and the release criteria. The first milestone is a baseline on ten paired tickets, not a GPU training run.

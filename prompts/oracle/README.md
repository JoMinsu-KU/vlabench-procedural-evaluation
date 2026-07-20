# Oracle prompts

The two archives contain the complete model-facing prompts for the Phase-oracle and Target-oracle conditions. Prompt contents were identical across the three backbones within each condition, so one canonical archive per condition is stored here.

Neither condition exposes ground-truth targets in the prompt. Phase-oracle provides the frozen reference phase/order and requests targets. Target-oracle requests a skill sequence and reads reference targets only after the sequence passes the frozen phase contract.

# SecAI model serving

The current target is [4× RTX PRO 6000 Blackwell + 1× RTX 5090](blackwell/README.md):
two TP2 vLLM replicas for the OrcaRouter Qwen3.8 checkpoint, with independent
embedding, reranking and OvisOCR3 services planned for the 5090. Translation
calls the main LLM. The 5090 network route was verified; its models have not
been started yet.

The older A100/Cyber-F1 scripts remain documented in [DEPLOY.md](DEPLOY.md)
and [vast/README.md](vast/README.md). They are separate profiles and should
not be used to start the Blackwell checkpoint.

Current Vast instance state, benchmark data, secret locations, and next steps
are recorded in the [Blackwell handoff](blackwell/HANDOFF.md).

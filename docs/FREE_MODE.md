# Free Mode

The proof-of-capability phase is designed to run with no paid model budget.

## Model

GPT-OSS 120B is the only reasoning model in V1. The application keeps the provider abstraction stable.

## Providers

1. Groq: primary provider through `GROQ_API_KEY`.
2. Hugging Face Inference Providers: fallback/alternate through `HF_TOKEN`.

Hugging Face routes `openai/gpt-oss-120b` through an OpenAI-compatible endpoint and supports provider-selection policies such as `:fastest`, `:cheapest`, or a specific provider.

## Free-tier strategy

Do not send every specialist agent to the LLM. Preprocessors, field checks, retrieval, document parsing, similarity, clone/delta comparison and basic evidence checks run locally. GPT-OSS 120B is reserved for the high-value synthesis/self-critique step.

Use environment variables:

```text
PRE_CAB_FREE_MODE=true
PRE_CAB_PROVIDER_ORDER=groq,huggingface
PRE_CAB_MODEL=openai/gpt-oss-120b
HF_PROVIDER=fastest
```

Never commit tokens. Keep them in environment/secret storage.

## Benchmark rule

The historical CAB outcome must be removed from model-visible input before prediction. Benchmark accuracy, false-pass rate, false-fail rate and requirement/evidence metrics separately.

## Later paid mode

The same contracts can move to higher-throughput inference, private/self-hosted GPT-OSS 120B, stronger vector retrieval and eventual domain fine-tuning without changing the agent or decision interfaces.

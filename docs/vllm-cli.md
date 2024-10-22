# VLLM CLI

## vllm complete

```
$ vllm complete --help
usage: vllm complete [options]

options:
  -h, --help            show this help message and exit
  --url URL             url of the running OpenAI-Compatible RESTful API server
  --model-name MODEL_NAME
                        The model name used in prompt completion, default to the first model in list models API call.
  --api-key API_KEY     API key for OpenAI services. If provided, this api key will overwrite the api key obtained
                        through environment variables.
```

Example

Client side:

```
$ vllm complete
INFO 10-22 12:13:06 importing.py:10] Triton not installed; certain GPU-related functions will not be available.
Using model: gpt2
Please enter prompt to complete:
> Summer is
 ruled in 1994.

There have been some suggestions that the Colt, the
>
```

Server side:

```
INFO:     ::1:36536 - "GET /v1/models HTTP/1.1" 200 OK
INFO 10-22 12:13:15 metrics.py:351] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 10-22 12:13:24 logger.py:36] Received request cmpl-1f94a8f598ea45e386c00f39ef6ab30c-0: prompt: 'Summer is', params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=16, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), prompt_token_ids: [33560, 318], lora_request: None, prompt_adapter_request: None.
INFO 10-22 12:13:24 async_llm_engine.py:205] Added request cmpl-1f94a8f598ea45e386c00f39ef6ab30c-0.
INFO 10-22 12:13:24 metrics.py:351] Avg prompt throughput: 0.2 tokens/s, Avg generation throughput: 0.1 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 10-22 12:13:24 async_llm_engine.py:173] Finished request cmpl-1f94a8f598ea45e386c00f39ef6ab30c-0.
INFO:     ::1:34014 - "POST /v1/completions HTTP/1.1" 200 OK
INFO 10-22 12:13:35 metrics.py:351] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1.3 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
```

## vllm chat

```
$ vllm chat --help
usage: vllm chat [options]

options:
  -h, --help            show this help message and exit
  --url URL             url of the running OpenAI-Compatible RESTful API server
  --model-name MODEL_NAME
                        The model name used in prompt completion, default to the first model in list models API call.
  --api-key API_KEY     API key for OpenAI services. If provided, this api key will overwrite the api key obtained
                        through environment variables.
  --system-prompt SYSTEM_PROMPT
                        The system prompt to be added to the chat template, used for models that support system
                        prompts.
```

## vllm serve

```
$ vllm serve --help
usage: vllm serve <model_tag> [options]

positional arguments:
  model_tag             The model tag to serve

options:
  -h, --help            show this help message and exit
  --config CONFIG       Read CLI options from a config file.Must be a YAML with the following
                        options:https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-
                        arguments-for-the-server
  --host HOST           host name
  --port PORT           port number
  --uvicorn-log-level {debug,info,warning,error,critical,trace}
                        log level for uvicorn
  --allow-credentials   allow credentials
  --allowed-origins ALLOWED_ORIGINS
                        allowed origins
  --allowed-methods ALLOWED_METHODS
                        allowed methods
  --allowed-headers ALLOWED_HEADERS
                        allowed headers
  --api-key API_KEY     If provided, the server will require this key to be presented in the header.
  --lora-modules LORA_MODULES [LORA_MODULES ...]
                        LoRA module configurations in the format name=path. Multiple modules can be specified.
  --prompt-adapters PROMPT_ADAPTERS [PROMPT_ADAPTERS ...]
                        Prompt adapter configurations in the format name=path. Multiple adapters can be specified.
  --chat-template CHAT_TEMPLATE
                        The file path to the chat template, or the template in single-line form for the specified
                        model
  --response-role RESPONSE_ROLE
                        The role name to return if `request.add_generation_prompt=true`.
  --ssl-keyfile SSL_KEYFILE
                        The file path to the SSL key file
  --ssl-certfile SSL_CERTFILE
                        The file path to the SSL cert file
  --ssl-ca-certs SSL_CA_CERTS
                        The CA certificates file
  --ssl-cert-reqs SSL_CERT_REQS
                        Whether client certificate is required (see stdlib ssl module's)
  --root-path ROOT_PATH
                        FastAPI root_path when app is behind a path based routing proxy
  --middleware MIDDLEWARE
                        Additional ASGI middleware to apply to the app. We accept multiple --middleware arguments.
                        The value should be an import path. If a function is provided, vLLM will add it to the server
                        using @app.middleware('http'). If a class is provided, vLLM will add it to the server using
                        app.add_middleware().
  --return-tokens-as-token-ids
                        When --max-logprobs is specified, represents single tokens as strings of the form
                        'token_id:{token_id}' so that tokens that are not JSON-encodable can be identified.
  --disable-frontend-multiprocessing
                        If specified, will run the OpenAI frontend server in the same process as the model serving
                        engine.
  --enable-auto-tool-choice
                        Enable auto tool choice for supported models. Use --tool-call-parserto specify which parser
                        to use
  --tool-call-parser {mistral,hermes}
                        Select the tool call parser depending on the model that you're using. This is used to parse
                        the model-generated tool call into OpenAI API format. Required for --enable-auto-tool-choice.
  --model MODEL         Name or path of the huggingface model to use.
  --tokenizer TOKENIZER
                        Name or path of the huggingface tokenizer to use. If unspecified, model name or path will be
                        used.
  --skip-tokenizer-init
                        Skip initialization of tokenizer and detokenizer
  --revision REVISION   The specific model version to use. It can be a branch name, a tag name, or a commit id. If
                        unspecified, will use the default version.
  --code-revision CODE_REVISION
                        The specific revision to use for the model code on Hugging Face Hub. It can be a branch name,
                        a tag name, or a commit id. If unspecified, will use the default version.
  --tokenizer-revision TOKENIZER_REVISION
                        Revision of the huggingface tokenizer to use. It can be a branch name, a tag name, or a
                        commit id. If unspecified, will use the default version.
  --tokenizer-mode {auto,slow,mistral}
                        The tokenizer mode. * "auto" will use the fast tokenizer if available. * "slow" will always
                        use the slow tokenizer. * "mistral" will always use the `mistral_common` tokenizer.
  --trust-remote-code   Trust remote code from huggingface.
  --download-dir DOWNLOAD_DIR
                        Directory to download and load the weights, default to the default cache dir of huggingface.
  --load-format {auto,pt,safetensors,npcache,dummy,tensorizer,sharded_state,gguf,bitsandbytes,mistral}
                        The format of the model weights to load. * "auto" will try to load the weights in the
                        safetensors format and fall back to the pytorch bin format if safetensors format is not
                        available. * "pt" will load the weights in the pytorch bin format. * "safetensors" will load
                        the weights in the safetensors format. * "npcache" will load the weights in pytorch format
                        and store a numpy cache to speed up the loading. * "dummy" will initialize the weights with
                        random values, which is mainly for profiling. * "tensorizer" will load the weights using
                        tensorizer from CoreWeave. See the Tensorize vLLM Model script in the Examples section for
                        more information. * "bitsandbytes" will load the weights using bitsandbytes quantization.
  --config-format {auto,hf,mistral}
                        The format of the model config to load. * "auto" will try to load the config in hf format if
                        available else it will try to load in mistral format
  --dtype {auto,half,float16,bfloat16,float,float32}
                        Data type for model weights and activations. * "auto" will use FP16 precision for FP32 and
                        FP16 models, and BF16 precision for BF16 models. * "half" for FP16. Recommended for AWQ
                        quantization. * "float16" is the same as "half". * "bfloat16" for a balance between precision
                        and range. * "float" is shorthand for FP32 precision. * "float32" for FP32 precision.
  --kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}
                        Data type for kv cache storage. If "auto", will use model data type. CUDA 11.8+ supports fp8
                        (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports fp8 (=fp8_e4m3)
  --quantization-param-path QUANTIZATION_PARAM_PATH
                        Path to the JSON file containing the KV cache scaling factors. This should generally be
                        supplied, when KV cache dtype is FP8. Otherwise, KV cache scaling factors default to 1.0,
                        which may cause accuracy issues. FP8_E5M2 (without scaling) is only supported on cuda
                        versiongreater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead supported for common
                        inference criteria.
  --max-model-len MAX_MODEL_LEN
                        Model context length. If unspecified, will be automatically derived from the model config.
  --guided-decoding-backend {outlines,lm-format-enforcer}
                        Which engine will be used for guided decoding (JSON schema / regex etc) by default. Currently
                        support https://github.com/outlines-dev/outlines and https://github.com/noamgat/lm-format-
                        enforcer. Can be overridden per request via guided_decoding_backend parameter.
  --distributed-executor-backend {ray,mp}
                        Backend to use for distributed serving. When more than 1 GPU is used, will be automatically
                        set to "ray" if installed or "mp" (multiprocessing) otherwise.
  --worker-use-ray      Deprecated, use --distributed-executor-backend=ray.
  --pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE
                        Number of pipeline stages.
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
                        Number of tensor parallel replicas.
  --max-parallel-loading-workers MAX_PARALLEL_LOADING_WORKERS
                        Load model sequentially in multiple batches, to avoid RAM OOM when using tensor parallel and
                        large models.
  --ray-workers-use-nsight
                        If specified, use nsight to profile Ray workers.
  --block-size {8,16,32}
                        Token block size for contiguous chunks of tokens. This is ignored on neuron devices and set
                        to max-model-len
  --enable-prefix-caching
                        Enables automatic prefix caching.
  --disable-sliding-window
                        Disables sliding window, capping to sliding window size
  --use-v2-block-manager
                        Use BlockSpaceMangerV2.
  --num-lookahead-slots NUM_LOOKAHEAD_SLOTS
                        Experimental scheduling config necessary for speculative decoding. This will be replaced by
                        speculative config in the future; it is present to enable correctness tests until then.
  --seed SEED           Random seed for operations.
  --swap-space SWAP_SPACE
                        CPU swap space size (GiB) per GPU.
  --cpu-offload-gb CPU_OFFLOAD_GB
                        The space in GiB to offload to CPU, per GPU. Default is 0, which means no offloading.
                        Intuitively, this argument can be seen as a virtual way to increase the GPU memory size. For
                        example, if you have one 24 GB GPU and set this to 10, virtually you can think of it as a 34
                        GB GPU. Then you can load a 13B model with BF16 weight,which requires at least 26GB GPU
                        memory. Note that this requires fast CPU-GPU interconnect, as part of the model isloaded from
                        CPU memory to GPU memory on the fly in each model forward pass.
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION
                        The fraction of GPU memory to be used for the model executor, which can range from 0 to 1.
                        For example, a value of 0.5 would imply 50% GPU memory utilization. If unspecified, will use
                        the default value of 0.9.
  --num-gpu-blocks-override NUM_GPU_BLOCKS_OVERRIDE
                        If specified, ignore GPU profiling result and use this numberof GPU blocks. Used for testing
                        preemption.
  --max-num-batched-tokens MAX_NUM_BATCHED_TOKENS
                        Maximum number of batched tokens per iteration.
  --max-num-seqs MAX_NUM_SEQS
                        Maximum number of sequences per iteration.
  --max-logprobs MAX_LOGPROBS
                        Max number of log probs to return logprobs is specified in SamplingParams.
  --disable-log-stats   Disable logging statistics.
  --quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,experts_int8,neuron_quant,None}, -q {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,experts_int8,neuron_quant,None}
                        Method used to quantize the weights. If None, we first check the `quantization_config`
                        attribute in the model config file. If that is None, we assume the model weights are not
                        quantized and use `dtype` to determine the data type of the weights.
  --rope-scaling ROPE_SCALING
                        RoPE scaling configuration in JSON format. For example, {"type":"dynamic","factor":2.0}
  --rope-theta ROPE_THETA
                        RoPE theta. Use with `rope_scaling`. In some cases, changing the RoPE theta improves the
                        performance of the scaled model.
  --enforce-eager       Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for
                        maximal performance and flexibility.
  --max-context-len-to-capture MAX_CONTEXT_LEN_TO_CAPTURE
                        Maximum context length covered by CUDA graphs. When a sequence has context length larger than
                        this, we fall back to eager mode. (DEPRECATED. Use --max-seq-len-to-capture instead)
  --max-seq-len-to-capture MAX_SEQ_LEN_TO_CAPTURE
                        Maximum sequence length covered by CUDA graphs. When a sequence has context length larger
                        than this, we fall back to eager mode.
  --disable-custom-all-reduce
                        See ParallelConfig.
  --tokenizer-pool-size TOKENIZER_POOL_SIZE
                        Size of tokenizer pool to use for asynchronous tokenization. If 0, will use synchronous
                        tokenization.
  --tokenizer-pool-type TOKENIZER_POOL_TYPE
                        Type of tokenizer pool to use for asynchronous tokenization. Ignored if tokenizer_pool_size
                        is 0.
  --tokenizer-pool-extra-config TOKENIZER_POOL_EXTRA_CONFIG
                        Extra config for tokenizer pool. This should be a JSON string that will be parsed into a
                        dictionary. Ignored if tokenizer_pool_size is 0.
  --limit-mm-per-prompt LIMIT_MM_PER_PROMPT
                        For each multimodal plugin, limit how many input instances to allow for each prompt. Expects
                        a comma-separated list of items, e.g.: `image=16,video=2` allows a maximum of 16 images and 2
                        videos per prompt. Defaults to 1 for each modality.
  --enable-lora         If True, enable handling of LoRA adapters.
  --max-loras MAX_LORAS
                        Max number of LoRAs in a single batch.
  --max-lora-rank MAX_LORA_RANK
                        Max LoRA rank.
  --lora-extra-vocab-size LORA_EXTRA_VOCAB_SIZE
                        Maximum size of extra vocabulary that can be present in a LoRA adapter (added to the base
                        model vocabulary).
  --lora-dtype {auto,float16,bfloat16,float32}
                        Data type for LoRA. If auto, will default to base model dtype.
  --long-lora-scaling-factors LONG_LORA_SCALING_FACTORS
                        Specify multiple scaling factors (which can be different from base model scaling factor - see
                        eg. Long LoRA) to allow for multiple LoRA adapters trained with those scaling factors to be
                        used at the same time. If not specified, only adapters trained with the base model scaling
                        factor are allowed.
  --max-cpu-loras MAX_CPU_LORAS
                        Maximum number of LoRAs to store in CPU memory. Must be >= than max_num_seqs. Defaults to
                        max_num_seqs.
  --fully-sharded-loras
                        By default, only half of the LoRA computation is sharded with tensor parallelism. Enabling
                        this will use the fully sharded layers. At high sequence length, max rank or tensor parallel
                        size, this is likely faster.
  --enable-prompt-adapter
                        If True, enable handling of PromptAdapters.
  --max-prompt-adapters MAX_PROMPT_ADAPTERS
                        Max number of PromptAdapters in a batch.
  --max-prompt-adapter-token MAX_PROMPT_ADAPTER_TOKEN
                        Max number of PromptAdapters tokens
  --device {auto,cuda,neuron,cpu,openvino,tpu,xpu}
                        Device type for vLLM execution.
  --num-scheduler-steps NUM_SCHEDULER_STEPS
                        Maximum number of forward steps per scheduler call.
  --scheduler-delay-factor SCHEDULER_DELAY_FACTOR
                        Apply a delay (of delay factor multiplied by previousprompt latency) before scheduling next
                        prompt.
  --enable-chunked-prefill [ENABLE_CHUNKED_PREFILL]
                        If set, the prefill requests can be chunked based on the max_num_batched_tokens.
  --speculative-model SPECULATIVE_MODEL
                        The name of the draft model to be used in speculative decoding.
  --speculative-model-quantization {aqlm,awq,deepspeedfp,tpu_int8,fp8,fbgemm_fp8,modelopt,marlin,gguf,gptq_marlin_24,gptq_marlin,awq_marlin,gptq,compressed-tensors,bitsandbytes,qqq,experts_int8,neuron_quant,None}
                        Method used to quantize the weights of speculative model.If None, we first check the
                        `quantization_config` attribute in the model config file. If that is None, we assume the
                        model weights are not quantized and use `dtype` to determine the data type of the weights.
  --num-speculative-tokens NUM_SPECULATIVE_TOKENS
                        The number of speculative tokens to sample from the draft model in speculative decoding.
  --speculative-draft-tensor-parallel-size SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE, -spec-draft-tp SPECULATIVE_DRAFT_TENSOR_PARALLEL_SIZE
                        Number of tensor parallel replicas for the draft model in speculative decoding.
  --speculative-max-model-len SPECULATIVE_MAX_MODEL_LEN
                        The maximum sequence length supported by the draft model. Sequences over this length will
                        skip speculation.
  --speculative-disable-by-batch-size SPECULATIVE_DISABLE_BY_BATCH_SIZE
                        Disable speculative decoding for new incoming requests if the number of enqueue requests is
                        larger than this value.
  --ngram-prompt-lookup-max NGRAM_PROMPT_LOOKUP_MAX
                        Max size of window for ngram prompt lookup in speculative decoding.
  --ngram-prompt-lookup-min NGRAM_PROMPT_LOOKUP_MIN
                        Min size of window for ngram prompt lookup in speculative decoding.
  --spec-decoding-acceptance-method {rejection_sampler,typical_acceptance_sampler}
                        Specify the acceptance method to use during draft token verification in speculative decoding.
                        Two types of acceptance routines are supported: 1) RejectionSampler which does not allow
                        changing the acceptance rate of draft tokens, 2) TypicalAcceptanceSampler which is
                        configurable, allowing for a higher acceptance rate at the cost of lower quality, and vice
                        versa.
  --typical-acceptance-sampler-posterior-threshold TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_THRESHOLD
                        Set the lower bound threshold for the posterior probability of a token to be accepted. This
                        threshold is used by the TypicalAcceptanceSampler to make sampling decisions during
                        speculative decoding. Defaults to 0.09
  --typical-acceptance-sampler-posterior-alpha TYPICAL_ACCEPTANCE_SAMPLER_POSTERIOR_ALPHA
                        A scaling factor for the entropy-based threshold for token acceptance in the
                        TypicalAcceptanceSampler. Typically defaults to sqrt of --typical-acceptance-sampler-
                        posterior-threshold i.e. 0.3
  --disable-logprobs-during-spec-decoding [DISABLE_LOGPROBS_DURING_SPEC_DECODING]
                        If set to True, token log probabilities are not returned during speculative decoding. If set
                        to False, log probabilities are returned according to the settings in SamplingParams. If not
                        specified, it defaults to True. Disabling log probabilities during speculative decoding
                        reduces latency by skipping logprob calculation in proposal sampling, target sampling, and
                        after accepted tokens are determined.
  --model-loader-extra-config MODEL_LOADER_EXTRA_CONFIG
                        Extra config for model loader. This will be passed to the model loader corresponding to the
                        chosen load_format. This should be a JSON string that will be parsed into a dictionary.
  --ignore-patterns IGNORE_PATTERNS
                        The pattern(s) to ignore when loading the model.Default to 'original/**/*' to avoid repeated
                        loading of llama's checkpoints.
  --preemption-mode PREEMPTION_MODE
                        If 'recompute', the engine performs preemption by recomputing; If 'swap', the engine performs
                        preemption by block swapping.
  --served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]
                        The model name(s) used in the API. If multiple names are provided, the server will respond to
                        any of the provided names. The model name in the model field of a response will be the first
                        name in this list. If not specified, the model name will be the same as the `--model`
                        argument. Noted that this name(s)will also be used in `model_name` tag content of prometheus
                        metrics, if multiple names provided, metricstag will take the first one.
  --qlora-adapter-name-or-path QLORA_ADAPTER_NAME_OR_PATH
                        Name or path of the QLoRA adapter.
  --otlp-traces-endpoint OTLP_TRACES_ENDPOINT
                        Target URL to which OpenTelemetry traces will be sent.
  --collect-detailed-traces COLLECT_DETAILED_TRACES
                        Valid choices are model,worker,all. It makes sense to set this only if --otlp-traces-endpoint
                        is set. If set, it will collect detailed traces for the specified modules. This involves use
                        of possibly costly and or blocking operations and hence might have a performance impact.
  --disable-async-output-proc
                        Disable async output processing. This may result in lower performance.
  --override-neuron-config OVERRIDE_NEURON_CONFIG
                        override or set neuron device configuration.
  --disable-log-requests
                        Disable logging requests.
  --max-log-len MAX_LOG_LEN
                        Max number of prompt characters or prompt ID numbers being printed in log. Default: Unlimited
```

Example:

```
$ vllm serve gpt2
INFO 10-22 11:16:39 api_server.py:495] vLLM API server version 0.6.1
INFO 10-22 11:16:39 api_server.py:496] args: Namespace(model_tag='gpt2', config='', host=None, port=8000, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_auto_tool_choice=False, tool_call_parser=None, model='gpt2', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=False, download_dir=None, load_format='auto', config_format='auto', dtype='auto', kv_cache_dtype='auto', quantization_param_path=None, max_model_len=None, guided_decoding_backend='outlines', distributed_executor_backend=None, worker_use_ray=False, pipeline_parallel_size=1, tensor_parallel_size=1, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=16, enable_prefix_caching=False, disable_sliding_window=False, use_v2_block_manager=False, num_lookahead_slots=0, seed=0, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.9, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_seqs=256, max_logprobs=20, disable_log_stats=False, quantization=None, rope_scaling=None, rope_theta=None, enforce_eager=False, max_context_len_to_capture=None, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, enable_lora=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, qlora_adapter_name_or_path=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, override_neuron_config=None, disable_log_requests=False, max_log_len=None, dispatch_function=<function serve at 0x79dd6c060b80>)
config.json: 100%|████████████████████████████████████████████████████████████████████| 665/665 [00:00<00:00, 11.2MB/s]
INFO 10-22 11:16:40 config.py:1653] Downcasting torch.float32 to torch.float16.
INFO 10-22 11:16:40 api_server.py:162] Multiprocessing frontend to use ipc:///tmp/22c26cb5-ee59-4c68-a19e-6075b61f83ff for RPC Path.
INFO 10-22 11:16:40 api_server.py:178] Started engine process with PID 741011
INFO 10-22 11:16:40 importing.py:10] Triton not installed; certain GPU-related functions will not be available.
INFO 10-22 11:16:42 config.py:1653] Downcasting torch.float32 to torch.float16.
WARNING 10-22 11:16:42 config.py:370] Async output processing is only supported for CUDA or TPU. Disabling it for other platforms.
INFO 10-22 11:16:42 llm_engine.py:232] Initializing an LLM engine (v0.6.1) with config: model='gpt2', speculative_config=None, tokenizer='gpt2', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=1024, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cpu, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=gpt2, use_v2_block_manager=False, num_scheduler_steps=1, enable_prefix_caching=False, use_async_output_proc=False)
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████| 26.0/26.0 [00:00<00:00, 103kB/s]
vocab.json: 100%|█████████████████████████████████████████████████████████████████| 1.04M/1.04M [00:00<00:00, 2.50MB/s]
merges.txt: 100%|███████████████████████████████████████████████████████████████████| 456k/456k [00:00<00:00, 1.46MB/s]
tokenizer.json: 100%|█████████████████████████████████████████████████████████████| 1.36M/1.36M [00:00<00:00, 5.91MB/s]
/home/pk/turbonext/vllm-tn/.venv/lib/python3.12/site-packages/transformers/tokenization_utils_base.py:1617: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be deprecated in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
generation_config.json: 100%|██████████████████████████████████████████████████████████| 124/124 [00:00<00:00, 672kB/s]
WARNING 10-22 11:16:44 cpu_executor.py:324] float16 is not supported on CPU, casting to bfloat16.
WARNING 10-22 11:16:44 cpu_executor.py:327] CUDA graph is not supported on CPU, fallback to the eager mode.
WARNING 10-22 11:16:44 cpu_executor.py:353] Environment variable VLLM_CPU_KVCACHE_SPACE (GB) for CPU backend is not set, using 4 by default.
(VllmWorkerProcess pid=741038) INFO 10-22 11:16:44 selector.py:183] Cannot use _Backend.FLASH_ATTN backend on CPU.
(VllmWorkerProcess pid=741038) INFO 10-22 11:16:44 selector.py:128] Using Torch SDPA backend.
(VllmWorkerProcess pid=741038) INFO 10-22 11:16:44 multiproc_worker_utils.py:215] Worker ready; awaiting tasks
(VllmWorkerProcess pid=741038) INFO 10-22 11:16:44 selector.py:183] Cannot use _Backend.FLASH_ATTN backend on CPU.
(VllmWorkerProcess pid=741038) INFO 10-22 11:16:44 selector.py:128] Using Torch SDPA backend.
(VllmWorkerProcess pid=741038) INFO 10-22 11:16:45 weight_utils.py:242] Using model weights format ['*.safetensors']
model.safetensors: 100%|████████████████████████████████████████████████████████████| 548M/548M [00:07<00:00, 74.1MB/s]
(VllmWorkerProcess pid=741038) INFO 10-22 11:16:53 weight_utils.py:287] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00, 12.72it/s]
(VllmWorkerProcess pid=741038)
INFO 10-22 11:16:53 cpu_executor.py:211] # CPU blocks: 7281
/home/pk/turbonext/vllm-tn/.venv/lib/python3.12/site-packages/transformers/tokenization_utils_base.py:1617: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be deprecated in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
INFO 10-22 11:16:53 api_server.py:226] vLLM to use /tmp/tmp56m8x_wi as PROMETHEUS_MULTIPROC_DIR
WARNING 10-22 11:16:53 serving_embedding.py:190] embedding_mode is False. Embedding API will not work.INFO 10-22 11:16:53 launcher.py:20] Available routes are:
INFO 10-22 11:16:53 launcher.py:28] Route: /openapi.json, Methods: GET, HEAD
INFO 10-22 11:16:53 launcher.py:28] Route: /docs, Methods: GET, HEAD
INFO 10-22 11:16:53 launcher.py:28] Route: /docs/oauth2-redirect, Methods: GET, HEAD
INFO 10-22 11:16:53 launcher.py:28] Route: /redoc, Methods: GET, HEAD
INFO 10-22 11:16:53 launcher.py:28] Route: /health, Methods: GET
INFO 10-22 11:16:53 launcher.py:28] Route: /tokenize, Methods: POST
INFO 10-22 11:16:53 launcher.py:28] Route: /detokenize, Methods: POST
INFO 10-22 11:16:53 launcher.py:28] Route: /v1/models, Methods: GET
INFO 10-22 11:16:53 launcher.py:28] Route: /version, Methods: GET
INFO 10-22 11:16:53 launcher.py:28] Route: /v1/chat/completions, Methods: POST
INFO 10-22 11:16:53 launcher.py:28] Route: /v1/completions, Methods: POST
INFO 10-22 11:16:53 launcher.py:28] Route: /v1/embeddings, Methods: POST
INFO 10-22 11:16:53 launcher.py:33] Launching Uvicorn with --limit_concurrency 32765. To avoid this limit at the expense of performance run with --disable-frontend-multiprocessing
INFO:     Started server process [740981]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
...
```
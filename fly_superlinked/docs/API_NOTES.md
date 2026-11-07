# API Notes: Superlinked SIE + cognee

Read on 2026-11-07. Sources are the official docs (SIE pages say `last_updated: 2026-11-05`, and the model catalog says `2026-11-07`), the SDK source on GitHub (`superlinked/sie` main), the `superlinked/cookbook` repo, and PyPI.

Legend: **[V]** means I read it in the docs or source. **[UNVERIFIED]** means it is my inference or I could not find it.

---

# Part 1: Superlinked SIE (Superlinked Inference Engine), Serverless

## 1.1 Package and install

- `pip install sie-sdk` [V]. PyPI latest is **0.11.0**, `requires_python >=3.12` [V]. The cookbook pins `sie-sdk>=0.10.1`.
- Imports [V]:
  ```python
  from sie_sdk import SIEClient, SIEAsyncClient
  from sie_sdk.types import Item
  from sie_sdk.client.errors import (SIEError, SIEConnectionError, RequestError, ServerError,
      ProvisioningError, RateLimitError, InputTooLongError, ModelLoadingError,
      ModelLoadFailedError, ResourceExhaustedError)
  from sie_sdk.scoring import maxsim
  ```
  Note: `RateLimitError` is not in the reference page. It is in `errors.py` and the cookbook imports it.

## 1.2 Base URL, auth, env vars

- Serverless base URL: **`https://api.superlinked.com`** [V]. Pass it without `/v1`. The SDK appends the path. For the OpenAI SDK, use `https://api.superlinked.com/v1` [V]. The serverless page also mentions "or my regional endpoint", but it names no regional URLs [UNVERIFIED].
- Auth header: **`Authorization: Bearer <SIE_API_KEY>`** [V]. Keys look like `sk-sie-…`.
- Get a key at https://console.superlinked.com/console/keys [V] (from the cookbook `.env.example`).
- Cookbook `.env.example` [V]:
  ```
  # SIE Serverless: create a key at https://console.superlinked.com/console/keys
  SIE_API_KEY=
  # SIE Cluster or SIE Local: set its URL, e.g. http://localhost:8080
  # Leave it empty for SIE Serverless. SIE_API_KEY is optional for a server without auth.
  SIE_BASE_URL=
  ```
- How the SDK resolves the base URL [V] (`sync.py` docstring, sdk 0.11): when `base_url` is omitted, it uses `SIE_BASE_URL`. Without that, it falls back to SIE Serverless (`https://api.superlinked.com`) when an API key is available (`api_key` or `SIE_API_KEY`). With neither a URL nor a key it raises `ValueError`. When `api_key` is omitted, the SDK uses `SIE_API_KEY` only for the `SIE_BASE_URL` origin, or for the Serverless origin when `SIE_BASE_URL` is unset. Pass `""` to send no credential.
  - Model pages therefore show `client = SIEClient(api_key="sk-sie-…")` with no URL.
  - The cookbook's explicit pattern [V]:
    ```python
    api_key = os.environ.get("SIE_API_KEY", "").strip()
    base_url = os.environ.get("SIE_BASE_URL", "").strip()
    if not api_key and not base_url:
        raise RuntimeError("Set SIE_API_KEY, or SIE_BASE_URL for SIE Cluster or SIE Local, in .env")
    base_url = base_url or "https://api.superlinked.com"
    return SIEClient(api_key=api_key or None, base_url=base_url, timeout_s=WAIT_S)
    ```

## 1.3 SIEClient constructor (sdk 0.11 source) [V]

```python
SIEClient(
    base_url: str | None = None,        # positional; see the resolution rules above
    *,
    timeout_s: float | None = None,     # sets connect + read timeouts (reference page says default 30.0;
                                        # v0.10 changelog: "Default reads now allow 150 seconds")
    api_key: str | None = None,
    gpu: str | None = None,
    options: dict | None = None,
    pool: PoolSpec | None = None,       # deprecated
    max_connections: int | None = None,
    control_plane_url: str | None = None,
    org: str | None = None,
    base_url_headers: Mapping[str, str] | None = None,
    connect_timeout_s: float | None = None,
    read_timeout_s: float | None = None,
    remote: Literal["forbid"] | None = None,   # sends X-SIE-Remote: forbid (no SIE Relay)
    http_client: httpx.Client | None = None,
)
```
`SIEAsyncClient` has the same API, and every method is a coroutine. Both clients are context managers (`with` / `async with`). Use `SIEAsyncClient` inside FastAPI.

## 1.4 Item and image inputs [V]

```python
class Item(TypedDict, total=False):
    id: str                  # echoed back
    text: str
    images: list[ImageInput] # multimodal models
    document: DocumentInput  # raw document bytes (docling)
    metadata: dict[str, Any]
    multivector: NDArray     # for client-side MaxSim
    # also audio / video slots

class ImageInput(TypedDict, total=False):
    data: PIL.Image.Image | np.ndarray | bytes | str | Path   # str/Path = local FILE PATH
    format: str              # 'jpeg', 'png' (inferred if absent)
```
- The SDK converts PIL images and numpy arrays to JPEG bytes. It passes `bytes` through unchanged and reads `str`/`Path` values from disk. **Image URLs are not accepted in `encode`**: download the bytes yourself. The model page does exactly that with `urlopen(...).read()`. Base64 is not an `Item` input.
- There is a shorthand without the dict wrapper: `Item(images=[pil_image, "photo2.jpg"])`.
- On the HTTP wire, `ImageInput = {data: bytes, format: 'jpeg'|'png'|'webp'}`. Msgpack is the default wire format. For JSON you'd have to encode the bytes, and the docs don't show how [UNVERIFIED]. Use the SDK.

## 1.5 encode() [V]

```python
def encode(model: str, items: Item | list[Item], *, output_types: list[str] | None = None,  # 'dense','sparse','multivector'
           instruction: str | None = None, output_dtype: str | None = None,              # float32|float16|int8|binary
           is_query: bool | None = None, options: dict | None = None, gpu: str | None = None,
           wait_for_capacity: bool = True, provision_timeout_s: float | None = None
) -> EncodeResult | list[EncodeResult]   # single Item in -> single result; list in -> list
```
`EncodeResult` (TypedDict): `model`, `id`, `dense: np.ndarray[float32]`, `sparse: {indices, values}`, `multivector`, `timing`, and `request` (`RequestMetadata`, which includes `usage` and `credits_debited`).

```python
vec = client.encode("google/siglip-so400m-patch14-384", [{"text": "a photo of a lamp."}], is_query=True)[0]["dense"]
imgs = client.encode("google/siglip-so400m-patch14-384",
                     [{"images": [{"data": jpeg_bytes, "format": "jpeg"}]}], is_query=False)
```
- The SigLIP model page batches 10 items per call: `items[start:start + 10]`. The docs state no hard batch limit [UNVERIFIED].
- SigLIP reads at most **64 text tokens** [V]. Cookbook advice: format queries as the caption `"a photo of {query}."`.
- In raw HTTP, `is_query` goes in `params.options.is_query` [V] (from the serverless page's build prompt).

## 1.6 score() for reranking [V]

```python
def score(model: str, query: Item, items: list[Item], *, instruction: str | None = None,
          options: dict | None = None, gpu=None, wait_for_capacity=True, provision_timeout_s=None) -> ScoreResult
# ScoreResult: {model, query_id, scores: [{item_id, score, rank}], usage: {input_tokens, images?}, request}
# scores are sorted by score descending; rank 0 = most relevant
ranked = client.score("Qwen/Qwen3-Reranker-0.6B", {"text": query}, [{"id": "0", "text": d} for d in docs])
```

## 1.7 extract() [V]

```python
def extract(model: str, items: Item | list[Item], *, labels: list[str] | None = None,
            output_schema: dict | None = None, instruction: str | None = None, options: dict | None = None,
            gpu=None, wait_for_capacity=True, provision_timeout_s=None) -> ExtractResult | list[ExtractResult]
# ExtractResult: {model, id, entities:[{text,label,score,start,end,bbox}], relations:[{head,tail,relation,score}],
#                 classifications:[{label,score}], objects, data: dict (structured), error, request}
```
These are GLiNER-family NER, relation, classify, OCR and detection models. It is not an LLM JSON tool. For structured output from an LLM, use `chat_completions` with `response_format` (see below).

## 1.8 generate() [V] (preview; prompt in, text out)

```python
def generate(model: str, prompt: str, *, max_new_tokens: int,   # REQUIRED
             temperature=None, top_p=None, stop=None, gpu=None, wait_for_capacity=True,
             provision_timeout_s=None, **kw)  # kw: images, grammar, seed, frequency_penalty, presence_penalty,
                                              #     logit_bias, lora_adapter, options, max_oom_retries, ...
-> GenerateResult  # {text, finish_reason ('stop'|'length'), usage{prompt_tokens,completion_tokens,total_tokens}, ttft_ms, tpot_ms}
```
`stream_generate(...)` yields chunks with `text_delta`, and the final chunk carries `done`. HTTP `/v1/generate/:model` needs model ids written with `__` in place of `/`. The SDK converts this for you.
`generate` takes a `grammar` kwarg ("native structured-output grammar"), but the docs give no format for it [UNVERIFIED]. **For JSON schema output, use chat_completions.**

## 1.9 chat_completions(): OpenAI-compatible, vision, JSON schema [V]

SDK method from `sync.py` (sdk 0.11). It is not on the reference page, but every model page and the cookbook use it:
```python
def chat_completions(model: str, messages: list[ChatMessage], *, max_completion_tokens: int | None = None,
    max_tokens=None, temperature=None, top_p=None, top_k=None, repetition_penalty=None, stop=None,
    tools=None, tool_choice=None, parallel_tool_calls=None, response_format: dict | None = None,
    frequency_penalty=None, presence_penalty=None, n=None, best_of=None, logprobs=None, top_logprobs=None,
    logit_bias=None, seed=None, user=None, safety_identifier=None, lora_adapter=..., include_reasoning=None,
    gpu=None, extra_body: dict | None = None, wait_for_capacity=True, provision_timeout_s=None,
    max_oom_retries=..., read_timeout_s=None) -> ChatCompletion   # OpenAI-shaped dict: result["choices"][0]["message"]["content"]
```
`SIEAsyncClient.chat_completions` and `stream_chat_completions` also exist [V].

**Vision plus strict JSON schema.** The cookbook (`ecommerce/tag_products_from_photo_and_text/recipe.py`) does this [V]:
```python
image = base64.b64encode(jpeg_bytes).decode("ascii")
content = [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
    {"type": "text", "text": prompt},
]
response = client.chat_completions(
    "Qwen/Qwen3.8-27B-FP8",
    [{"role": "user", "content": content}],
    max_completion_tokens=1500,
    temperature=0.0,
    response_format={"type": "json_schema",
                     "json_schema": {"name": "tags", "schema": schema, "strict": True}},
    provision_timeout_s=WAIT_S,
)
choice = response["choices"][0]; text = choice["message"]["content"]; choice.get("finish_reason")
```
- `image_url` content parts take **data URIs** (`data:image/jpeg;base64,...`) [V]. Whether remote `https://` image URLs work in chat is **[UNVERIFIED]**: every example uses data URIs. The cookbook shrinks photos to a 512 px maximum edge first.
- `response_format: {"type":"json_schema", "json_schema": {...,"strict": true}}` is supported [V]. Changelog: "Strict JSON-schema and regex output is verified at completion". Also: "Local chat now adds `json_schema` to the prompt by default (`schema_in_prompt: false` to disable)".
- Raw HTTP [V]:
  ```bash
  curl https://api.superlinked.com/v1/chat/completions \
    -H "Authorization: Bearer $SIE_API_KEY" -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3.5-4B","messages":[{"role":"user","content":"hello"}],"max_completion_tokens":16}'
  ```
- The OpenAI Python SDK works too: `OpenAI(api_key="sk-sie-…", base_url="https://api.superlinked.com/v1")` [V]. The docs show this for `/v1/decisions`, and the chat endpoint is described as OpenAI-compatible.
- Model ids can carry a **profile suffix**. The Gemma page uses `"google/gemma-4-31B-it:h100-96k-hires-out8k-no-spec"`, and the GLM page uses `"zai-org/GLM-5.3-Flash:relay"` [V].

## 1.10 HTTP endpoints [V] (from https://superlinked.com/docs/reference/api)

| Endpoint | Purpose |
|---|---|
| `POST /v1/encode/:model` | embeddings; body `{items:[Item], params:{output_types, instruction, output_dtype, options}}` |
| `POST /v1/score/:model` | rerank; body `{query: Item, items:[Item], instruction?, options?}` |
| `POST /v1/extract/:model` | NER, relations, classification, structured extraction; `{items, params:{labels, output_schema, instruction, options}}` |
| `POST /v1/generate/:model` | text gen (model id with `__` instead of `/`) |
| `GET /v1/models`, `GET /v1/models/:model` | catalog: inputs, outputs, `dims`, max_sequence_length |
| `POST /v1/embeddings` | OpenAI-compatible embeddings (`{"model":..., "input":[...]}`) |
| `POST /v1/chat/completions` | OpenAI-compatible chat (gateway) |
| `POST /v1/completions`, `/v1/responses`, `/v1/rerank`, `/v1/audio/transcriptions`, `/v1/decisions` | OpenAI-style extras |

- The wire format defaults to msgpack. Send `Accept: application/json` to get JSON back.
- Model ids in the path can be URL-encoded (`Qwen%2FQwen3-Embedding-4B`) or raw (`/v1/encode/BAAI/bge-m3`) [V].
- Errors come back as `{"detail":{"code":"MODEL_NOT_FOUND","message":...}}`. Codes: `MODEL_NOT_FOUND` 404, `INVALID_INPUT` 400, `MODEL_LOADING` 503 (Retry-After), `QUEUE_FULL` 503, `RESOURCE_EXHAUSTED` 503, `PROVISIONING` 503 (Retry-After: 60). Chat and generate return OpenAI-style error objects instead.

## 1.11 Model IDs (Serverless-available unless noted) and prices

Catalog: https://superlinked.com/models (Markdown version at https://superlinked.com/models.md). "44 of the 149 run on SIE Serverless." Prices are Serverless list prices [V].

| Role | Model id | Reads | Serverless price |
|---|---|---|---|
| Chat, small, **vision** | `Qwen/Qwen3.5-4B` | text up to 8,192 tok, **image** | $0.305 in / $1.53 out per 1M tokens |
| Chat, larger, **vision** (used by the cookbook with image + JSON schema) | `Qwen/Qwen3.8-27B-FP8` | text up to 8,192, **image** | $0.25 in / $2.75 out per 1M tokens |
| Chat / VLM | `google/gemma-4-31B-it` | text up to 8,192, image | $0.45 in / $1.50 out per 1M |
| Chat, cheap, text only, 131k context | `zai-org/GLM-5.3-Flash` (page uses `:relay` suffix) | text | $0.15 in / $0.50 out per 1M |
| Chat (NOT on Serverless) | `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3-0.6B` | | self-host only |
| **Image embedding** | `google/siglip-so400m-patch14-384` (1152-d, 384px; text up to 64 tok) | text + image | $0.031 per 1M tokens; $0.0233 per 1,000 images |
| Image embedding, smaller | `google/siglip2-base-patch16-224` | text (64 tok) + image | $0.0231 per 1M tokens; $0.0143 per 1,000 images |
| (CLIP models) | `openai/clip-vit-*`, `laion/CLIP-*` | | NOT on Serverless |
| **Text embedding** | `Qwen/Qwen3-Embedding-4B` (up to 32,768 tok) | text | $0.06 per 1M tokens |
| Text embedding, cheaper | `BAAI/bge-m3` (1024-d dense; 8,192 tok) | text | $0.0377 per 1M tokens (dense) |
| Text embedding | `Snowflake/snowflake-arctic-embed-l-v2.0` (8,192 tok) | text | $0.0377 per 1M tokens |
| **Reranker** | `Qwen/Qwen3-Reranker-0.6B` (32,768 tok) | text | $0.017 per 1M tokens |
| Reranker, best | `Qwen/Qwen3-Reranker-4B` (55 ms/query) | text | $0.0425 per 1M tokens |
| Reranker, multimodal | `Qwen/Qwen3-VL-Reranker-2B` | text + image | $0.149 per 1,000 images; $222.80 per 1M pairs |
| NER (extract) | `urchade/gliner_multi-v2.1` | text | $0.0907 per 1M tokens |

- Dimensions: SigLIP so400m-384 = 1152 [V] (multimodal docs table). bge-m3 dense = 1024 [V] (API example). The docs don't state dims for Qwen3-Embedding-4B or arctic-embed-l-v2.0 [UNVERIFIED]. Query `GET /v1/models/:model` and read `dims` at runtime.
- **Free tier** [V] (https://superlinked.com/serverless): "$20 Free credits — 530M tokens embedded, 50k pages parsed, 10M tokens generated (1:1 in:out) — no credit card required — apply for $500 more."
- Bring-your-own-model GPU pricing is per GPU-second, from T4 $0.000481 to H100 $0.00142 per GPU-second [V].
- SOC 2 Type 2, with US or EU data residency [V].

## 1.12 Rate limits and cold starts

- Rate limits **exist but no numbers are published** [V for existence, UNVERIFIED for limits]. `RateLimitError` docstring: "HTTP `429 TOO_MANY_REQUESTS` (code `RATE_LIMIT`, per-key or per-account, default-on)". The SDK retries and honours `Retry-After` within `provision_timeout_s`, then raises `RateLimitError`. A 402 means no credits or a spend limit was hit (cookbook `is_fatal` comment).
- Cold starts [V]: dedicated capacity scales to zero, so the first request can be slow. While capacity starts, **SIE Relay** (a partner, on by default) may serve the request. `result["request"]["served_by"|"upstream"|"fallback_reason"]` show which one did. `remote="forbid"` (or the header `X-SIE-Remote: forbid`) disables Relay, and you then get `503 PROVISIONING` with `Retry-After` until capacity is up. The SDK waits by default (`wait_for_capacity=True`). Set `provision_timeout_s` for a hackathon demo.

---

# Part 2: cognee

## 2.1 Package

- `pip install cognee`. PyPI latest is **1.6.7**, `requires_python >=3.10,<3.15` [V]. Python 3.12 is fine.
- Extras mentioned [V]: `cognee[fastembed]` (local embeddings) and `cognee[baml]`.
- Everything is `async`. `import cognee`. `from cognee import SearchType`.

## 2.2 v1.0 core API (exact signatures) [V]

```python
async def remember(
    data: str | list[str] | UploadFile | list[UploadFile] | DataItem | list[DataItem] | MemoryEntry | MemorySource | PresortReport | dict,
    dataset_name: str = "main_dataset",
    *, session_id: str | None = None, chunk_size: int | None = None, chunker=None, custom_prompt: str | None = None,
    run_in_background: bool = False, self_improvement: bool = True, session_ids: list[str] | None = None,
    dry_run: bool | Literal["presort"] = False, raise_on_error: bool = True,
    **kwargs,   # incl. user=, node_set: list[str], dataset_id: UUID, graph_model, extractor, summary_method,
                # incremental_loading, llm_config, embedding_config, vector_db_config, graph_db_config, ...
) -> RememberResult | DryRunEstimate | PresortReport | dict
```
- `RememberResult` fields: `status`, `dataset_name`, `dataset_id` (a **string**, so wrap it in `UUID(...)`), `session_ids`, `elapsed_seconds`, `raw_result`, `improve`, `improve_error`, `improve_skipped`.
- `data` accepts text, file paths, URLs, and FastAPI `UploadFile` ("a plain `open(..., 'rb')` handle is not accepted — pass the path"). Images, audio and video need an LLM at ingestion time, because their loaders describe them with the LLM (`IMAGE_TRANSCRIBE_MODEL`, else `LLM_MODEL`).
- A failed graph build raises `CognifyFailedError` (pass `raise_on_error=False` to get a result back instead).
- `node_set` must be a list of non-empty strings, even for a single tag. Anything else raises `InvalidNodeSetError`.

```python
async def recall(
    query_text: str, query_type: SearchType | None = None, *,
    datasets: list[str] | None = None, dataset_ids: list[UUID] | None = None, top_k: int = 15,
    auto_route: bool = True, scope: str | list[str] | None = None,   # "graph"|"session"|"trace"|"session_context"|"session_first"|"all"|"auto"
    # keyword-only extras (NO **kwargs; unknown kwargs -> TypeError):
    system_prompt=..., system_prompt_path=..., node_name: list[str]=..., node_name_filter_operator=..., only_context: bool=...,
    session_id: str=..., context_profile=..., wide_search_top_k=..., triplet_distance_penalty=..., feedback_influence: float=...,
    verbose=..., retriever_specific_config: dict=..., code_query=..., response_model: type[BaseModel]=..., include_references: bool=...,
    user=..., llm_config=..., embedding_config=...,
) -> list[RecallResponse]
```
- Results are **Pydantic objects** (`r.text`), not dicts. Graph results are `ResponseGraphEntry` with `text, kind, search_type, score, dataset_id, dataset_name, metadata, raw, structured`. Session results are `ResponseQAEntry` with `qa_id, question, answer, context, feedback_text, feedback_score`. Some results are `source="system"` markers (`memory_warming_up`, `build_failed`).
- `datasets` names resolve **only against datasets the current user owns**. Use `dataset_ids` for datasets shared with you. An unknown name raises `DatasetNotFoundError` (404). If you omit both, recall auto-routes across every readable dataset.
- Use `node_name=[...]` to filter by node set. `node_type` raises `TypeError` on recall.
- Calling `recall()` before any ingestion raises `RecallPreconditionError`.
- Structured output: pass `response_model=MyPydanticModel` and read `results[0].structured` (a dict).

```python
async def improve(
    dataset: str | UUID = "main_dataset", *, run_in_background: bool = False,
    node_name: list[str] | None = None,              # deprecated
    session_ids: list[str] | None = None,
    build_global_context_index: bool | None = None, build_truth_subspace: bool = False,
    review_conflicts: bool | None = None,
    **kwargs,   # user=, feedback_alpha=, (deprecated) extraction_tasks / enrichment_tasks / data / node_type
) -> ImproveResult   # .status, .stages[StageResult(stage,status,reason,error,counts,duration_ms)], .stage(name)
```
- Needs **write** permission. Another user's dataset must be given as a UUID. A name resolves to (or creates) *your own* dataset.

```python
async def forget(*, data_id: UUID | None = None, dataset: str | None = None, dataset_id: UUID | None = None,
                 everything: bool = False, memory_only: bool = False, clean_history: bool = False, user=None) -> dict
```
- Needs **delete** permission. A name you can't delete raises `DatasetNotFoundError` (404). A `dataset_id` you can't delete raises `PermissionDeniedError` (403).

## 2.3 Legacy API [V]

```python
async def add(data, dataset_name: str = 'main_dataset', user: User = None, node_set: list[str] | None = None,
              vector_db_config=None, graph_db_config=None, dataset_id: UUID | None = None, preferred_loaders=None,
              dlt_config=None, incremental_loading: bool = True, data_per_batch: int | None = 20,
              importance_weight: float | None = 0.5, run_in_background: bool = False, llm_config=None,
              embedding_config=None, data_cache: bool = True, skip_connection_test: bool = False, branch: str | None = None)

async def cognify(datasets: str | list[str] | list[UUID] | None = None, user: User = None,
                  graph_model=KnowledgeGraph, chunker=TextChunker, chunk_size=None, chunks_per_batch=None, config=None,
                  vector_db_config=None, graph_db_config=None, run_in_background=False, incremental_loading=True,
                  custom_prompt=None, data_per_batch=20, llm_config=None, embedding_config=None, data_cache=True,
                  dry_run=False, raise_on_error=True, chunk_attachment=None, extractor=None,   # 'llm'|'gliner_demo'|'gliner'
                  ontology_file_path=None, summary_method=None, early_chunk_indexing=None)

async def search(query_text: str, query_type: SearchType = SearchType.HYBRID_COMPLETION, user: User | None = None,
                 datasets: list[str] | str | None = None, dataset_ids: list[UUID] | UUID | None = None,
                 system_prompt_path='answer_simple_question.txt', system_prompt=None, top_k=15,
                 node_type=NodeSet, node_name: list[str] | None = None, node_name_filter_operator='OR',
                 only_context=False, session_id=None, wide_search_top_k=None, triplet_distance_penalty=None,
                 feedback_influence=<DEFAULT_FEEDBACK_INFLUENCE, 0.0>, verbose=False, retriever_specific_config=None,
                 neighborhood_depth=None, neighborhood_seed_top_k=None, skills=None, tools=None, max_iter=None,
                 include_references=False, llm_config=None, embedding_config=None, code_query=None) -> list[SearchResult]

async def memify(...)   # DEPRECATED: a call with no tasks runs nothing and returns {}. Use improve().
```

**SearchType values** [V]: `SUMMARIES, CHUNKS, CHUNKS_LEXICAL, RAG_COMPLETION, HYBRID_COMPLETION (default), TRIPLET_COMPLETION (deprecated), GRAPH_COMPLETION, GRAPH_COMPLETION_DECOMPOSITION, GRAPH_SUMMARY_COMPLETION, GRAPH_COMPLETION_COT, GRAPH_COMPLETION_CONTEXT_EXTENSION, TEMPORAL, CYPHER, NATURAL_LANGUAGE, CODING_RULES, CODE, GRAPH_REPORT, BROAD, SKILLS, AGENTIC_COMPLETION, FEELING_LUCKY`.
- **There is no `SearchType.FEEDBACK`** and no `save_interaction` flag anywhere in the current docs [V: searched all fetched pages]. Both belong to older APIs. Feedback now goes through sessions (see 2.6).
- `CHUNKS`, `SUMMARIES` and `CHUNKS_LEXICAL` make no LLM call, so they are the fastest.

## 2.4 Multi-user access control [V]

- `ENABLE_BACKEND_ACCESS_CONTROL=true`. The docs say "this is set to true by default". It auto-enables when the graph and vector providers support per-dataset handlers, and the defaults Ladybug + LanceDB do. Set it to `false` to force single-user mode. `REQUIRE_AUTHENTICATION=true` matters for the REST server.
- Set these in `.env`, the shell, or `os.environ` **before importing cognee**.
- With access control on, storage is per user and per dataset: `.cognee_system/databases/<user_uuid>/<dataset_uuid>.pkl` (graph) plus `<dataset_uuid>.lance.db`.
- Setup: `from cognee.modules.engine.operations.setup import setup`, then `await setup()` creates the user-management tables. The first `remember()` also does this.
- Users:
  ```python
  from cognee.modules.users.methods import get_default_user, get_user, get_user_by_email, get_user_id_by_email, create_user
  user = await create_user(email="a@x.com", password="pw", is_superuser=False)   # raises fastapi_users.exceptions.UserAlreadyExists
  default = await get_default_user()   # default_user@example.com (DEFAULT_USER_EMAIL / DEFAULT_USER_PASSWORD); superuser
  ```
  `create_user` kwargs: `is_superuser=False, is_active=True, is_verified=False, auto_login=False, parent_user_id=None`.
- Ownership: `await cognee.remember(..., dataset_name="X", user=user)`. The creator gets read, write, delete and share. `create_authorized_dataset(name, user)` comes from `cognee.modules.data.methods`.
- Granting permissions:
  ```python
  # unchecked grant (no authorization check):
  from cognee.modules.users.permissions.methods import give_permission_on_dataset
  await give_permission_on_dataset(principal, dataset.id, "read")   # principal = User | Tenant | Role object; one perm per call; idempotent

  # authorized grant (acting user must hold "share"):
  from cognee.modules.users.permissions.methods import authorized_give_permission_on_datasets
  await authorized_give_permission_on_datasets(grantee_principal_id, [dataset_id], "read", owner_user_id)
  ```
  - The only valid permission names are **`"read"`, `"write"`, `"delete"`, `"share"`**. Any other name raises `PermissionNotFoundError`.
  - `revoke_permission_on_dataset` exists (ACL page; I did not read its signature [UNVERIFIED]).
  - `get_all_user_permission_datasets(user, "read")` lists the datasets a user can access.
  - Tenants and roles: `create_tenant(name, owner_id)`, `select_tenant(user_id=, tenant_id=)`, and `add_user_to_tenant(user_id, tenant_id, owner_id)` live in `cognee.modules.users.tenants.methods`. `create_role(role_name, owner_id)` and `add_user_to_role(user_id, role_id, owner_id)` live in `cognee.modules.users.roles.methods`.
- **Lacking a permission raises** `from cognee.modules.users.exceptions import PermissionDeniedError` (HTTP 403). Message example: `"...does not have permission to write to dataset: QUANTUM"`. This applies to both `recall` (read) and `remember` into another user's dataset (write). Write example from the docs:
  ```python
  try:
      await cognee.remember(text, dataset_id=quantum_dataset_id, user=user_1)
  except PermissionDeniedError: ...
  ```
- Datasets are shared **by id, not name**. A dataset id is a `uuid5` of the name, owner id and tenant id. Names cannot contain spaces or dots.

## 2.5 Node sets and datasets [V]

- `await cognee.remember(..., node_set=["projectA", "finance"])` tags content. Filter with `recall(..., node_name=["projectA"])`.
- A node set is a retrieval filter, **not an isolation boundary**. Use separate datasets for isolation.
- Which parameter each call uses to name a dataset:

  | Call | By name | By id |
  |---|---|---|
  | `remember` / `add` | `dataset_name=` | `dataset_id=` |
  | `cognify` | `datasets=` | `datasets=` |
  | `recall` / `search` | `datasets=[...]` | `dataset_ids=[...]` |
  | `improve` | `dataset=` | `dataset=` |
  | `forget` | `dataset=` | `dataset_id=` |

- `await cognee.datasets.list_datasets()` returns objects with `.id` and `.name`.

## 2.6 Feedback and self-improvement [V]

1. Run `recall(..., session_id="s1")`. With `CACHING=true` (the default), this stores the Q&A in the session.
2. `entries = await cognee.session.get_session(session_id="s1", last_n=5)`, then `qa_id = entries[-1].qa_id`. Entries come back in chronological order.
3. `ok = await cognee.session.add_feedback(session_id="s1", qa_id=qa_id, feedback_text="...", feedback_score=5, user=None)`. The score is an integer 1–5. It returns `False` only when the entry is not found or caching is off. Any other problem raises (for example `SessionParameterValidationError` from `cognee.infrastructure.databases.exceptions`).
4. `await cognee.improve(dataset="main_dataset", session_ids=["s1"])` folds feedback weights and session Q&A into the graph.
5. `await cognee.recall(..., feedback_influence=0.5)`. The default `DEFAULT_FEEDBACK_INFLUENCE` is `0.0`, so feedback has **no ranking effect until you opt in**.
- `cognee.session.delete_feedback(session_id=, qa_id=)` clears feedback.
- `PERSONALIZATION_ENABLED=true` turns on per-user preference weights. Their maximum influence is `PERSONALIZATION_INFLUENCE` (0.3).
- `remember(self_improvement=True)` (the default) auto-runs `improve()`. Turn that off with `IMPROVE_AUTO_ENABLED=false`, or pass `self_improvement=False` to keep things fast in a demo.
- Session-only memory: `remember(text, dataset_name=D, session_id=S)`. Bridge it into the graph later with `improve(dataset=D, session_ids=[S])`.
- Other env vars that cost LLM calls: `AUTO_FEEDBACK=true` (default) adds one session-turn LLM call to most searches, and `IMPROVE_REVIEW_CONFLICTS=true` (default) adds conflict review.

## 2.7 LLM provider env vars (OpenAI-compatible custom) [V]

```dotenv
LLM_PROVIDER="custom"                       # REQUIRED for openrouter/, hosted_vllm/, deepseek/ etc. (else ProviderNotDeducibleError)
LLM_MODEL="openai/<model-id-your-endpoint-reports>"   # 'openai/' = protocol prefix for any OpenAI-compatible server
LLM_ENDPOINT="https://<host>/v1"            # base URL; endpoint must expose /v1/chat/completions
LLM_API_KEY="<key>"                         # custom always sends one; use "." if no auth
# optional
LLM_MAX_COMPLETION_TOKENS=16384             # also sizes cognify chunks (~half per chunk)
LLM_TEMPERATURE=...
LLM_INSTRUCTOR_MODE=json_schema_mode|json_mode|tool_call|md_json   # only used with STRUCTURED_OUTPUT_FRAMEWORK=instructor
COGNEE_SKIP_CONNECTION_TEST=true            # skip 30s preflight LLM test
FALLBACK_MODEL / FALLBACK_ENDPOINT / FALLBACK_API_KEY
IMAGE_TRANSCRIBE_MODEL=...                  # model used to describe images during ingestion (defaults to LLM_MODEL)
```
- Valid `LLM_PROVIDER` values: `openai, azure, anthropic, gemini, mistral, bedrock, ollama, llama_cpp, custom, mcp-sampling`. `openai_compatible` is **not** a valid LLM provider.
- **Warning:** if you configure only the LLM side or only the embedding side, the other side defaults to OpenAI, and `add()`/`remember()` raise `ProviderConfigMismatchError`. Configure both.
- [UNVERIFIED: my inference, not documented] To point cognee at SIE Serverless, the likely settings are `LLM_PROVIDER=custom`, `LLM_MODEL=openai/Qwen/Qwen3.5-4B`, `LLM_ENDPOINT=https://api.superlinked.com/v1`, `LLM_API_KEY=$SIE_API_KEY`. Test it with `COGNEE_SKIP_CONNECTION_TEST=false` first. SIE models list an 8,192-token text window, so lower `LLM_MAX_COMPLETION_TOKENS` (for example to 4096) so prompts fit.

## 2.8 Structured-output framework [V]

- `STRUCTURED_OUTPUT_FRAMEWORK` is `litellm_native` by default. The other options are `instructor` and `baml` (install `cognee[baml]`, which uses its own `BAML_LLM_PROVIDER`, `BAML_LLM_MODEL`, ... settings).
- With `litellm_native` and a **custom OpenAI-compatible endpoint**, cognee takes the "JSON-object fallback" path. It requests a JSON object, injects the schema into the prompt, validates the reply, strips a wrapping code fence, and retries with error feedback.
- With `instructor`, the `custom` provider defaults to `json_mode`. If extraction produces malformed JSON, try `tool_call` or `json_schema_mode`.
- Weak or small models can produce malformed JSON or poor graphs. If you see repeated `InstructorRetryException`, use BAML or a stronger model.

## 2.9 Embedding provider env vars [V]

| `EMBEDDING_PROVIDER` | Notes |
|---|---|
| `openai` (default) | `EMBEDDING_MODEL="openai/text-embedding-3-large"`, `EMBEDDING_DIMENSIONS=3072` |
| `fastembed` | Local CPU (ONNX), no key. Needs `pip install 'cognee[fastembed]'`. E.g. `EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"`, `EMBEDDING_DIMENSIONS=384` (also `BAAI/bge-small-en-v1.5` 384, `BAAI/bge-base-en-v1.5` 768). The first run downloads weights. |
| `openai_compatible` | OpenAI SDK straight to `/v1/embeddings`, bypassing LiteLLM. Use the bare model id. `EMBEDDING_ENDPOINT` is normalised (adds `/v1`). The key falls back to `LLM_API_KEY`. |
| `custom` | LiteLLM. `EMBEDDING_MODEL` needs a prefix (`openai/…`). The endpoint is passed verbatim (do **not** end it in `/embeddings`). `EMBEDDING_API_KEY` is required: with `LLM_PROVIDER=custom` it does **not** fall back to `LLM_API_KEY`. |
| `ollama` | Endpoint `http://localhost:11434/api/embed` |

Other variables: `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` (**always set it**, because otherwise it falls back to 3072 and mismatches the vector store), `EMBEDDING_API_KEY`, `EMBEDDING_ENDPOINT` (alias `EMBEDDING_API_BASE`), `EMBEDDING_MAX_COMPLETION_TOKENS` (default 4096), `EMBEDDING_BATCH_SIZE`, `EMBEDDING_ENCODING_FORMAT`.
- [UNVERIFIED inference] For SIE embeddings: `EMBEDDING_PROVIDER=openai_compatible`, `EMBEDDING_ENDPOINT=https://api.superlinked.com/v1`, `EMBEDDING_MODEL=BAAI/bge-m3`, `EMBEDDING_DIMENSIONS=1024`, `EMBEDDING_API_KEY=$SIE_API_KEY`. SIE serves OpenAI-compatible `/v1/embeddings`. Changing the embedding model or dimensions requires a prune or reset.

## 2.10 Local default stores and directories [V]

| Store | Default | Path |
|---|---|---|
| Relational | SQLite (`DB_PROVIDER=sqlite`, db `cognee_db`) | `<SYSTEM_ROOT_DIRECTORY>/databases/cognee_db` |
| Vector | LanceDB (`VECTOR_DB_PROVIDER=lancedb`) | `<SYSTEM_ROOT_DIRECTORY>/databases/cognee.lancedb` |
| Graph | **Ladybug** (`GRAPH_DATABASE_PROVIDER=ladybug`; **`kuzu` is accepted as an alias**, since Ladybug is the Kùzu successor) | `<SYSTEM_ROOT_DIRECTORY>/databases/cognee_graph_ladybug` |
| Session cache | `CACHE_BACKEND=sqlite` (`cache.db`), `CACHING=true` | |

- `SYSTEM_ROOT_DIRECTORY` defaults to `<cognee package>/.cognee_system`, which is inside the venv. `DATA_ROOT_DIRECTORY` defaults to `<cognee package>/.data_storage`.
- **`.env` values must be absolute paths** (or `s3://`). Relative paths raise an error. The runtime setters are `cognee.config.system_root_directory(abs)` and `cognee.config.data_root_directory(abs)`.
- Reset helpers: `await cognee.prune.prune_data()` and `await cognee.prune.prune_system(metadata=True)`. The second one drops users and ACLs too.
- Ladybug is a single-writer embedded store, so use one worker process. The improve and dataset locks are process-local.

## 2.11 REST server (optional) [V]

- Start it with `uvicorn cognee.api.client:app --host 0.0.0.0 --port 8000`. API docs are at `/docs`.
- Auth: `POST /api/v1/auth/register` with JSON `{email, password}`, then `POST /api/v1/auth/login` (form-encoded `username`, `password`) returns `access_token`. Send it as `Authorization: Bearer <token>`.
- Endpoints: `POST /api/v1/remember` (multipart `data`, `datasetName`), `POST /api/v1/recall` (JSON `{"query", "datasets", "search_type", "top_k"}`, add `"stream": true` for SSE), `POST /api/v1/improve` (`{"dataset_name", "run_in_background"}`), `POST /api/v1/forget`, `GET/POST /api/v1/datasets`.

---

## Things I could not verify

- SIE: numeric rate limits per key or account, beyond confirming that 429 `RATE_LIMIT` exists and is on by default.
- SIE: whether `image_url` with a remote `https://` URL works in chat. Only data URIs are documented and used.
- SIE: a hard batch-size limit for `encode` (the examples batch 10).
- SIE: embedding dims for Qwen3-Embedding-4B and arctic-embed-l-v2.0.
- SIE: names of regional endpoints.
- SIE: whether `/v1/embeddings` on Serverless accepts every encode model (it is documented as OpenAI-compatible on the gateway).
- SIE: `generate(grammar=...)` format.
- cognee + SIE together: the env-var combos in 2.7 and 2.9 follow from both docs but are untested. Cognee's JSON-object fallback against SIE chat models is also untested.
- cognee: the `revoke_permission_on_dataset` signature.
- Legacy `save_interaction` and `SearchType.FEEDBACK`: neither appears in the current (1.6.x) docs, so treat them as removed.

## Source URLs

- https://superlinked.com/docs/reference/sdk (.md), https://superlinked.com/docs/reference/api (.md), https://superlinked.com/serverless, https://superlinked.com/models (.md), https://superlinked.com/models/qwen-qwen3-5-4b.md, …/google-gemma-4-31b-it.md, …/google-siglip-so400m-patch14-384.md, …/qwen-qwen3-reranker-0-6b.md, https://superlinked.com/llms-full.txt
- https://github.com/superlinked/sie/blob/main/packages/sie_sdk/src/sie_sdk/client/sync.py and errors.py
- https://github.com/superlinked/cookbook (`.env.example`, `ecommerce/search_products_by_photo_and_text/{rules.yaml,recipe.py}`, `ecommerce/tag_products_from_photo_and_text/{rules.yaml,recipe.py}`)
- https://docs.cognee.ai/python-api.md and python-api/{remember,recall,improve,forget,add,cognify,search,memify,users,search-type}.md, setup-configuration/{permissions,llm-providers,embedding-providers,structured-output-backends,overview,graph-stores,vector-stores,relational-databases}.md, guides/{permission-snippets,feedback-system,self-improving-memory-quickstart,deploy-rest-api-server}.md, examples/multi-tenant-access-control.md, core-concepts/further-concepts/{datasets,node-sets}.md
- https://github.com/topoteretes/cognee/blob/main/examples/demos/permissions/user_permissions_and_access_control_example.py (import paths)
- PyPI JSON for `sie-sdk` (0.11.0) and `cognee` (1.6.7)

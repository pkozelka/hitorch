use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Ordered by official OpenAI API documentation
/// https://platform.openai.com/docs/api-reference/completions/create
#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    /// TODO: Union[List[int], List[List[int]], str, List[str]]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<i32>,
    #[serde(skip_serializing_if = "is_false")]
    pub echo: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<i32>,
    pub max_tokens: usize,
    pub n: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// TODO: Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "is_false")]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    /// TODO: Optional[float] = 1.0
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// TODO: Optional[float] = 1.0
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// doc: begin-completion-sampling-params
    #[serde(skip_serializing_if = "is_false")]
    pub use_beam_search: bool,
    /// TODO: int = -1
    pub top_k: i32,
    pub min_p: f32,
}

fn is_false(b: &bool) -> bool {
    *b == false
}

impl CompletionRequest {
    /// Simple constructor: fill just model and prompt
    pub fn simple(model: String, max_tokens: usize, prompt: String) -> CompletionRequest {
        CompletionRequest {
            model,
            prompt: Some(prompt),
            best_of: None,
            echo: false,
            frequency_penalty: None,
            logit_bias: None,
            logprobs: None,
            max_tokens,
            n: 1,
            presence_penalty: None,
            seed: None,
            stop: None,
            stream: false,
            stream_options: None,
            suffix: None,
            temperature: None,
            top_p: None,
            user: None,
            use_beam_search: false,
            top_k: -1,
            min_p: 0.0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamOptions {
    include_usage: bool,
    continuous_usage_stats: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionResponseStreamChoice {
    pub text: String,
    pub index: i32,
    pub logprobs: Option<HashMap<String, f32>>,
    pub finish_reason: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CompletionStreamResponse {
    pub id: Option<String>,
    pub object: Option<String>,
    pub created: Option<i32>,
    pub model: Option<String>,
    pub choices: Vec<CompletionResponseStreamChoice>,
    pub usage: Option<UsageInfo>,
}

/// {"prompt_tokens":2,"total_tokens":18,"completion_tokens":16}
#[derive(Debug, Serialize, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: Option<i32>,
    pub completion_tokens: Option<i32>,
    pub total_characters: Option<i32>,
    pub total_tokens: Option<i32>,
    pub total_completions: Option<i32>,
}

/*

from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator

class CompletionLogProbs(OpenAIBaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str,
                                     float]]] = Field(default_factory=list)


class CompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class OpenAIBaseModel(BaseModel):
    # OpenAI API does not allow extra fields
    model_config = ConfigDict(extra="forbid")


class StreamOptions(OpenAIBaseModel):
    include_usage: Optional[bool] = True
    continuous_usage_stats: Optional[bool] = True


class CompletionRequest(OpenAIBaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(None, ge=_LONG_INFO.min, le=_LONG_INFO.max)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    # doc: begin-completion-sampling-params
    use_beam_search: bool = False
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    allowed_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[int] = None
    # doc: end-completion-sampling-params

    # doc: begin-completion-extra-params
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."),
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=
        ("Similar to chat completion, this parameter specifies the format of "
         "output. Only {'type': 'json_object'} or {'type': 'text' } is "
         "supported."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description="If specified, the output will follow the JSON schema.",
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be one of "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))

    # doc: end-completion-extra-params

 */
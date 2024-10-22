//! VLLM

use entrypoints::openai::protocol::CompletionRequest;
use crate::entrypoints::openai::protocol::CompletionStreamResponse;

mod entrypoints;


pub async fn cmd_complete(url: &str, model_name: &str, api_key: &str, max_tokens: usize, prompt: &str) -> anyhow::Result<CompletionStreamResponse> {
    let request = CompletionRequest::simple(model_name.to_string(), max_tokens, prompt.to_string());
    // let request = serde_json::to_string_pretty(&request)?;
    let request = serde_json::to_string(&request)?;
    log::debug!("Request: {}", request);
    let response = reqwest::Client::new()
        .post(format!("{url}/completions"))
        .header("Authorization", format!("Bearer {}", api_key))
        .body(request)
        .send().await?;
    let status_code = response.status();
    let response_text = &response.text().await?;
    log::debug!("Response: {} {}", status_code, response_text);
    let response: CompletionStreamResponse = serde_json::from_str(response_text)?;
    Ok(response)
}

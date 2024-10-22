use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use reqwest::header;
use std::path::PathBuf;

pub(crate) async fn cmd_serve(_api_key: Option<String>, _model_tag: String, _config: PathBuf, host: String, port: u16, _model: String, _tokenizer: String,
                              _seed: Option<u64>, _dtype: String) -> anyhow::Result<()>{
    let addr = format!("{}:{}", host, port);
    let router = Router::new()
        .route("/health", get(serve_health))
        .route("/version", get(serve_version))
        .route("/v1/models", get(serve_v1_models))
        .route("/v1/completions", post(serve_v1_completions))
    ;
    println!("Listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}

async fn serve_health() -> &'static str {
    log::debug!("Health check");
    ""
}

async fn serve_version() -> &'static str {
    log::debug!("Version check");
    // TODO implement properly
    r##"{"version":"0.1.0"}"##
}

async fn serve_v1_models() -> &'static str {
    log::debug!("Models check");
    // TODO implement properly
    r##"{"object":"list","data":[{"id":"gpt2","object":"model","created":1729618624,"owned_by":"vllm","root":"gpt2","parent":null,"max_model_len":1024,"permission":[{"id":"modelperm-e912cde2d0a04eb9bb41e493edc0d3c2","object":"model_permission","created":1729618624,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}"##
}

/// TODO implement completions here
async fn serve_v1_completions(data: String) -> impl IntoResponse {
    // data contains something like: `{"model": "gpt2", "prompt": "Prague is"}`
    log::error!("TODO: /v1/completions with data: {data}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        [(header::CONTENT_TYPE, "text/plain")],
        "Not implemented yet"
    )
}

use std::path::PathBuf;
use clap::{Parser, Subcommand};
use vrllm::cmd_complete;

mod cmd_serve;


/// vLLM CLI - toy reimplementation in Rust
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the vLLM OpenAI Compatible API server
    Serve {
        /// If provided, the server will require this key to be presented in the header.
        #[arg(long, default_value = "")]
        api_key: Option<String>,
        /// The model tag to serve
        #[arg(long, default_value = "")]
        model_tag: String,
        /// Read CLI options from a config file.
        ///
        /// Must be a YAML with the following options:
        /// https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-serve
        #[arg(long, default_value = "./config.yaml")]
        config: PathBuf,
        /// host name
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// port number
        #[arg(long,default_value_t=8000)]
        port: u16,
        /// Name or path of the huggingface model to use.
        model: String,
        /// Name or path of the huggingface tokenizer to use. If unspecified, model name or path will be used.
        #[arg(long, default_value = "")]
        tokenizer: String,
        /// Random seed for operations.
        #[arg(long, default_value = "123")]
        seed: Option<u64>,
        /// Data type for model weights and activations.
        /// * "auto" will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.
        /// * "half" for FP16. Recommended for AWQ quantization.
        /// * "float16" is the same as "half".
        /// * "bfloat16" for a balance between precision and range.
        /// * "float" is shorthand for FP32 precision.
        /// * "float32" for FP32 precision.
        #[arg(long, default_value = "")]
        dtype: String,
        // + many, many other options
    },
    /// Generate text completions based on the given prompt via the running API server
    Complete {
        /// url of the running OpenAI-Compatible RESTful API server
        #[arg(long)]
        url: String,
        /// The model name used in prompt completion, default to the first model in list models API call.
        #[arg(long, default_value = "gpt2")]
        model_name: String,
        /// API key for OpenAI services. If provided, this api key will overwrite the api key obtained through environment variables.
        #[arg(long, default_value = "")]
        api_key: String,
        #[arg(long, default_value = "16")]
        max_tokens: usize,
        /// The prompt to be completed
        prompt: String,
    },
    /// Generate chat completions via the running API server
    Chat {
        /// url of the running OpenAI-Compatible RESTful API server
        #[arg(long)]
        url: String,
        /// The model name used in prompt completion, default to the first model in list models API call.
        #[arg(long)]
        model_name: String,
        /// API key for OpenAI services. If provided, this api key will overwrite the api key obtained through environment variables.
        #[arg(long)]
        api_key: String,
        /// The system prompt to be added to the chat template, used for models that support system prompts.
        #[arg(long)]
        system_prompt: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .init();

    let cli = Cli::parse();

    // You can check for the existence of subcommands, and if found use their
    // matches just as you would the top level cmd
    match cli.command {
        Commands::Serve { api_key, model_tag, config, host, port, model, tokenizer, seed, dtype} => {
            cmd_serve::cmd_serve(api_key, model_tag, config, host, port, model, tokenizer, seed, dtype).await?;
            Ok(())
        }
        Commands::Complete { url, model_name, api_key, max_tokens, prompt } => {
            // Call the completion function
            let response = cmd_complete(&url, &model_name, &api_key, max_tokens, &prompt).await?;
            for choice in response.choices {
                println!("---\n{}\n---", choice.text);
            }
            Ok(())
        }
        Commands::Chat { .. } => {
            Ok(())
        }
    }

    // Continued program logic goes here...
}

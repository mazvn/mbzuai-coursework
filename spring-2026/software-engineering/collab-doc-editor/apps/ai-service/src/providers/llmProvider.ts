// generic LLM provider contract.
// all providers (Gemini, Mock, etc.) must implement this interface.

export type LLMOperation = "enhance" | "summarize" | "translate" | "reformat";

export type LLMParameters = {
  style?: string;
  summaryStyle?: string;
  language?: string;
  formatStyle?: string;
};

export interface LLMRunParams {
  operation: LLMOperation;

  // the raw user-selected text (the thing we are transforming).
  selectedText: string;

  // optional instruction prompt constructed by the service (templates, policies, etc.).
  // providers can ignore this and build their own prompts if they want.
  prompt?: string;

  parameters?: LLMParameters;
}

export interface LLMRunResult {
  result: string;
  prompt?: string;
  model?: string;
}

export interface LLMStreamParams extends LLMRunParams {
  signal?: AbortSignal;
  onChunk: (chunk: string) => void | Promise<void>;
}

export interface LLMProvider {
  run(params: LLMRunParams): Promise<LLMRunResult>;
  stream(params: LLMStreamParams): Promise<LLMRunResult>;
}

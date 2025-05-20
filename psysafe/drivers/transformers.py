# psysafe/drivers/transformers.py
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from psysafe.drivers.base import ChatDriverABC
from psysafe.typing.requests import TransformersChatRequest, TransformersChatMessage
from psysafe.typing.responses import TransformersChatResponse, TransformersStreamChunk

try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    # Specific pipeline types can be imported for better type hinting if focusing on one
    # from transformers.pipelines.text_generation import TextGenerationPipeline
    # from transformers.pipelines.conversational import ConversationalPipeline
except ImportError:
    raise ImportError(
        "Transformers library is required for TransformersChatDriver. "
        "Please install it (e.g., `pip install transformers torch`)."
    )

import threading # For non-blocking streaming with TextIteratorStreamer

class TransformersChatDriver(ChatDriverABC[TransformersChatRequest, TransformersChatResponse]):
    """
    Driver for Hugging Face Transformers models.
    Supports text generation and basic conversational pipelines.
    Note: Streaming implementation is basic and might vary by model/pipeline.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        task: str = "text-generation", # or "conversational"
        device: Optional[Union[str, int]] = None, # e.g., "cuda:0" or 0, or -1 for CPU
        **pipeline_kwargs: Any
    ):
        """
        Initializes the Transformers Chat Driver.

        Args:
            model_name_or_path: Name or path of the Hugging Face model.
            tokenizer_name_or_path: Optional name or path for the tokenizer. If None,
                                   uses model_name_or_path.
            task: The task for the pipeline (e.g., "text-generation", "conversational").
            device: Device to run the model on (e.g., "cpu", "cuda", "mps", or device index).
                    If None, transformers will attempt to auto-detect.
            **pipeline_kwargs: Additional keyword arguments for the transformers.pipeline.
        """
        self.model_name_or_path = model_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path if tokenizer_name_or_path else model_name_or_path
        self.task = task
        self.device = device
        self.pipeline_kwargs = pipeline_kwargs

        self._pipeline = None
        self._tokenizer = None # Store tokenizer separately for streaming if needed

        # Eagerly initialize pipeline and tokenizer to catch errors early
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
            # For some tasks like conversational, model might not be explicitly loaded here
            # if pipeline handles it. For text-generation with streaming, explicit model is better.
            if self.task == "text-generation": # Or other tasks requiring explicit model for advanced control
                 model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
                 self._pipeline = pipeline(
                    task=self.task,
                    model=model,
                    tokenizer=self._tokenizer,
                    device=self.device,
                    **self.pipeline_kwargs
                )
            else: # For tasks like "conversational"
                self._pipeline = pipeline(
                    task=self.task,
                    model=self.model_name_or_path, # Pipeline can load model directly
                    tokenizer=self._tokenizer,
                    device=self.device,
                    **self.pipeline_kwargs
                )

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Transformers pipeline for model '{self.model_name_or_path}' "
                f"with task '{self.task}': {e}"
            ) from e


    @property
    def pipeline(self) -> Any: # Should be Union[TextGenerationPipeline, ConversationalPipeline, etc.]
        """Lazy-loaded Transformers pipeline."""
        # Pipeline is now initialized in __init__
        if self._pipeline is None:
             # This part should ideally not be reached if __init__ succeeds.
            raise RuntimeError("Transformers pipeline was not initialized.")
        return self._pipeline

    def send(self, request: TransformersChatRequest) -> TransformersChatResponse:
        """
        Send a request to the Transformers pipeline.

        Args:
            request: A dictionary containing input for the pipeline.
                     For "text-generation": {"text_input": "...", "generation_kwargs": {...}}
                     For "conversational": {"messages": [{"role": "user", "content": "..."}], "conversation_id": "..."}
                                        or simply a string for the first turn.
        """
        input_data = request.get("text_input")
        messages = request.get("messages")
        generation_kwargs = request.get("generation_kwargs", {})

        try:
            if self.task == "text-generation":
                if not input_data:
                    raise ValueError("Missing 'text_input' for text-generation task.")
                # The pipeline returns a list of dicts
                # e.g. [{'generated_text': '...'}]
                response_data = self.pipeline(input_data, **generation_kwargs)
                # We'll return the first result, assuming single input
                return response_data[0] if response_data else {"generated_text": ""}
            
            elif self.task == "conversational":
                from transformers.pipelines.conversational import Conversation
                # Input can be a string (for new conversation) or a Conversation object or messages list
                if messages: # Construct or update Conversation object
                    # This part needs careful handling of Conversation state if maintaining history
                    # For a stateless call based on messages:
                    conv_input = messages # Or convert to Conversation object if pipeline expects it
                    # The conversational pipeline might manage its own history or expect it.
                    # For simplicity, let's assume it can take a list of messages directly
                    # or a string for a single turn.
                    # This is a simplification. Real conversational use might need state.
                    if isinstance(messages, list) and len(messages) > 0:
                         # If it's a list of message dicts, take the last user message content
                         # or format it as the pipeline expects.
                         # This is highly dependent on the specific conversational model.
                         # A common pattern is to pass the Conversation object.
                         # For now, let's assume a simple string input from the last message.
                         last_user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
                         if not last_user_message:
                             raise ValueError("No user message found for conversational task.")
                         conv_input = last_user_message
                    elif isinstance(messages, str): # Single string input
                        conv_input = messages
                    else:
                        raise ValueError("Invalid 'messages' format for conversational task.")

                    # Conversational pipeline often returns a Conversation object.
                    # We need to extract the generated response.
                    # This is a placeholder, actual usage depends on pipeline version and model.
                    # response_object = self.pipeline(conv_input, **generation_kwargs)
                    # generated_text = response_object.generated_responses[-1] if response_object.generated_responses else ""
                    # return {"generated_text": generated_text, "conversation_history": response_object.messages}
                    # For now, let's make a simplified assumption that it returns something convertible
                    # This part is tricky without knowing the exact pipeline behavior.
                    # Let's assume a simplified response structure for now.
                    # This is a placeholder and likely needs refinement.
                    raw_response = self.pipeline(conv_input, **generation_kwargs)
                    # Attempt to get the last generated response
                    if hasattr(raw_response, 'generated_responses') and raw_response.generated_responses:
                        return {"generated_text": raw_response.generated_responses[-1]}
                    elif hasattr(raw_response, 'messages') and raw_response.messages: # For some pipelines
                         # Find last assistant message
                        assistant_responses = [m['content'] for m in raw_response.messages if m['role'] == 'assistant']
                        return {"generated_text": assistant_responses[-1] if assistant_responses else ""}
                    else: # Fallback if structure is unknown
                        return {"generated_text": str(raw_response)}


                else:
                    raise ValueError("Missing 'messages' for conversational task.")
            else:
                # Generic fallback, may not work well
                response_data = self.pipeline(input_data if input_data else messages, **generation_kwargs)
                return {"output": response_data} # Adjust based on typical output for other tasks

        except Exception as e:
            # print(f"Transformers pipeline error: {e}") # Replace with proper logging
            raise

    async def stream(self, request: TransformersChatRequest) -> AsyncIterator[TransformersStreamChunk]:
        """
        Stream response from the Transformers text-generation pipeline.
        Note: This is a basic implementation for text-generation and might need
              adjustments for other tasks or more complex streaming needs.
              Conversational streaming is more complex.
        """
        if self.task != "text-generation":
            # Yield a single event indicating streaming is not standardly supported for this task
            yield {"error": f"Streaming not implemented for task '{self.task}' in this basic driver."}
            return

        text_input = request.get("text_input")
        if not text_input:
            raise ValueError("Missing 'text_input' for streaming with text-generation task.")
        
        generation_kwargs = request.get("generation_kwargs", {})
        
        # Ensure tokenizer is available
        if not self._tokenizer:
            raise RuntimeError("Tokenizer not available for streaming.")

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generation needs to run in a separate thread
        # Ensure pipeline model is directly accessible for this
        if not hasattr(self.pipeline, 'model') or not hasattr(self.pipeline, 'tokenizer'):
             yield {"error": "Pipeline model or tokenizer not directly accessible for threaded generation."}
             return

        generation_thread_kwargs = {
            **self.pipeline.model.generate(
                **self.pipeline.tokenizer(text_input, return_tensors="pt").to(self.pipeline.device),
                streamer=streamer,
                **generation_kwargs
            )
        }
        
        # This is a simplified way to run generation in a thread.
        # In a real async context, ensure this doesn't block the event loop.
        # `asyncio.to_thread` would be better in Python 3.9+.
        thread = threading.Thread(target=self.pipeline.model.generate, kwargs=generation_thread_kwargs)
        thread.start()

        try:
            for new_text in streamer:
                yield {"token_text": new_text} # Or {"partial_text": new_text}
        finally:
            thread.join() # Ensure thread finishes

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "driver_type": "transformers",
            "model_name_or_path": self.model_name_or_path,
            "task": self.task,
            "supports_streaming": self.task == "text-generation", # Basic support for text-generation
            "pipeline_config": self.pipeline_kwargs,
            "device": str(self.pipeline.device) if self.pipeline else str(self.device)
        }
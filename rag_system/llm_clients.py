"""
LLM Clients Module
Provides interfaces for different LLM providers.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any


class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Generate a response for a chat conversation."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self.client
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Generate a response for a chat conversation."""
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=kwargs.get('model', self.model),
            messages=messages,
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens)
        )
        
        return response.choices[0].message.content


class GoogleGeminiClient(BaseLLMClient):
    """Google Gemini API client."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-pro",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = None
        
    def _get_model(self):
        """Lazy initialization of Gemini model."""
        if self.model is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError(
                    "google-generativeai package required. "
                    "Install with: pip install google-generativeai"
                )
        return self.model
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        model = self._get_model()
        
        generation_config = {
            'temperature': kwargs.get('temperature', self.temperature),
            'max_output_tokens': kwargs.get('max_tokens', self.max_tokens),
        }
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Generate a response for a chat conversation."""
        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"Instructions: {content}\n")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}\n")
            else:
                prompt_parts.append(f"User: {content}\n")
        
        full_prompt = "".join(prompt_parts)
        return self.generate(full_prompt, **kwargs)


class OllamaClient(BaseLLMClient):
    """Ollama local LLM client."""
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name
            base_url: Ollama server URL
            temperature: Sampling temperature
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        import requests
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', self.temperature)
                }
            }
        )
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            raise Exception(f"Ollama error: {response.text}")
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Generate a response for a chat conversation."""
        import requests
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', self.temperature)
                }
            }
        )
        
        if response.status_code == 200:
            return response.json().get('message', {}).get('content', '')
        else:
            raise Exception(f"Ollama error: {response.text}")


class HuggingFaceClient(BaseLLMClient):
    """HuggingFace Transformers local client."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "auto",
        max_length: int = 512
    ):
        """
        Initialize HuggingFace client.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on
            max_length: Maximum generation length
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.pipeline = None
        
    def _get_pipeline(self):
        """Lazy initialization of the pipeline."""
        if self.pipeline is None:
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device_map=self.device
                )
            except ImportError:
                raise ImportError(
                    "transformers package required. "
                    "Install with: pip install transformers"
                )
        return self.pipeline
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        pipe = self._get_pipeline()
        
        result = pipe(
            prompt,
            max_length=kwargs.get('max_length', self.max_length),
            num_return_sequences=1,
            do_sample=True,
            temperature=kwargs.get('temperature', 0.7)
        )
        
        generated_text = result[0]['generated_text']
        # Remove the prompt from the response
        response = generated_text[len(prompt):].strip()
        
        return response
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Generate a response for a chat conversation."""
        # Convert to single prompt
        prompt = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])
        prompt += "\nassistant:"
        
        return self.generate(prompt, **kwargs)


def get_llm_client(
    provider: str = 'openai',
    **kwargs
) -> BaseLLMClient:
    """
    Factory function to get an LLM client.
    
    Args:
        provider: LLM provider ('openai', 'gemini', 'ollama', 'huggingface')
        **kwargs: Provider-specific arguments
        
    Returns:
        LLM client instance
    """
    providers = {
        'openai': OpenAIClient,
        'gemini': GoogleGeminiClient,
        'ollama': OllamaClient,
        'huggingface': HuggingFaceClient
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")
    
    return providers[provider](**kwargs)

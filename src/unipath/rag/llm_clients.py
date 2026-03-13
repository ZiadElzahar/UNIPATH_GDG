"""LLM Clients Module."""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str:
        pass


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", temperature: float = 0.3, max_tokens: int = 1000):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

    def _get_client(self):
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
        return self.client

    def generate(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def chat(self, messages: List[Dict], **kwargs) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.content


class GoogleGeminiClient(BaseLLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro", temperature: float = 0.3, max_tokens: int = 1000):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = None

    def _get_model(self):
        if self.model is None:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        return self.model

    def generate(self, prompt: str, **kwargs) -> str:
        model = self._get_model()
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": kwargs.get("temperature", self.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            },
        )
        return response.text

    def chat(self, messages: List[Dict], **kwargs) -> str:
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"Instructions: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
            else:
                prompt_parts.append(f"User: {content}\n")
        return self.generate("".join(prompt_parts), **kwargs)


class OllamaClient(BaseLLMClient):
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434", temperature: float = 0.3):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": kwargs.get("temperature", self.temperature)},
            },
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        raise Exception(f"Ollama error: {response.text}")

    def chat(self, messages: List[Dict], **kwargs) -> str:
        import requests

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": kwargs.get("temperature", self.temperature)},
            },
        )
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "")
        raise Exception(f"Ollama error: {response.text}")


class HuggingFaceClient(BaseLLMClient):
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto", max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.pipeline = None

    def _get_pipeline(self):
        if self.pipeline is None:
            from transformers import pipeline

            self.pipeline = pipeline("text-generation", model=self.model_name, device_map=self.device)
        return self.pipeline

    def generate(self, prompt: str, **kwargs) -> str:
        pipe = self._get_pipeline()
        result = pipe(
            prompt,
            max_length=kwargs.get("max_length", self.max_length),
            num_return_sequences=1,
            do_sample=True,
            temperature=kwargs.get("temperature", 0.7),
        )
        generated_text = result[0]["generated_text"]
        return generated_text[len(prompt) :].strip()

    def chat(self, messages: List[Dict], **kwargs) -> str:
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) + "\nassistant:"
        return self.generate(prompt, **kwargs)


def get_llm_client(provider: str = "openai", **kwargs) -> BaseLLMClient:
    providers = {
        "openai": OpenAIClient,
        "gemini": GoogleGeminiClient,
        "ollama": OllamaClient,
        "huggingface": HuggingFaceClient,
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")
    return providers[provider](**kwargs)

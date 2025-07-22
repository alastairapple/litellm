"""
Azure Speech Services (Cognitive Services) TTS implementation.

This provides a direct interface to Azure Speech Services Text-to-Speech API,
which is different from Azure OpenAI TTS.
"""

from typing import Any, Callable, Optional, Union

import httpx

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    _get_httpx_client
)
from litellm.types.llms.openai import HttpxBinaryResponseContent

from .azure import AzureChatCompletion
from .common_utils import AzureOpenAIError


class AzureSpeechServices(AzureChatCompletion):
    """
    Azure Speech Services TTS implementation.
    
    This uses the Azure Cognitive Services Speech API directly, 
    not the Azure OpenAI TTS endpoint.
    """

    def audio_speech(
        self,
        model: str,
        input: str,
        voice: str,
        optional_params: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 2,
        timeout: Union[float, httpx.Timeout] = 300,
        azure_ad_token: Optional[str] = None,
        azure_ad_token_provider: Optional[Callable] = None,
        aspeech: Optional[bool] = None,
        client=None,
        litellm_params: Optional[dict] = None,
        **kwargs,
    ) -> HttpxBinaryResponseContent:
        """
        Make a text-to-speech request to Azure Speech Services.

        Args:
            model: Azure Speech Services model/region (e.g., "eastus", "westus2")
            input: Text to convert to speech
            voice: Voice name (e.g., "en-US-AriaNeural", "en-US-JennyNeural")
            optional_params: Additional parameters
            api_key: Azure Speech Services API key
            api_base: Azure Speech Services endpoint URL
            max_retries: Maximum number of retries
            timeout: Request timeout
            aspeech: Whether this is an async request
            **kwargs: Additional keyword arguments

        Returns:
            HttpxBinaryResponseContent: Audio data response
        """
        if aspeech is not None and aspeech is True:
            return self.async_audio_speech(
                model=model,
                input=input,
                voice=voice,
                optional_params=optional_params,
                api_key=api_key,
                api_base=api_base,
                max_retries=max_retries,
                timeout=timeout,
                azure_ad_token=azure_ad_token,
                azure_ad_token_provider=azure_ad_token_provider,
                litellm_params=litellm_params,
                **kwargs,
            )  # type: ignore

        # Build the request
        headers, ssml_content, url = self._build_speech_request(
            input=input,
            voice=voice,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            model=model,
            azure_ad_token=azure_ad_token,
        )

        # Make the request
        sync_handler = _get_httpx_client()
        response = sync_handler.post(
            url=url,
            headers=headers,
            content=ssml_content,
            timeout=timeout,
        )

        if response.status_code != 200:
            raise AzureOpenAIError(
                status_code=response.status_code,
                message=f"Azure Speech Services request failed: {response.text}",
            )

        return HttpxBinaryResponseContent(response=response)

    async def async_audio_speech(
        self,
        model: str,
        input: str,
        voice: str,
        optional_params: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        max_retries: int,
        timeout: Union[float, httpx.Timeout],
        azure_ad_token: Optional[str] = None,
        azure_ad_token_provider: Optional[Callable] = None,
        litellm_params: Optional[dict] = None,
        **kwargs,
    ) -> HttpxBinaryResponseContent:
        """
        Make an async text-to-speech request to Azure Speech Services.
        """
        # Build the request
        headers, ssml_content, url = self._build_speech_request(
            input=input,
            voice=voice,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            model=model,
            azure_ad_token=azure_ad_token,
        )

        # Make the async request
        async_handler = get_async_httpx_client(llm_provider="azure_speech")
        response = await async_handler.post(
            url=url,
            headers=headers,
            content=ssml_content,
            timeout=timeout,
        )

        if response.status_code != 200:
            raise AzureOpenAIError(
                status_code=response.status_code,
                message=f"Azure Speech Services request failed: {response.text}",
            )

        return HttpxBinaryResponseContent(response=response)

    def _build_speech_request(
        self,
        input: str,
        voice: str,
        optional_params: dict,
        api_key: Optional[str],
        api_base: Optional[str],
        model: str,
        azure_ad_token: Optional[str] = None,
    ) -> tuple[dict, str, str]:
        """
        Build the Azure Speech Services request components.

        Returns:
            tuple: (headers, ssml_content, url)
        """
        # Determine API base URL
        if api_base is None:
            # Default to East US if no region specified
            region = model if model else "eastus"
            api_base = f"https://{region}.tts.speech.microsoft.com"

        # Clean up API base URL
        if api_base.endswith("/"):
            api_base = api_base[:-1]

        url = f"{api_base}/cognitiveservices/v1"

        # Build headers
        headers = {
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": optional_params.get("response_format", "audio-16khz-128kbitrate-mono-mp3"),
            "User-Agent": "LiteLLM",
        }

        # Add authentication
        if azure_ad_token:
            headers["Authorization"] = f"Bearer {azure_ad_token}"
        elif api_key:
            headers["Ocp-Apim-Subscription-Key"] = api_key
        else:
            raise AzureOpenAIError(
                status_code=401,
                message="Either azure_ad_token or api_key must be provided for Azure Speech Services",
            )

        # Build SSML content
        ssml_content = self._build_ssml(
            input=input,
            voice=voice,
            optional_params=optional_params,
        )

        return headers, ssml_content, url

    def _build_ssml(
        self,
        input: str,
        voice: str,
        optional_params: dict,
    ) -> str:
        """
        Build SSML (Speech Synthesis Markup Language) content.

        Args:
            input: Text to convert to speech
            voice: Voice name
            optional_params: Additional SSML parameters

        Returns:
            str: SSML content
        """
        # Check if input is already SSML
        if input.strip().startswith("<speak>") and input.strip().endswith("</speak>"):
            return input

        # Extract language code from voice name (e.g., "en-US-AriaNeural" -> "en-US")
        lang_parts = voice.split("-")
        if len(lang_parts) >= 2:
            lang_code = f"{lang_parts[0]}-{lang_parts[1]}"
        else:
            lang_code = "en-US"  # Default fallback

        # Build SSML
        ssml_parts = [
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{}">'.format(lang_code),
            '<voice name="{}">'.format(voice),
        ]

        # Add prosody settings if provided
        prosody_attrs = []
        if "speed" in optional_params:
            # Convert speed to rate (OpenAI uses 0.25-4.0, Azure uses percentages)
            speed = optional_params["speed"]
            if isinstance(speed, (int, float)):
                # Convert OpenAI speed format to Azure rate format
                rate_percent = int((speed - 1.0) * 100)
                if rate_percent > 0:
                    prosody_attrs.append(f'rate="+{rate_percent}%"')
                elif rate_percent < 0:
                    prosody_attrs.append(f'rate="{rate_percent}%"')

        if "pitch" in optional_params:
            pitch = optional_params["pitch"]
            prosody_attrs.append(f'pitch="{pitch}"')

        if prosody_attrs:
            ssml_parts.append(f'<prosody {" ".join(prosody_attrs)}>')
            ssml_parts.append(input)
            ssml_parts.append('</prosody>')
        else:
            ssml_parts.append(input)

        ssml_parts.extend([
            '</voice>',
            '</speak>'
        ])

        return "".join(ssml_parts)
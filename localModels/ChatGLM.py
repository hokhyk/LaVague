from dataclasses import Field
from typing import Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.ollama import Ollama


class ChatGLMEmbeddings(BaseEmbedding):
    model: str = Field(default='embedding-2', description="The ChatGlM model to use. embedding-2")
    api_key: str = Field(default=None, description="The ChatGLM API key.")
    reuse_client: bool = Field(default=True, description=(
        "Reuse the client between requests. When doing anything with large "
        "volumes of async API calls, setting this to false can improve stability."
    ),
                               )

    _client: Optional[Any] = PrivateAttr()
    def __init__(
            self,
            model: str = 'embedding-2',
            reuse_client: bool = True,
            api_key: Optional[str] = None,
            **kwargs: Any,
    )-> None:
        super().__init__(
            model=model,
            api_key=api_key,
            reuse_client=reuse_client,
            **kwargs,
        )
        self._client = None

    def _get_client(self) -> ZhipuAI:
        if not self.reuse_client :
            return ZhipuAI(api_key=self.api_key)

        if self._client is None:
            self._client = ZhipuAI(api_key=self.api_key)
        return self._client

    @classmethod
    def class_name(cls) -> str:
        return "ChatGLMEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.get_general_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        return self.get_general_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.get_general_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        return self.get_general_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.get_general_text_embedding(text)
            embeddings_list.append(embeddings)

        return embeddings_list

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        return self._get_text_embeddings(texts)

    def get_general_text_embedding(self, prompt: str) -> List[float]:
        response = self._get_client().embeddings.create(
            model=self.model, #填写需要调用的模型名称
            input=prompt,
        )
        return response.data[0].embedding

class ChatGLM(CustomLLM):
    num_output: int = DEFAULT_NUM_OUTPUTS
    context_window: int = Field(default=DEFAULT_CONTEXT_WINDOW,description="The maximum number of context tokens for the model.",gt=0,)
    model: str = Field(default=DEFAULT_MODEL, description="The ChatGlM model to use. glm-4 or glm-3-turbo")
    api_key: str = Field(default=None, description="The ChatGLM API key.")
    reuse_client: bool = Field(default=True, description=(
        "Reuse the client between requests. When doing anything with large "
        "volumes of async API calls, setting this to false can improve stability."
    ),
                               )

    _client: Optional[Any] = PrivateAttr()
    def __init__(
            self,
            model: str = DEFAULT_MODEL,
            reuse_client: bool = True,
            api_key: Optional[str] = None,
            **kwargs: Any,
    )-> None:
        super().__init__(
            model=model,
            api_key=api_key,
            reuse_client=reuse_client,
            **kwargs,
        )
        self._client = None

    def _get_client(self) -> ZhipuAI:
        if not self.reuse_client :
            return ZhipuAI(api_key=self.api_key)

        if self._client is None:
            self._client = ZhipuAI(api_key=self.api_key)
        return self._client

    @classmethod
    def class_name(cls) -> str:
        return "chatglm_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model,
        )

    def _chat(self, messages:List, stream=False) -> Any:
        response = self._get_client().chat.completions.create(
            model=self.model,  # 填写需要调用的模型名称
            messages=messages,
        )
        return response

    #@llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        message_dicts: List = to_message_dicts(messages)
        response = self._chat(message_dicts, stream=False)
        rsp = ChatResponse(
            message=ChatMessage(content=response.choices[0].message.content, role=MessageRole(response.choices[0].message.role),
                                additional_kwargs= {}),
            raw=response, additional_kwargs= get_additional_kwargs(response),
        )
        print(f"chat: {rsp} ")

        return rsp

    #@llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> CompletionResponseGen:
        response_txt = ""
        message_dicts: List = to_message_dicts(messages)
        response = self._chat(message_dicts, stream=True)
        for chunk in response:

            token = chunk.choices[0].delta.content
            response_txt += token
            yield ChatResponse(message=ChatMessage(content=response_txt,role=MessageRole(message.get("role")),
                                                   additional_kwargs={},), delta=token, raw=chunk,)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self._chat(messages, stream=False)

            rsp = CompletionResponse(text=str(response.choices[0].message.content),
                                     raw=response,
                                     additional_kwargs=get_additional_kwargs(response),)
        except Exception as e:
            print(f"complete: exception {e}")

        return rsp

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        response_txt = ""
        messages = [{"role": "user", "content": prompt}]
        response = self._chat(messages, stream=True)
        CompletionResponse(text=response.choices[0].message.content, delta=response.choices[0].message)

        for chunk in response.choices[0].message.content.splitlines():


            try:

                token = chunk+"\r\n"

            except:
                print(f"stream exception :{chunk}")
                continue


            response_txt += token

            yield CompletionResponse(text=response_txt, delta=token)
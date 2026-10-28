"""LangChain document loader for TranscriptFetch."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Literal

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from transcriptfetch import TranscriptFetch

from ._core import FetchOptions, iter_records, normalize_videos, validate


class TranscriptFetchLoader(BaseLoader):
    """Load transcripts of YouTube, TikTok and Instagram videos as Documents.

    Pass exactly one source: ``videos`` (a URL or ID, or a list of them),
    ``channel``, ``playlist`` or ``search``. Listings are capped at
    ``max_videos``.

    By default each video becomes one Document. With ``chunks=True`` each video
    is split server-side into pieces sized for an embedding model
    (``chunk_tokens``, ``chunk_overlap``), one Document per piece, with
    ``start`` / ``end`` seconds and a ``source`` link that opens the video at
    that moment, ready for citations.

    Setup:
        .. code-block:: bash

            pip install -U langchain-transcriptfetch
            export TRANSCRIPTFETCH_API_KEY="tf_live_..."

    Example:
        .. code-block:: python

            from langchain_transcriptfetch import TranscriptFetchLoader

            loader = TranscriptFetchLoader(channel="@3blue1brown", max_videos=10, chunks=True)
            docs = loader.load()
            docs[0].metadata["source"]  # https://www.youtube.com/watch?v=...&t=4s

    Args:
        videos: A video URL or ID, or a list of them (YouTube, TikTok,
            Instagram, or a direct audio or video file URL).
        channel: A YouTube channel (@handle, URL or ID), or a TikTok or
            Instagram profile URL.
        playlist: A YouTube or TikTok playlist URL or ID.
        search: A search query. YouTube unless ``search_platform`` says otherwise.
        search_platform: ``"youtube"`` (default), ``"tiktok"`` or ``"instagram"``.
        max_videos: Most videos to take from a channel, playlist or search.
        mode: ``"auto"`` (default) reads captions and transcribes the audio
            when a video has none; ``"captions"`` never transcribes audio, so
            captionless videos are skipped and cost nothing.
        chunks: Split each transcript into embedding-sized Documents.
        chunk_tokens: Target chunk size in tokens (100 to 2000).
        chunk_overlap: Tokens shared with the previous chunk (0 to 200).
        wait_for_audio: Wait for audio transcriptions to finish (default).
            If False, captionless videos are skipped.
        audio_timeout: Seconds to wait for one audio transcription.
        continue_on_failure: Log and skip a video that fails (default) instead
            of raising.
        api_key: TranscriptFetch API key; defaults to ``TRANSCRIPTFETCH_API_KEY``.
        base_url: API base URL override.

    Billing: a transcript or a listing page costs 1 credit, audio
    transcription 1 credit per started 5 minutes, and failed requests are
    free. Every account gets 50 free credits a month.
    """

    def __init__(
        self,
        videos: str | Iterable[str] | None = None,
        *,
        channel: str | None = None,
        playlist: str | None = None,
        search: str | None = None,
        search_platform: Literal["youtube", "tiktok", "instagram"] | None = None,
        max_videos: int = 25,
        mode: Literal["auto", "captions"] = "auto",
        chunks: bool = False,
        chunk_tokens: int = 500,
        chunk_overlap: int = 50,
        wait_for_audio: bool = True,
        audio_timeout: float = 900.0,
        continue_on_failure: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.options = FetchOptions(
            videos=normalize_videos(videos),
            channel=channel,
            playlist=playlist,
            search=search,
            search_platform=search_platform,
            max_videos=max_videos,
            mode=mode,
            chunks=chunks,
            chunk_tokens=chunk_tokens,
            chunk_overlap=chunk_overlap,
            wait_for_audio=wait_for_audio,
            audio_timeout=audio_timeout,
            continue_on_failure=continue_on_failure,
        )
        validate(self.options)
        self.api_key = api_key or os.environ.get("TRANSCRIPTFETCH_API_KEY")
        self.base_url = base_url

    def _client(self) -> TranscriptFetch:
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return TranscriptFetch(**kwargs)

    def lazy_load(self) -> Iterator[Document]:
        """Yield one Document per video, or per chunk with ``chunks=True``."""
        with self._client() as client:
            for record in iter_records(client, self.options):
                yield Document(page_content=record.text, metadata=record.metadata, id=record.id)

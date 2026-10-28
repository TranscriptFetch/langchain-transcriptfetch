"""LangChain integration for TranscriptFetch: transcripts of YouTube, TikTok
and Instagram videos as Documents, whole channels, playlists and searches."""

from importlib import metadata

from langchain_transcriptfetch.document_loaders import TranscriptFetchLoader

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""

__all__ = ["TranscriptFetchLoader", "__version__"]

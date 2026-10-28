# langchain-transcriptfetch

A LangChain document loader for [TranscriptFetch](https://transcriptfetch.com/?utm_source=github&utm_medium=integration&utm_campaign=langchain-transcriptfetch): transcripts of YouTube, TikTok and Instagram videos, from a single link to whole channels, playlists and searches, split server-side into embedding-sized chunks with a link to the moment each one starts.

```bash
pip install -U langchain-transcriptfetch
export TRANSCRIPTFETCH_API_KEY="tf_live_..."   # free key: https://transcriptfetch.com
```

## Load a few videos

```python
from langchain_transcriptfetch import TranscriptFetchLoader

docs = TranscriptFetchLoader([
    "https://www.youtube.com/watch?v=aircAruvnKk",
    "https://www.tiktok.com/@user/video/7428000000000000000",
]).load()
```

One Document per video. `metadata` holds `source` (the video URL), `title`, `channel`, `platform`, `duration`, `language` and `transcript_source` (`captions` or `audio`).

## A whole channel, chunked for RAG

```python
docs = TranscriptFetchLoader(channel="@3blue1brown", max_videos=20, chunks=True).load()

docs[0].metadata
# {'source': 'https://www.youtube.com/watch?v=aircAruvnKk&t=4s',
#  'video_url': 'https://www.youtube.com/watch?v=aircAruvnKk',
#  'chunk_index': 0, 'start': 4.22, 'end': 114.23, 'tokens': 400,
#  'title': 'But what is a neural network? | Deep learning chapter 1', ...}
```

With `chunks=True` the transcript is split by the API between caption lines (never mid-word), with a small overlap, so you can skip the text splitter. `source` links to the exact moment, which makes citations in answers clickable. Tune the size with `chunk_tokens` (100 to 2000, default 500) and `chunk_overlap` (0 to 200, default 50).

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

store = InMemoryVectorStore.from_documents(docs, OpenAIEmbeddings())
hits = store.similarity_search("what does a neuron hold?", k=3)
for h in hits:
    print(h.metadata["source"], h.page_content[:80])
```

## Sources

Pass exactly one:

| Argument | What it loads |
|---|---|
| `videos` | A URL or ID, or a list: YouTube, TikTok, Instagram, or a direct audio or video file URL |
| `channel` | A YouTube channel (`@handle`, URL or ID), or a TikTok or Instagram profile URL |
| `playlist` | A YouTube or TikTok playlist |
| `search` | A keyword search; YouTube unless `search_platform="tiktok"` or `"instagram"` |

Listings are capped at `max_videos` (default 25).

## Options

- `mode="auto"` (default) reads captions and transcribes the audio when a video has none, which most TikToks and Reels need. `mode="captions"` never transcribes, so captionless videos are skipped and cost nothing.
- `wait_for_audio=True` (default) waits for audio transcriptions to finish (`audio_timeout`, default 900 seconds each).
- `continue_on_failure=True` (default) logs and skips a video that fails instead of raising.
- `api_key` and `base_url` override the environment.

`lazy_load()` streams Documents as they arrive. Transcripts are fetched 50 at a time.

## Billing

A transcript or a listing page costs 1 credit, audio transcription 1 credit per started 5 minutes of audio, and failed requests are free. Every account gets 50 free credits a month, no card needed. See [pricing](https://transcriptfetch.com/pricing?utm_source=github&utm_medium=integration&utm_campaign=langchain-transcriptfetch).

## License

MIT

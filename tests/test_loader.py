from __future__ import annotations

import json

import httpx
import pytest
import respx

from langchain_transcriptfetch import TranscriptFetchLoader

BASE = "https://api.test"


def _item(vid: str, **extra) -> dict:
    return {
        "video_id": vid,
        "outcome": "ok",
        "url": f"https://www.youtube.com/watch?v={vid}",
        "title": f"Video {vid}",
        "channel": "Example",
        "duration": 120,
        "language": "en",
        "source": "captions",
        "text": f"text of {vid}",
        **extra,
    }


def _batch(results: list) -> dict:
    return {"ok": True, "data": {"kind": "transcript_batch", "results": results}}


def _loader(**kw) -> TranscriptFetchLoader:
    return TranscriptFetchLoader(api_key="tf_test", base_url=BASE, **kw)


@respx.mock
def test_one_document_per_video() -> None:
    route = respx.post(f"{BASE}/api/v2/transcripts/batch").mock(
        return_value=httpx.Response(200, json=_batch([_item("a"), _item("b")]))
    )
    docs = _loader(videos=["a", "b"]).load()
    assert [d.page_content for d in docs] == ["text of a", "text of b"]
    assert docs[0].metadata["source"] == "https://www.youtube.com/watch?v=a"
    assert docs[0].metadata["title"] == "Video a"
    assert docs[0].metadata["platform"] == "youtube"  # read off the URL
    assert docs[0].id == "a"
    body = json.loads(route.calls.last.request.content)
    assert body == {"videoIds": ["a", "b"], "mode": "auto"}


@respx.mock
def test_chunks_become_documents_with_deep_links() -> None:
    chunks = [
        {"index": 0, "text": "first", "start": 0.0, "end": 30.5, "tokens": 400,
         "url": "https://www.youtube.com/watch?v=a&t=0s"},
        {"index": 1, "text": "second", "start": 28.0, "end": 61.0, "tokens": 380,
         "url": "https://www.youtube.com/watch?v=a&t=28s"},
    ]
    route = respx.post(f"{BASE}/api/v2/transcripts/batch").mock(
        return_value=httpx.Response(200, json=_batch([_item("a", chunks=chunks)]))
    )
    docs = _loader(videos="a", chunks=True, chunk_tokens=800, chunk_overlap=80).load()
    assert [d.page_content for d in docs] == ["first", "second"]
    m = docs[1].metadata
    assert m["source"] == "https://www.youtube.com/watch?v=a&t=28s"
    assert m["video_url"] == "https://www.youtube.com/watch?v=a"
    assert (m["chunk_index"], m["start"], m["end"], m["tokens"]) == (1, 28.0, 61.0, 380)
    assert docs[1].id == "a:1"
    body = json.loads(route.calls.last.request.content)
    assert body["chunks"] == {"max_tokens": 800, "overlap_tokens": 80}


@respx.mock
def test_channel_is_listed_then_capped() -> None:
    respx.post(f"{BASE}/api/v2/transcripts/channel").mock(
        return_value=httpx.Response(
            200,
            json={"ok": True, "data": {"kind": "video_list", "videos": [
                {"videoId": v, "url": f"https://www.youtube.com/watch?v={v}"} for v in "abcd"
            ], "nextCursor": "more"}},
        )
    )
    batch = respx.post(f"{BASE}/api/v2/transcripts/batch").mock(
        return_value=httpx.Response(200, json=_batch([_item("a"), _item("b")]))
    )
    docs = _loader(channel="@example", max_videos=2).load()
    assert len(docs) == 2
    sent = json.loads(batch.calls.last.request.content)["videoIds"]
    assert sent == ["https://www.youtube.com/watch?v=a", "https://www.youtube.com/watch?v=b"]


@respx.mock
def test_audio_job_is_awaited() -> None:
    respx.post(f"{BASE}/api/v2/transcripts/batch").mock(
        return_value=httpx.Response(200, json=_batch([
            {"video_id": "tt", "outcome": "processing", "job_id": "job_1"},
        ]))
    )
    respx.get(f"{BASE}/api/v2/transcripts/jobs/job_1").mock(side_effect=[
        httpx.Response(202, json={"ok": True, "status": "processing", "job_id": "job_1",
                                  "data": None}),
        httpx.Response(200, json={"ok": True, "status": "completed", "data": {
            "kind": "transcript", "video_id": "tt", "platform": "tiktok",
            "url": "https://www.tiktok.com/@x/video/1", "source": "audio", "text": "spoken words",
        }}),
    ])
    loader = _loader(videos="https://www.tiktok.com/@x/video/1")
    loader.options.poll_interval = 0
    docs = loader.load()
    assert [d.page_content for d in docs] == ["spoken words"]
    assert docs[0].metadata["platform"] == "tiktok"
    assert docs[0].metadata["transcript_source"] == "audio"


@respx.mock
def test_failures_are_skipped_by_default_and_raised_on_request() -> None:
    respx.post(f"{BASE}/api/v2/transcripts/batch").mock(
        return_value=httpx.Response(200, json=_batch([
            _item("a"),
            {"video_id": "b", "outcome": "error",
             "error": {"code": "no_captions", "message": "No captions"}},
        ]))
    )
    assert [d.id for d in _loader(videos=["a", "b"], mode="captions").load()] == ["a"]
    with pytest.raises(Exception, match="no_captions"):
        _loader(videos=["a", "b"], continue_on_failure=False).load()


def test_needs_exactly_one_source() -> None:
    with pytest.raises(ValueError):
        TranscriptFetchLoader(api_key="k")
    with pytest.raises(ValueError):
        TranscriptFetchLoader("a", channel="@b", api_key="k")
    with pytest.raises(ValueError):
        TranscriptFetchLoader("a", chunks=True, chunk_tokens=50, api_key="k")

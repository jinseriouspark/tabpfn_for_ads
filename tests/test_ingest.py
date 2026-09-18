import io

import pandas as pd
from PIL import Image

from adlift.ingest import banner_to_creative, frame_from_csv, load_image
from adlift.llm import StubCopyLLM
from adlift.schema import validate


def _png_bytes():
    img = Image.new("RGB", (64, 32), (90, 40, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_load_image_bytes():
    raw, media_type = load_image(_png_bytes())
    assert media_type == "image/png" and len(raw) > 0


def test_load_image_reencodes_bmp(tmp_path):
    path = tmp_path / "b.bmp"
    Image.new("RGB", (8, 8)).save(path, format="BMP")
    raw, media_type = load_image(path)
    assert media_type == "image/png"


def test_banner_to_creative_stub():
    c = banner_to_creative(
        _png_bytes(),
        llm=StubCopyLLM(),
        campaign_id="c1",
        context={"device": "desktop", "vertical": "fintech"},
    )
    assert c.device == "desktop" and c.vertical == "fintech"
    assert c.extra["extractor"] == "stub"
    assert c.word_count > 0


def test_frame_from_csv_roundtrip(tmp_path, account):
    path = tmp_path / "acct.csv"
    account[[c for c in account.columns if not c.startswith("_")]].to_csv(path, index=False)
    frame = frame_from_csv(path)
    validate(frame)
    assert len(frame) == len(account)


def test_frame_from_csv_minimal_with_mapping(tmp_path):
    raw = pd.DataFrame(
        {
            "title": ["Cut costs 20%", "Discover savings", "Save now", "Unlock value"],
            "camp": ["a", "a", "b", "b"],
            "writer": ["Human", "LLM", "human", "llm"],
            "clicks": [10, 8, 12, 9],
            "impressions": [1000, 1000, 1000, 1000],
        }
    )
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    frame = frame_from_csv(
        path, column_map={"title": "headline", "camp": "campaign_id", "writer": "author"}
    )
    validate(frame)
    assert set(frame.author) == {"human", "llm"}
    assert frame.ctr.iloc[0] == 0.01

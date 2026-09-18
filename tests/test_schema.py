import pytest

from adlift.schema import AdCreative, creatives_to_frame, validate


def test_creative_derives_counts_and_ctr():
    c = AdCreative(
        headline="Cut churn by 30%.",
        body="Free trial.",
        cta_text="Start",
        clicks=12,
        impressions=1000,
    )
    assert c.word_count == 7
    assert c.char_count == len("Cut churn by 30%. Free trial. Start")
    assert c.ctr == pytest.approx(0.012)


def test_frame_has_all_columns():
    frame = creatives_to_frame(
        [
            AdCreative(headline="x", campaign_id="a", author="human", ctr=0.01),
            AdCreative(headline="y", campaign_id="a", author="llm", ctr=0.02),
        ]
    )
    validate(frame)
    assert "creative_id" in frame.columns and "week_index" in frame.columns


def test_validate_rejects_single_author():
    frame = creatives_to_frame(
        [AdCreative(headline="x", campaign_id="a", author="human", ctr=0.01)]
    )
    with pytest.raises(ValueError, match="both"):
        validate(frame)


def test_validate_rejects_perfect_confounding():
    frame = creatives_to_frame(
        [
            AdCreative(headline="x", campaign_id="a", author="human", ctr=0.01),
            AdCreative(headline="y", campaign_id="b", author="llm", ctr=0.02),
        ]
    )
    with pytest.raises(ValueError, match="perfectly confounded"):
        validate(frame)


def test_validate_rejects_bad_author():
    frame = creatives_to_frame(
        [AdCreative(headline="x", campaign_id="a", author="human", ctr=0.01)]
    )
    frame.loc[0, "author"] = "robot"
    with pytest.raises(ValueError, match="human"):
        validate(frame)

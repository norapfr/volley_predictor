import pytest
from pydantic import ValidationError

from src.schema import Competition, Gender, Match, Stage, Winner


def _base_match_kwargs(**overrides):
    kwargs = dict(
        match_id="test:1",
        date="2025-06-11",
        gender=Gender.men,
        competition="Volleyball Nations League",
        competition_id="VNL_MEN",
        stage=Stage.pool,
        team_a="USA",
        team_b="BRA",
        team_a_score=101,
        team_b_score=95,
        sets_a=3,
        sets_b=1,
        winner=Winner.team_a,
        source="kaggle_vnl",
        source_match_id="1",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_match_passes():
    match = Match(**_base_match_kwargs())
    assert match.winner == Winner.team_a
    assert match.neutral is True  # default


def test_winner_must_match_set_score():
    with pytest.raises(ValidationError):
        Match(**_base_match_kwargs(winner=Winner.team_b))  # A ganó 3-1 pero se afirma que ganó B


def test_invalid_set_score_rejected():
    with pytest.raises(ValidationError):
        Match(**_base_match_kwargs(sets_a=3, sets_b=3))  # el vóley no permite empate a sets


def test_team_cannot_play_itself():
    with pytest.raises(ValidationError):
        Match(**_base_match_kwargs(team_b="USA"))


def test_club_competition_rejected():
    with pytest.raises(ValidationError):
        Competition(
            competition_id="CLUB_1",
            name="Club World Championship",
            gender=Gender.men,
            level="club",
            is_club_competition=True,
        )

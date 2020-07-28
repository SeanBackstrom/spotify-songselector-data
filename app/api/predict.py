import logging
import random

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator
import json

log = logging.getLogger(__name__)
router = APIRouter()


# parse JSON from favorite songs
class Itemfav(BaseModel):
    """Use this data model to parse the request body JSON."""

    x1: list = Field(..., example=["songid", "songid", "songid"])

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])


'''
    @validator('x1')
    def x1_must_be_positive(cls, value):
        """Validate that x1 is a positive number."""
        assert value > 0, f'x1 == {value}, must be > 0'
        return value
'''


# send back predictions from favorite songs
@router.post('/predictfav')
async def predict(item: Itemfav):
    """Make random baseline predictions for classification problem."""
    X_new = item.to_df()
    log.info(X_new)

    # TODO: populate song ID's with real ID's
    song_ids_fav = []
    return song_ids_fav


# parse JSON from mood
class Itemmood(BaseModel):
    """Use this data model to parse the request body JSON."""

    x1: list = Field(..., example=[{"mood": "Danceability", "value": "high"},
                                   {"mood": "Energy", "value": "medium"},
                                   {"mood": "Speechiness", "value": "medium"},
                                   {"mood": "Acousticness", "value": "low"}])

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame(self)


'''
    @validator('x1')
    def x1_must_be_positive(cls, value):
        """Validate that x1 is a positive number."""
        assert value > 0, f'x1 == {value}, must be > 0'
        return value
'''


# send back predictions for mood
@router.post('/predictmood')
async def predict(item: Itemmood):
    """Make random baseline predictions for classification problem."""
    X_new = item.to_df()
    log.info(X_new)

    song_ids_mood = []
    return song_ids_mood

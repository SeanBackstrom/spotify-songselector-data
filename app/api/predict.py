import logging
from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field


log = logging.getLogger(__name__)
router = APIRouter()

 
# parse JSON from favorite songs
class Itemfav(BaseModel):
    """Use this data model to send the request body correctly,
     so data is received back (in JSON) based on the songs they selected."""
    
    songs: list = Field(example=["song id", "song id 2", "song id etc"])

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
async def predictfav(item: Itemfav):
    """(CURRENTLY IN TEST MODE) Make song predictions from favorite songs
     and return Song ID's in an array"""

    df = pd.read_csv("https://raw.githubusercontent.com/BW-Spotify-Song-Suggester-3/ds/master/large_song_data.csv", index_col=0)
    allpredlist = []
    for song in item.songs:
        track = song
        trackdf = df[df['track_id'].str.match(track)].iloc[0:3]
        song_ids_fav = trackdf[['suggested_id_1', 'suggested_id_2', 'suggested_id_3', 'suggested_id_4', 'suggested_id_5']].values.tolist()[0]
        for suggestion in song_ids_fav:
            allpredlist.append(suggestion)


    # item.songs is my list of songs
    # TODO: populate song ID's with real ID's

    return allpredlist


# parse JSON from mood
class Itemmood(BaseModel):
    """Use this data model to send the request body correctly,
     so data is received back (in JSON) based on the moods selected."""

    moods: list=Field(..., example=[{"mood": "Danceability", "value": "high"},
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
async def predictmood(item: Itemmood):
    """(CURRENTLY IN TEST MODE) Make song predictions from mood
     and return Song ID's in an array"""
    X_new = item.to_df()
    log.info(X_new)

    song_ids_mood = ["test pred song id", "test pred song id2"]
    return song_ids_mood

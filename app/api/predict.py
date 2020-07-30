import logging
from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field
from collections import Counter
import numpy as np

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
    """Make song predictions from favorite songs
     and return Song ID's in an array"""

    df = pd.read_csv("https://raw.githubusercontent.com/BW-Spotify-Song-Suggester-3/ds/master/large_song_data.csv", index_col=0)
    allpredlist = []
    for song in item.songs:
        track = song
        trackdf = df[df['track_id'].str.match(track)].iloc[0:3]
        song_ids_fav = trackdf[['suggested_id_1', 'suggested_id_2', 'suggested_id_3', 'suggested_id_4', 'suggested_id_5']].values.tolist()[0]
        for suggestion in song_ids_fav:
            allpredlist.append(suggestion)

    # Graph
    gdrive_file_id = '1grunsDCYl4Ro67EuDiFfcWXMBoAhrY6J'
    df = pd.read_csv(f'https://docs.google.com/uc?id={gdrive_file_id}&export=download', encoding='ISO-8859-1',index_col=0)

    df['tempo'] = df['tempo'] * 0.001

    #---------
    track_id = '52V5lBmgcqEZCtHjXtJHk9'

    ogtrack = df[df['track_id'].str.match(track_id)]
    sug1track = df[df['track_id'].str.match(ogtrack['suggested_id_1'].iloc[0])]
    sug2track = df[df['track_id'].str.match(ogtrack['suggested_id_2'].iloc[0])]
    sug3track = df[df['track_id'].str.match(ogtrack['suggested_id_3'].iloc[0])]
    sug4track = df[df['track_id'].str.match(ogtrack['suggested_id_4'].iloc[0])]
    sug5track = df[df['track_id'].str.match(ogtrack['suggested_id_5'].iloc[0])]

    #--------------

    import plotly.graph_objects as go
    labels = [ogtrack.iloc[-1]['track_name'], sug1track.iloc[-1]['track_name'], sug2track.iloc[-1]['track_name'], sug3track.iloc[-1]['track_name'],sug4track.iloc[-1]['track_name'], sug5track.iloc[-1]['track_name']]
    colors = ['#d62728','#1f77b4', '#e377c2', '#17becf',
            'rgb(115,115,115)', '#2ca02c']
    xlabels = ['Accousticness', 'Danceability', 'Energy',
                'Instrumentalness', 'Liveness', 'Speechiness',
                'Tempo', 'Valence']
    line_size = [3, 1, 1, 1, 1, 1]
    x_data = np.array([['Accousticness', 'Danceability', 'Energy',
                'Instrumentalness', 'Liveness', 'Speechiness',
                'Tempo', 'Valence']])
    y_data = np.array([
                    [ogtrack.iloc[-1]['acousticness'], ogtrack.iloc[-1]['danceability'], ogtrack.iloc[-1]['energy'], ogtrack.iloc[-1]['instrumentalness'], ogtrack.iloc[-1]['liveness'], ogtrack.iloc[-1]['speechiness'], ogtrack.iloc[-1]['tempo'], ogtrack.iloc[-1]['valence']],
                    [sug1track.iloc[-1]['acousticness'], sug1track.iloc[-1]['danceability'], sug1track.iloc[-1]['energy'], sug1track.iloc[-1]['instrumentalness'], sug1track.iloc[-1]['liveness'], sug1track.iloc[-1]['speechiness'], sug1track.iloc[-1]['tempo'], sug1track.iloc[-1]['valence']],
                    [sug2track.iloc[-1]['acousticness'], sug2track.iloc[-1]['danceability'], sug2track.iloc[-1]['energy'], sug2track.iloc[-1]['instrumentalness'], sug2track.iloc[-1]['liveness'], sug2track.iloc[-1]['speechiness'], sug2track.iloc[-1]['tempo'], sug2track.iloc[-1]['valence']],
                    [sug3track.iloc[-1]['acousticness'], sug3track.iloc[-1]['danceability'], sug3track.iloc[-1]['energy'], sug3track.iloc[-1]['instrumentalness'], sug3track.iloc[-1]['liveness'], sug3track.iloc[-1]['speechiness'], sug3track.iloc[-1]['tempo'], sug3track.iloc[-1]['valence']],
                    [sug4track.iloc[-1]['acousticness'], sug4track.iloc[-1]['danceability'], sug4track.iloc[-1]['energy'], sug4track.iloc[-1]['instrumentalness'], sug4track.iloc[-1]['liveness'], sug4track.iloc[-1]['speechiness'], sug4track.iloc[-1]['tempo'], sug4track.iloc[-1]['valence']],
                    [sug5track.iloc[-1]['acousticness'], sug5track.iloc[-1]['danceability'], sug5track.iloc[-1]['energy'], sug5track.iloc[-1]['instrumentalness'], sug5track.iloc[-1]['liveness'], sug5track.iloc[-1]['speechiness'], sug5track.iloc[-1]['tempo'], sug5track.iloc[-1]['valence']]
    ])
    fig = go.Figure()
    for i in range(0, 6):
        fig.add_trace(go.Scatter(x=x_data[0], y=y_data[i], mode='lines',
            name=labels[i],
            line=dict(color=colors[i], width=line_size[i]),
            connectgaps=True,
        ))
    fig.update_layout(
        xaxis=dict(
            ticktext=xlabels,
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='black',
            ),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False),
        autosize=False,
        margin=dict(
            autoexpand=False,
            l=100,
            r=20,
            t=110,
        ),
        showlegend=True,
        plot_bgcolor='white',
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=.9,
        xanchor="right",
        x=1.05)
    )
    annotations = []
    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.2,
                                xanchor='left', yanchor='bottom',
                                text='Song & Suggestions',
                                font=dict(family='Cursive',
                                            size=24,
                                            color='rgb(82, 82, 82)'),
                                showarrow=False))
    fig.update_layout(annotations=annotations)
    figtext =fig.to_json()

    figlist = []
    figlist.append(figtext)
    return allpredlist, figlist


# parse JSON from mood
class Itemmood(BaseModel):
    """Use this data model to send the request body correctly,
     so data is received back (in JSON) based on the moods selected."""

    moods: list=Field(..., example=[{"mood": "danceability", "value": "high"},
                                    {"mood": "energy", "value": "medium"},
                                    {"mood": "speechiness", "value": "medium"},
                                    {"mood": "acousticness", "value": "low"}])

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
    """Make song predictions from mood
     and return Song ID's in an array"""

    df2 = pd.read_csv("https://raw.githubusercontent.com/BW-Spotify-Song-Suggester-3/ds/master/large_song_data.csv", index_col=0)
    df2.columns = ['track_id', 'suggested_id_1', 'suggested_id_2', 'suggested_id_3',
        'suggested_id_4', 'suggested_id_5', 'danceability', 'liveness', 'valence',
        'energy', 'tempo', 'speechiness', 'instrument', 'acousticness']

    post_return = item.moods

    df2 = df2.drop(labels=['suggested_id_1', 'suggested_id_2', 'suggested_id_3',
       'suggested_id_4', 'suggested_id_5', 'instrument'], axis=1)

    # Make data even with json requests
    df2['danceability'] = df2['danceability'].str.lower()
    df2["danceability"]= df2["danceability"].replace("med", "medium") 
    df2['liveness'] = df2['liveness'].str.lower()
    df2["liveness"]= df2["liveness"].replace("med", "medium") 
    df2['valence'] = df2['valence'].str.lower()
    df2["valence"]= df2["valence"].replace("med", "medium") 
    df2['energy'] = df2['energy'].str.lower()
    df2["energy"]= df2["energy"].replace("med", "medium") 
    df2['tempo'] = df2['tempo'].str.lower()
    df2["tempo"]= df2["tempo"].replace("med", "medium") 
    df2['speechiness'] = df2['speechiness'].str.lower()
    df2["speechiness"]= df2["speechiness"].replace("med", "medium") 
    df2['acousticness'] = df2['acousticness'].str.lower()
    df2["acousticness"]= df2["acousticness"].replace("med", "medium") 

    # for loops to create and fill all data to be sent back
    moods = []
    values = []
    all_mood_names = ['danceability','liveness','valence','energy','tempo','speechiness','acousticness']
    for i in range(0, len(post_return)):

        mood = post_return[i]['mood']
        value = post_return[i]['value']

        moods.append(mood)
        values.append([value, value, value])

    remaining_moods = list((Counter(all_mood_names) - Counter(moods)).elements())

    for i in range(0, len(remaining_moods)):
        mood = remaining_moods[i]
        value = ["low", "medium", "high"]

        moods.append(mood)
        values.append(value)

    filterdf = df2.loc[
    ((df2[moods[0]] == values[0][0]) | (df2[moods[0]] == values[0][1]) | (df2[moods[0]] == values[0][2])) & 
    ((df2[moods[1]] == values[1][0]) | (df2[moods[1]] == values[1][1]) | (df2[moods[1]] == values[1][2])) & 
    ((df2[moods[2]] == values[2][0]) | (df2[moods[2]] == values[2][1]) | (df2[moods[2]] == values[2][2])) &
    ((df2[moods[3]] == values[3][0]) | (df2[moods[3]] == values[3][1]) | (df2[moods[3]] == values[3][2])) &
    ((df2[moods[4]] == values[4][0]) | (df2[moods[4]] == values[4][1]) | (df2[moods[4]] == values[4][2])) &
    ((df2[moods[5]] == values[5][0]) | (df2[moods[5]] == values[5][1]) | (df2[moods[5]] == values[5][2])) &
    ((df2[moods[6]] == values[6][0]) | (df2[moods[6]] == values[6][1]) | (df2[moods[6]] == values[6][2]))]

    final5 = filterdf[:5]['track_id']
    finallist = list(final5)
    """
    for song in item.songs:
        track = song
        trackdf = df[df['track_id'].str.match(track)].iloc[0:3]
        song_ids_fav = trackdf[['suggested_id_1', 'suggested_id_2', 'suggested_id_3', 'suggested_id_4', 'suggested_id_5']].values.tolist()[0]
        for suggestion in song_ids_fav:
            allpredlist.append(suggestion)

    song_ids_mood = ["test pred song id", "test pred song id2"]
    """
    return finallist

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api import predict, viz

app = FastAPI(
    title='Spotify Data API',
    description="""Send data in JSON, receive song predictions back in JSON. 
    Open either POST to find commands to send and receive data as well as the
     model for correctly requesting data.""",
    version='0.1',
    docs_url='/',
)

app.include_router(predict.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    uvicorn.run(app)

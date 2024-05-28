import os
import serpapi
from dotenv import load_dotenv


load_dotenv()


api_key = os.getenv("SERPAPI_KEY")


client = serpapi.Client(api_key=api_key)


result = client.search(
    q="Lion",
    engine="google",
    location="India",
    hl="en",
    gl="in",
    num=5
)


if 'organic_results' in result:
    for item in result['organic_results']:
        print(item['link'])
else:
    print("No links found.")

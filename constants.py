import os
apiKey = os.getenv('OPEN_AI_API_KEY')
os.environ['OPENAI_API_KEY'] = apiKey
serpapi_api_key = os.getenv('SERPAPI_API_KEY')
os.environ['SERPAPI_API_KEY'] = serpapi_api_key

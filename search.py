from atproto import Client
from secrets import USERNAME, PASSWORD

# Initialize client and authenticate
client = Client()

# You'll need to provide your Bluesky credentials
# For app passwords: https://bsky.app/settings/app-passwords
# Save these to secrets.py which is in the .gitignore file

if USERNAME and PASSWORD:
    client.login(USERNAME, PASSWORD)
else:
    raise ValueError("Authentication required. Please provide USERNAME and PASSWORD.")

# for more search parameters, see https://docs.bsky.app/docs/api/app-bsky-feed-search-posts
search_params = {
    "q": "vaccine side effects",
    "limit": 10,   
    "sort": 'top' # top or latest, default is latest
}

# Perform search
response = client.app.bsky.feed.search_posts(params=search_params)

# Simplify the structure
results = []
for p in response.posts:
    record = p.record
    author = p.author
    results.append(
        {
            "uri": p.uri,
            "handle": author.handle,
            "displayName": author.display_name,
            "text": record.text,
            "createdAt": record.created_at,
        }
    )

for i, p in enumerate(results, start=1):
    print(f"{i}. @{p['handle']} ({p['createdAt']})")
    print(f"   {p['text']}\n")
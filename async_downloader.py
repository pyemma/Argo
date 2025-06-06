import ray
import asyncio
import aiohttp
from typing import List
import time

# Initialize Ray
ray.init()

@ray.remote
class DownloaderActor:
    def __init__(self, actor_id: int):
        self.actor_id = actor_id
        self.session = None
    
    async def _init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def _close_session(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def download_url(self, url: str) -> dict:
        await self._init_session()
        try:
            async with self.session.get(url) as response:
                data = await response.text()
                return {
                    "url": url,
                    "status": response.status,
                    "data": data,
                    "actor_id": self.actor_id
                }
        except Exception as e:
            return {
                "url": url,
                "status": "error",
                "error": str(e),
                "actor_id": self.actor_id
            }
    
    def download_batch(self, urls: List[str]) -> List[dict]:
        # Create event loop for this actor
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create tasks for all URLs
            tasks = [self.download_url(url) for url in urls]
            # Run all tasks concurrently
            results = loop.run_until_complete(asyncio.gather(*tasks))
            return results
        finally:
            # Clean up
            loop.run_until_complete(self._close_session())
            loop.close()

def main():
    # Example URLs to download
    urls = [
        "https://api.github.com/users/octocat",
        "https://api.github.com/users/microsoft",
        "https://api.github.com/users/google",
        "https://api.github.com/users/apple",
    ]
    
    # Create multiple actors
    num_actors = 2
    actors = [DownloaderActor.remote(i) for i in range(num_actors)]
    
    # Split URLs among actors
    urls_per_actor = len(urls) // num_actors
    url_batches = [urls[i:i + urls_per_actor] for i in range(0, len(urls), urls_per_actor)]
    
    # Start time
    start_time = time.time()
    
    # Submit tasks to actors
    futures = [actor.download_batch.remote(batch) for actor, batch in zip(actors, url_batches)]
    
    # Get results
    results = ray.get(futures)
    
    # End time
    end_time = time.time()
    
    # Print results
    print(f"Download completed in {end_time - start_time:.2f} seconds")
    for actor_results in results:
        for result in actor_results:
            print(f"Actor {result['actor_id']} downloaded {result['url']} with status {result['status']}")

if __name__ == "__main__":
    main() 
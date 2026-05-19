import asyncio
import os
import random
from dataclasses import dataclass

import aiohttp


@dataclass(frozen=True)
class SeederConfig:
    api_base_url: str = os.getenv("API_BASE_URL", "http://api:8000")
    concurrency: int = int(os.getenv("SEEDER_CONCURRENCY", "4"))
    sleep_seconds: float = float(os.getenv("SEEDER_SLEEP_SECONDS", "2"))


POSITIVE_TEXTS = [
    "I love this product, it works perfectly.",
    "The experience was excellent and very reliable.",
    "This platform is fast, stable and useful.",
    "The model gives accurate and helpful results.",
    "The deployment pipeline is working beautifully.",
    "This service is impressive and easy to use.",
]

NEGATIVE_TEXTS = [
    "I hate this product, it is slow and unreliable.",
    "The experience was terrible and disappointing.",
    "This platform is unstable and not useful.",
    "The model gives poor and inaccurate results.",
    "The deployment pipeline has many problems.",
    "This service is frustrating and hard to use.",
]


def build_sample() -> dict[str, str]:
    if random.random() >= 0.5:
        return {
            "text": random.choice(POSITIVE_TEXTS),
            "expected_label": "POSITIVE",
        }

    return {
        "text": random.choice(NEGATIVE_TEXTS),
        "expected_label": "NEGATIVE",
    }


async def wait_until_api_is_ready(
    session: aiohttp.ClientSession,
    base_url: str,
) -> None:
    while True:
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    return
        except Exception:
            pass

        await asyncio.sleep(3)


async def send_prediction(
    session: aiohttp.ClientSession,
    config: SeederConfig,
    worker_id: int,
) -> None:
    endpoint = f"{config.api_base_url}/predict"

    while True:
        payload = build_sample()

        try:
            async with session.post(endpoint, json=payload) as response:
                body = await response.text()

                print(
                    {
                        "worker": worker_id,
                        "status": response.status,
                        "expected_label": payload["expected_label"],
                        "body": body[:250],
                    },
                    flush=True,
                )

        except Exception as exc:
            print(
                {
                    "worker": worker_id,
                    "error": str(exc),
                },
                flush=True,
            )

        await asyncio.sleep(config.sleep_seconds)


async def main() -> None:
    config = SeederConfig()
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await wait_until_api_is_ready(session, config.api_base_url)

        tasks = [
            send_prediction(
                session=session,
                config=config,
                worker_id=index,
            )
            for index in range(config.concurrency)
        ]

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
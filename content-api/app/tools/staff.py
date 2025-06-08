import httpx
from bs4 import BeautifulSoup

URL = "https://finki.ukim.mk/mk/staff-list/kadar/nastaven-kadar"


async def get_staff() -> str | list[str]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(URL)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            staff = soup.select("h2 > a")

            if not staff:
                return "No staff found."

            result: list[str] = []

            for person in staff:
                name = person.get_text(strip=True)
                result.append(name)
            return result

    except Exception as e:
        return f"An error occurred: {e!s}"

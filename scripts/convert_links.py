import json


def convert_links(data: list) -> str:
    links = []

    for link in data:
        link_id = link["id"]
        name = link["name"]
        description = link["description"]
        url = link["url"]
        user_id = link["userId"]
        created_at = link["createdAt"]
        updated_at = link["updatedAt"]

        link = f"""INSERT INTO link (id, name, description, url, user_id, created_at, updated_at)
        VALUES ('{link_id}', '{name}', '{description}', '{url}', '{user_id}', '{created_at}', '{updated_at}' );
        """

        links.append(link)

    return "\n".join(links)


if __name__ == "__main__":
    with open("./scripts/links.json", encoding="utf-8") as f:
        links = json.load(f)

    with open("./scripts/links.sql", "w", encoding="utf-8") as f:
        f.write(convert_links(links))

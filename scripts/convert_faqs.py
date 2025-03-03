import json


def convert_links(data: list) -> str:
    links = {}

    for link in data:
        name = link["name"]
        url = link["url"]

        links[name] = url

    return json.dumps(links)


def convert_questions(data: list) -> str:
    queries = []

    for question in data:
        question_id = question["id"]
        name = question["name"]
        content = question["content"]
        user_id = question["userId"]
        links = convert_links(question["links"])
        created_at = question["createdAt"]
        updated_at = question["updatedAt"]

        query = f"""INSERT INTO question (id, name, content, user_id, links, created_at, updated_at)
        VALUES ('{question_id}', '{name}', '{content}', '{user_id}', '{links}', '{created_at}', '{updated_at}' );
        """

        queries.append(query)

    return "\n".join(queries)


if __name__ == "__main__":
    with open("./scripts/faqs.json", encoding="utf-8") as f:
        faq = json.load(f)

    with open("./scripts/faqs.sql", "w", encoding="utf-8") as f:
        f.write(convert_questions(faq))

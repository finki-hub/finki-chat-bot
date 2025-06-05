from app.schemas.questions import QuestionSchema

DEFAULT_SYSTEM_PROMPT = (
    "Ти си љубезен асистент и експерт за сумаризации кој одговара на прашања поврзани со ФИНКИ. "
    "Секогаш одговарај на македонски јазик. Дај јасни, точни и концизни одговори на сите прашања "
    "што се однесуваат на универзитетот, факултетот, студиите, административните процеси и слично. "
    "Користи само информации од дадените извори. Ако е можно, наведи од каде е информацијата. "
    "Ако не знаеш одговор или прашањето не е поврзано со ФИНКИ, кажи дека не си сигурен и препорачај "
    "корисникот да се обрати во Студентската служба на ФИНКИ."
)


def build_context(questions: list[QuestionSchema]) -> str:
    """
    Build a context string from a list of questions.
    """
    return "\n".join(f"- Наслов: {q.name}\n  Содржина: {q.content}" for q in questions)


def build_user_prompt(context: str, prompt: str) -> str:
    """
    Build a user prompt for the LLM with the prompt and context.
    """
    return f"""\
Контекст:
{context}

Прашање:
{prompt}

Одговор:"""


def stitch_system_user(system: str, user_prompt: str) -> str:
    """
    Stitch the system prompt and user prompt into a single string for the LLM.
    """
    return f"<|system|> {system}\n\n<|user|> {user_prompt}\n\n<|assistant|>"

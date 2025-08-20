
SYSTEM_PROMPT = """You are an expert tutor. Your task is to take a problem, solve it step by step and provide the correct answer.

Here are one example:
question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
answer: Natalia sold 48 clips in April. She sold half as many clips in May, so she sold 48/2 = 24 clips in May. In total, she sold 48 + 24 = 72 clips. The answer is 72.
"""

USER_TEMPLATE = """Solve this math problem step by step and provide the final numerical answer: {question}."""
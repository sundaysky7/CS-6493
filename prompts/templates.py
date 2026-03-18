"""Prompt-template factory.

中文：集中管理所有提示方法模板，保证实验可复现且格式统一。
English: Centralized prompt templates for consistent and reproducible experiments.
"""

from __future__ import annotations


def generate_prompt(question: str, method: str) -> str:
    """
    根据提示方法名称生成格式化提示词。
    Generate a formatted prompt by method name.

    中文：
    - standard
    - cot
    - self_refine_stage1
    - self_refine_stage2::<preliminary_answer>
    - least_to_most

    English:
    Supported methods are the same as listed above.

    :param question: 数学问题文本 / math question text
    :param method: 提示方法名 / prompt method
    :return: 格式化提示词 / formatted prompt
    """
    method = method.lower().strip()

    if method == "standard":
        return f"Question: {question}\nAnswer:\n"

    if method == "cot":
        return (
            f"Question: {question}\n"
            "Let's solve this problem step by step.\n"
            "Step 1:\n"
            "Step 2:\n"
            "Final Answer:\n"
        )

    if method == "self_refine_stage1":
        return f"Question: {question}\nPreliminary Answer:\n"

    if method.startswith("self_refine_stage2"):
        marker = "::"
        if marker not in method:
            raise ValueError(
                "self_refine_stage2 method must include preliminary answer as "
                "'self_refine_stage2::<answer>'"
            )
        preliminary = method.split(marker, maxsplit=1)[1].strip()
        return (
            f"Question: {question}\n"
            f"Preliminary Answer: {preliminary}\n"
            "Please check and refine this answer to correct any errors.\n"
            "Refined Answer:\n"
        )

    if method == "least_to_most":
        return (
            f"Question: {question}\n"
            "Let's break this problem into simpler sub-problems.\n"
            "Sub-problem 1:\n"
            "Answer to Sub-problem 1:\n"
            "Sub-problem 2:\n"
            "Answer to Sub-problem 2:\n"
            "Final Answer to the original question:\n"
        )

    raise ValueError(f"Unsupported prompt method: {method}")

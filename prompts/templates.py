"""提示词模板模块 / Prompt template module.

中文（版本1）：
    按方法名生成统一格式提示词，确保实验中不同提示策略可复用、可比较。

English (Version 2):
    Provides method-specific prompt templates with a unified API, enabling
    fair comparison across prompting strategies.
"""

from __future__ import annotations


def generate_prompt(question: str, method: str) -> str:
    """
    中文（版本1）：
        根据方法名拼接模板。Self-Refine 的第二阶段需要传入初稿答案，
        本实现使用 `self_refine_stage2::<preliminary_answer>` 作为参数约定。

    English (Version 2):
        Build prompt text by method name. For Self-Refine stage-2, this function
        expects the convention `self_refine_stage2::<preliminary_answer>`.

    :param question: 数学问题文本 / math question text.
    :param method: 提示方法名称 / method name.
    :return: 格式化提示词 / formatted prompt.
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

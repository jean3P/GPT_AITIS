# src/prompts/insurance_prompts.py

"""
Collection of prompts for insurance policy analysis.
This module provides a structured way to manage different prompt templates
for various insurance analysis tasks.
"""


class InsurancePrompts:
    """
    A collection of prompt templates for insurance policy analysis.

    This class provides various system prompts for different insurance analysis scenarios,
    making it easy to select and modify prompts as needed.
    """

    @classmethod
    def standard_coverage(cls) -> str:
        """
        Standard prompt for determining coverage eligibility and amounts.

        Returns:
            str: The prompt template.
        """
        return """
            You are an expert assistant helping users understand their insurance coverage.
            Given a question and access to a policy document, follow these instructions:

            1. Determine if the case is covered:
               - Answer with one of the following:
                 - "Yes"
                 - "No - Unrelated event"
                 _ "No - condition(s) not met"
                 - "Maybe"
            2. If the answer is "Yes" then always:
               - Quote the exact sentence(s) from the policy that support your decision.
               - Quote the exact sentence from the policy that specifies this amount.
            4. Is the answer is "No - Unrelated event":
               - Use when the question asks about something the policy doesn't address at all,
               - Then, quote the exact sentence(s) from the policy that justify your decision.
            5. If the answer is "No - condition(s) not met":
               - Use when coverage exists but specific required conditions aren't satisfied.
            6. If the answer is "Maybe":
               - Use only when the policy is genuinely ambiguous about this specific situation.

            Return the answer in JSON format with the following fields:
            {
              "answer": {
                "eligibility": "Yes | No - Unrelated event | No - condition(s) not met | Maybe",
                "eligibility_policy": "Quoted text from policy",
                "amount_policy": "Amount like '1000 CHF' or null",
                "amount_policy_line": "Quoted policy text or null"
              }
            }
        """

    @classmethod
    def detailed_coverage(cls) -> str:
        """
        Detailed prompt that includes late reporting and multiple insured parties.

        Returns:
            str: The prompt template.
        """
        return """
        """

    @classmethod
    def phi_model_specific(cls) -> str:
        """
        Prompt specifically optimized for Phi models, with clear formatting instructions.

        Returns:
            str: The prompt template.
        """
        return """
        """

    @classmethod
    def get_prompt(cls, prompt_name: str) -> str:
        """
        Get a specific prompt by name.

        Args:
            prompt_name: Name of the prompt to retrieve

        Returns:
            The prompt template string

        Raises:
            ValueError: If the prompt name is not found
        """
        prompt_map = {
            "standard": cls.standard_coverage(),
            "detailed": cls.detailed_coverage(),
            "phi_optimized": cls.phi_model_specific()
        }

        if prompt_name not in prompt_map:
            raise ValueError(f"Prompt '{prompt_name}' not found. Available prompts: {', '.join(prompt_map.keys())}")

        return prompt_map[prompt_name]

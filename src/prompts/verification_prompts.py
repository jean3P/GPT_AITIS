# src/prompts/insurance_prompts.py

"""
Verification prompts for evaluating and correcting insurance policy analysis outputs.
"""

class VerificationPrompts:
    """
    Collection of prompts for verifying and correcting insurance analysis results.
    """

    @classmethod
    def verification_prompt_phi4(cls) -> str:
        """
        Verification prompt optimized for Phi-4 model.
        """
        return (
            # ROLE DEFINITION
            "You are an insurance policy verification expert. Your task is to review and potentially correct "
            "previous insurance coverage determinations.\n\n"
            
            # INPUT SECTIONS
            "ORIGINAL QUESTION:\n"
            "{{USER_QUESTION}}\n\n"
            
            "POLICY CONTEXT PROVIDED:\n"
            "{{POLICY_CONTEXT}}\n\n"
            
            "PREVIOUS ANALYSIS RESULT:\n"
            "{{PREVIOUS_RESULT}}\n\n"
            
            # VERIFICATION TASKS
            "VERIFICATION TASKS:\n"
            "1. Check if the eligibility decision is correct based on the policy context\n"
            "2. Verify that ALL quoted text actually exists in the policy context (character-for-character)\n"
            "3. Ensure the logic connecting the question to the policy text is sound\n"
            "4. Confirm that amount information (if any) is accurately extracted\n\n"
            
            # CRITICAL RULE
            "CRITICAL VERIFICATION RULE:\n"
            "If ANY quoted text in outcome_justification or payment_justification does NOT appear "
            "VERBATIM in the policy context above, you MUST:\n"
            "- Set outcome_justification to empty string \"\"\n"
            "- Set payment_justification to null\n"
            "- Change eligibility to \"No - Unrelated event\" if no relevant text exists\n\n"
            
            # DECISION LOGIC
            "DECISION PROCESS:\n"
            "1. If the previous result is CORRECT:\n"
            "   - Keep all fields exactly as they are\n"
            "   - Set verification_status to 'confirmed'\n\n"
            
            "2. If the previous result has ERRORS:\n"
            "   - Correct the eligibility if wrong\n"
            "   - Fix any misquoted text (must be verbatim from policy)\n"
            "   - Update justifications with correct quotes\n"
            "   - Set verification_status to 'corrected'\n\n"
            
            "3. Common errors to check:\n"
            "   - Eligibility says 'Yes' but no coverage text exists\n"
            "   - Quoted text not found in policy context\n"
            "   - Wrong eligibility category selected\n"
            "   - Hallucinated or paraphrased quotes\n\n"
            
            # OUTPUT FORMAT - MATCHING ORIGINAL
            "OUTPUT FORMAT (JSON only, matching original format exactly):\n"
            "{\n"
            "  \"answer\": {\n"
            "    \"eligibility\": \"Yes|No - Unrelated event|No - condition(s) not met\",\n"
            "    \"outcome_justification\": \"<verbatim policy text or empty string>\",\n"
            "    \"payment_justification\": \"<verbatim amount text or null>\"\n"
            "  },\n"
            "  \"verification\": {\n"
            "    \"status\": \"confirmed|corrected\",\n"
            "    \"changes_made\": \"<brief description of corrections or 'none'>\"\n"
            "  }\n"
            "}\n\n"
            
            # STRICT RULES
            "CRITICAL RULES:\n"
            "- ONLY use text that appears verbatim in POLICY CONTEXT PROVIDED above\n"
            "- Never create new sentences or paraphrase\n"
            "- If you cannot find supporting text in the context, use empty string \"\"\n"
            "- For payment_justification: use null (not \"null\" string) when no amount exists\n"
            "- Output must start with '{' and end with '}'\n"
            "- No markdown, no commentary, just the JSON\n"
        )

    @classmethod
    def verification_prompt_qwen(cls) -> str:
        """
        Verification prompt optimized for Qwen models.
        """
        return (
            # SYSTEM ROLE
            "SYSTEM: Insurance verification expert - reviews and corrects coverage determinations\n"
            "CONSTRAINT: Output only text found verbatim in POLICY_CONTEXT\n\n"
            
            # INPUTS
            "INPUTS FOR VERIFICATION:\n"
            "=====================================\n"
            "USER_QUESTION:\n"
            "{{USER_QUESTION}}\n\n"
            
            "POLICY_CONTEXT:\n"
            "{{POLICY_CONTEXT}}\n\n"
            
            "PREVIOUS_RESULT:\n"
            "{{PREVIOUS_RESULT}}\n"
            "=====================================\n\n"
            
            # VERIFICATION CHECKLIST
            "VERIFICATION CHECKLIST:\n"
            "□ Eligibility matches policy content?\n"
            "□ ALL quotes exist verbatim in context?\n"
            "□ Logic from question→policy is valid?\n"
            "□ Amount correctly extracted?\n"
            "□ No hallucinated content?\n\n"
            
            # CRITICAL VERIFICATION RULE
            "CRITICAL: If quoted text is NOT in policy context:\n"
            "→ Set outcome_justification = \"\"\n"
            "→ Set payment_justification = null\n"
            "→ Correct eligibility accordingly\n\n"
            
            # ERROR PATTERNS
            "COMMON ERROR PATTERNS:\n"
            "• 'Yes' with no supporting coverage text → Change to 'No - Unrelated event'\n"
            "• Paraphrased quotes → Replace with exact text or \"\"\n"
            "• Wrong eligibility category → Correct based on policy\n"
            "• Created text not in policy → Remove and use \"\"\n\n"
            
            # CORRECTION WORKFLOW
            "WORKFLOW:\n"
            "1. Parse previous result\n"
            "2. Verify each field against policy\n"
            "3. If correct → status='confirmed'\n"
            "4. If errors → fix and status='corrected'\n\n"
            
            # OUTPUT SCHEMA - EXACT FORMAT
            "REQUIRED OUTPUT (exact format):\n"
            "{\n"
            "  \"answer\": {\n"
            "    \"eligibility\": \"[exact_value]\",\n"
            "    \"outcome_justification\": \"[exact_policy_text_or_empty]\",\n"
            "    \"payment_justification\": \"[exact_amount_text_or_null]\"\n"
            "  },\n"
            "  \"verification\": {\n"
            "    \"status\": \"confirmed|corrected\",\n"
            "    \"changes_made\": \"[what_was_fixed_or_none]\"\n"
            "  }\n"
            "}\n\n"
            
            # ENFORCEMENT
            "FORBIDDEN:\n"
            "× Creating new sentences\n"
            "× Using [...] or ellipsis\n"
            "× Paraphrasing policy text\n"
            "× Adding explanations outside JSON\n"
            "× Using string \"null\" instead of null value\n"
        )

    @classmethod
    def get_verification_prompt(cls, model_type: str) -> str:
        """
        Get appropriate verification prompt based on model type.

        Args:
            model_type: Either 'phi4' or 'qwen'

        Returns:
            Verification prompt string
        """
        if 'phi' in model_type.lower():
            return cls.verification_prompt_phi4()
        elif 'qwen' in model_type.lower():
            return cls.verification_prompt_qwen()
        else:
            # Default to phi4 style
            return cls.verification_prompt_phi4()

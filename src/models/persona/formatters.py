# src/models/persona/formatters.py

"""
Formatting functions for persona information.
Converts extracted persona information into text for prompts.
"""
from typing import Dict, Any


def format_persona_text(personas_info: Dict[str, Any]) -> str:
    """
    Format persona information for inclusion in the prompt, with enhanced focus
    on the affected person and their location during the event.

    Args:
        personas_info: Dictionary with persona information

    Returns:
        Formatted persona text
    """
    persona_text = "IMPORTANT PERSONA INFORMATION FROM THE QUESTION:\n"

    try:
        # Extract persona details
        policy_user = personas_info["personas"]["policy_user"]
        affected_person = personas_info["personas"].get("affected_person", f"{policy_user} (inferred)")
        location = personas_info["personas"].get("location", "unspecified location")
        is_abroad = personas_info["personas"].get("is_abroad", False)
        relationship_to_policyholder = personas_info["personas"].get("relationship_to_policyholder",
                                                                     "self (inferred)")
        is_affected_covered = personas_info["personas"].get("is_affected_covered", True)
        mentioned_people = personas_info["personas"]["mentioned_people"]
        total_count = personas_info["personas"]["total_count"]
        claimant_count = personas_info["personas"]["claimant_count"]
        relationship = personas_info["personas"]["relationship"]

        # Build detailed persona text
        persona_text += f"- Primary policy user/claimant: {policy_user}\n"
        persona_text += f"- Person who experienced the event/accident: {affected_person}\n"
        persona_text += f"- Location where the event occurred: {location}\n"
        persona_text += f"- Event occurred abroad: {'Yes' if is_abroad else 'No'}\n"
        persona_text += f"- Relationship to policyholder: {relationship_to_policyholder}\n"

        # Add coverage information for affected person
        if is_affected_covered:
            persona_text += f"- The affected person is likely covered under this policy\n"
        else:
            persona_text += f"- The affected person may NOT be covered under this policy\n"

        persona_text += f"- Other people mentioned (not policy users): {mentioned_people}\n"
        persona_text += f"- Total number of people mentioned: {total_count}\n"
        persona_text += f"- Number of people actually claiming/covered: {claimant_count}\n"
        persona_text += f"- Relationships between all people: {relationship}\n\n"

        # Add location-specific guidance
        if is_abroad:
            persona_text += "When determining coverage, check if the policy covers events occurring abroad and any special conditions or exclusions for international coverage.\n"

        if "airport" in location.lower():
            persona_text += "When determining coverage, focus on travel-related provisions, especially those related to baggage, delays, or airport incidents.\n"

        elif "hotel" in location.lower() or "resort" in location.lower():
            persona_text += "When determining coverage, check if accommodation-related incidents are covered and under what conditions.\n"

        elif "hospital" in location.lower() or "medical" in location.lower():
            persona_text += "When determining coverage, focus on medical coverage provisions, including emergency treatment and hospitalization.\n"

        elif "home" in location.lower() or "domestic" in location.lower():
            persona_text += "When determining coverage, verify if domestic incidents are covered, as some travel policies only apply when traveling.\n"

        # Add person-specific guidance
        if relationship_to_policyholder == "self":
            persona_text += "Also check provisions that apply to the policyholder directly.\n"
        elif relationship_to_policyholder in ["spouse/partner", "child/dependent"]:
            persona_text += "Also verify if family members/dependents are covered and under what conditions.\n"
        else:
            persona_text += "Also carefully verify if non-family members are covered under this policy.\n"

        persona_text += "Take into account all of these factors when assessing eligibility for coverage.\n"
    except (KeyError, TypeError):
        persona_text += "Unable to extract detailed persona information. Consider who is actually making the claim vs. who experienced the event vs. who is just mentioned and where the event occurred.\n"

    return persona_text

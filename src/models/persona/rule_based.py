# src/models/persona/rule_based.py

"""
Rule-based persona extraction.
Extracts persona information using pattern matching and heuristics.
"""
import re
import logging
from typing import Dict, Any, List, Tuple, Optional

from models.persona.location import LocationDetector

logger = logging.getLogger(__name__)


class RuleBasedExtractor:
    """
    Rule-based extractor for persona information.
    Uses regex patterns and heuristics to extract information.
    """

    def __init__(self):
        """Initialize the rule-based extractor."""
        self.location_detector = LocationDetector()

    def extract(self, question: str) -> Dict[str, Any]:
        """
        Extract persona information using rule-based approach.

        Args:
            question: The insurance query to analyze

        Returns:
            Dictionary with persona information
        """
        # Initialize defaults
        policy_user = None
        affected_person = None
        mentioned_people = []
        relationship = None
        total_count = 1
        claimant_count = 1

        # Extract family relationships
        found_relationships, family_mentioned = self._extract_family_relationships(question)
        mentioned_people.extend(family_mentioned)

        # Extract non-family relationships
        other_relationships, other_mentioned = self._extract_non_family_relationships(question)
        found_relationships.extend(other_relationships)
        mentioned_people.extend(other_mentioned)

        # Try to identify who experienced the event
        affected_person = self._identify_affected_person(question)

        # Extract location information
        location, is_abroad = self.location_detector.identify_location(question)

        # Handle special cases
        (special_policy_user,
         special_mentioned,
         special_relationship,
         special_total_count,
         special_claimant_count,
         special_affected_person) = self._handle_special_case_personas(question, found_relationships)

        if special_policy_user:
            policy_user = special_policy_user
        if special_mentioned:
            mentioned_people = special_mentioned
        if special_relationship:
            relationship = special_relationship
        if special_total_count:
            total_count = special_total_count
        if special_claimant_count:
            claimant_count = special_claimant_count
        if special_affected_person:
            affected_person = special_affected_person

        # If not a special case, determine defaults
        if not policy_user:
            (policy_user,
             relationship,
             default_total,
             default_claimant,
             relationship) = self._determine_default_persona_info(
                question,
                found_relationships
            )

            # Only use defaults if not set by special cases
            if not special_total_count:
                total_count = default_total
            if not special_claimant_count:
                claimant_count = default_claimant

        # If there are specific people mentioned, update total count
        if mentioned_people:
            mentioned_count = len(mentioned_people)
            total_count = max(total_count, mentioned_count)

        # If affected_person is still None, default to policy_user
        if not affected_person:
            affected_person = f"{policy_user} (inferred)"

        # Determine relationship to policyholder
        relationship_to_policyholder = self._determine_relationship_to_policyholder(policy_user, affected_person)

        # Determine if affected person is likely covered
        is_affected_covered = self._is_likely_covered(relationship_to_policyholder)

        # Format the result
        return {
            "personas": {
                "policy_user": policy_user,
                "affected_person": affected_person,
                "location": location,
                "is_abroad": is_abroad,
                "relationship_to_policyholder": relationship_to_policyholder,
                "is_affected_covered": is_affected_covered,
                "mentioned_people": ", ".join(mentioned_people) if mentioned_people else "None specifically mentioned",
                "total_count": total_count,
                "claimant_count": claimant_count,
                "relationship": relationship if relationship else "Not clearly specified"
            }
        }

    def _extract_family_relationships(self, question: str) -> Tuple[List[str], List[str]]:
        """
        Extract family relationship terms from the question.

        Args:
            question: The question to analyze

        Returns:
            Tuple of (found_relationships, mentioned_people)
        """
        found_relationships = []
        mentioned_people = []

        # Family relationship terms
        family_terms = {
            'daughter': 'daughter',
            'son': 'son',
            'child': 'child',
            'children': 'children',
            'wife': 'wife',
            'husband': 'husband',
            'spouse': 'spouse',
            'partner': 'partner',
            'mother': 'mother',
            'father': 'father',
            'parent': 'parent',
            'parents': 'parents',
            'grandparent': 'grandparent',
            'grandmother': 'grandmother',
            'grandfather': 'grandfather',
            'sister': 'sister',
            'brother': 'brother',
            'sibling': 'sibling',
            'aunt': 'aunt',
            'uncle': 'uncle',
            'cousin': 'cousin',
            'niece': 'niece',
            'nephew': 'nephew',
            'family': 'family member'
        }

        # Patterns for indirect references
        indirect_references = [
            r'my\s+(?:business\s+)?partner\s+had',
            r'my\s+(?:family\s+)?member\s+(?:had|was|got)',
            r'my\s+(?:colleague|coworker)\s+(?:had|was|got)',
            r'(?:death|illness|injury)\s+of\s+(?:my|the)',
        ]

        # Check for family members and relationships
        for term, relationship_type in family_terms.items():
            if re.search(r'\b' + term + r'\b', question, re.IGNORECASE):
                found_relationships.append(relationship_type)

                # Check if this is mentioned as a non-claimant
                is_indirect = any(
                    re.search(pattern + r'.*\b' + term + r'\b', question, re.IGNORECASE)
                    for pattern in indirect_references
                )

                # Add to mentioned people
                if is_indirect:
                    mentioned_people.append(f"{relationship_type} (not a claimant)")
                else:
                    mentioned_people.append(relationship_type)

        return found_relationships, mentioned_people

    def _extract_non_family_relationships(self, question: str) -> Tuple[List[str], List[str]]:
        """
        Extract non-family relationship terms from the question.

        Args:
            question: The question to analyze

        Returns:
            Tuple of (found_relationships, mentioned_people)
        """
        found_relationships = []
        mentioned_people = []

        # Non-family relationship terms
        other_terms = {
            'friend': 'friend',
            'friends': 'friends',
            'colleague': 'colleague',
            'coworker': 'coworker',
            'business partner': 'business partner',
            'neighbor': 'neighbor',
            'guest': 'guest',
            'traveler': 'fellow traveler'
        }

        # Patterns for indirect references
        indirect_references = [
            r'my\s+(?:business\s+)?partner\s+had',
            r'my\s+(?:family\s+)?member\s+(?:had|was|got)',
            r'my\s+(?:colleague|coworker)\s+(?:had|was|got)',
            r'(?:death|illness|injury)\s+of\s+(?:my|the)',
        ]

        # Check for non-family relationships
        for term, relationship_type in other_terms.items():
            if re.search(r'\b' + term + r'\b', question, re.IGNORECASE):
                found_relationships.append(relationship_type)

                # Check if this is mentioned as a non-claimant
                is_indirect = any(
                    re.search(pattern + r'.*\b' + term + r'\b', question, re.IGNORECASE)
                    for pattern in indirect_references
                )

                # Add to mentioned people
                if is_indirect:
                    mentioned_people.append(f"{relationship_type} (not a claimant)")
                else:
                    mentioned_people.append(relationship_type)

        return found_relationships, mentioned_people

    def _count_people_from_numeric_mentions(self, question: str) -> int:
        """
        Extract explicit numeric mentions of people in the question.

        Args:
            question: The question to analyze

        Returns:
            The explicitly mentioned count of people, or 0 if none found
        """
        # Pattern for phrases like "family of X", "group of X", "X of us", etc.
        patterns = [
            r'family\s+of\s+(\d+)',
            r'group\s+of\s+(\d+)',
            r'(\d+)\s+of\s+us',
            r'(\d+)\s+people',
            r'(\d+)\s+persons',
            r'(\d+)\s+travelers',
            r'(\d+)\s+members',
            r'for\s+(\d+)'
        ]

        # Look for matches
        for pattern in patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                # Return the first numeric match as an integer
                try:
                    return int(matches[0])
                except (ValueError, IndexError):
                    continue

        return 0  # No explicit numeric mentions found

    def _count_people_from_pronouns(self, question: str) -> int:
        """
        Count distinct people based on pronoun usage, with improved contextual understanding.

        Args:
            question: The question to analyze

        Returns:
            Estimated count of distinct people
        """
        # Check if "our" is being used in institutional context rather than indicating multiple people
        institutional_our_pattern = r'\b(our|at our) (hotel|resort|company|office|facility|premises|building|property|organization|institution)\b'
        has_institutional_our = bool(re.search(institutional_our_pattern, question, re.IGNORECASE))

        # Check for different types of pronouns
        has_first_person_singular = bool(re.search(r'\b(I|me|my)\b', question, re.IGNORECASE))

        # Modified first person plural check to exclude institutional "our"
        has_first_person_plural = bool(re.search(r'\b(we|us)\b', question, re.IGNORECASE))
        if not has_first_person_plural and re.search(r'\bour\b', question, re.IGNORECASE) and not has_institutional_our:
            has_first_person_plural = True

        has_second_person = bool(re.search(r'\b(you|your)\b', question, re.IGNORECASE))
        has_third_person_singular_male = bool(re.search(r'\b(he|him|his)\b', question, re.IGNORECASE))
        has_third_person_singular_female = bool(re.search(r'\b(she|her)\b', question, re.IGNORECASE))
        has_third_person_plural = bool(re.search(r'\b(they|them|their)\b', question, re.IGNORECASE))

        # Count distinct people based on pronoun types
        distinct_people_count = 0

        if has_first_person_singular:
            distinct_people_count += 1  # The speaker/policy holder

        if has_first_person_plural and not has_institutional_our:
            # 'We' implies at least 2 people, but only if not institutional
            distinct_people_count = max(distinct_people_count, 2)

        # Only count 'you' as another person when it's clearly referring to a different individual
        # and not the customer service or entity being addressed
        if has_second_person and re.search(r'\b(you|your) (?:and|with|also|too)\b', question, re.IGNORECASE):
            distinct_people_count += 1

        # More contextual checks for second person
        # Don't count 'you' in service requests like "can you help"
        if has_second_person and not re.search(r'(?:can|could|would|will|please)\s+you', question, re.IGNORECASE):
            distinct_people_count += 1

        if has_third_person_singular_male:
            distinct_people_count += 1

        if has_third_person_singular_female:
            distinct_people_count += 1

        if has_third_person_plural:
            # 'They' implies at least 2 more people
            if distinct_people_count == 0:
                distinct_people_count = 2  # Minimum for 'they'
            else:
                distinct_people_count += 1  # Add at least one more person

        # If no pronouns were found, default to 1 person (the claimant)
        if distinct_people_count == 0:
            distinct_people_count = 1

        return distinct_people_count

    def _determine_default_persona_info(
            self,
            question: str,
            found_relationships: List[str]
    ) -> Tuple[str, str, int, int, str]:
        """
        Determine default persona information based on pronouns and context.

        Args:
            question: The question to analyze
            found_relationships: List of relationships found in the question

        Returns:
            Tuple of (policy_user, relationship, total_count, claimant_count, relationship)
        """
        # Default based on pronouns
        if re.search(r'\b(I|me|my)\b', question, re.IGNORECASE) and not found_relationships:
            policy_user = "policyholder (individual)"
            relationship = "none mentioned"
            claimant_count = 1

        elif re.search(r'\b(we|us|our)\b', question, re.IGNORECASE) and not found_relationships:
            policy_user = "policyholder (group)"
            relationship = "group (unspecified)"
            claimant_count = 2  # At least 2 people

        elif found_relationships:
            # If relationships found but no clear policy user, assume relationship is claimant
            policy_user = found_relationships[0]
            relationship = "family member of policyholder"
            claimant_count = 1

        else:
            # Default if nothing could be determined
            policy_user = "undetermined"
            relationship = "Not clearly specified"
            claimant_count = 1

        # Get people count from pronouns
        pronoun_count = self._count_people_from_pronouns(question)

        # Get explicit numeric mentions of people count
        numeric_count = self._count_people_from_numeric_mentions(question)

        # Use the larger of the two counts
        total_count = max(pronoun_count, numeric_count)

        # If we have an explicit numeric count and we're in a family/group context,
        # update the claimant count accordingly
        if numeric_count > 0 and re.search(r'\b(we|us|our|family|group)\b', question, re.IGNORECASE):
            claimant_count = numeric_count

        return policy_user, relationship, total_count, claimant_count, relationship


    def _identify_affected_person(self, question: str) -> Optional[str]:
        """
        Identify who experienced the event/accident/illness in the query.

        Args:
            question: The query to analyze

        Returns:
            String describing who experienced the event or None if unclear
        """
        # Patterns for first-person event experiencing
        first_person_patterns = [
            r'I\s+(?:had|have|got|experienced|suffered|am suffering|was diagnosed with|developed)\s+(?:a|an)?\s*(?:illness|sickness|disease|condition|injury|accident|problem)',
            r'I\s+(?:broke|injured|hurt|damaged|lost)\s+my',
            r'I\s+(?:am|was|feel|felt)\s+(?:sick|ill|unwell|not well|injured)',
            r'my\s+(?:illness|sickness|disease|condition|injury|accident|problem)',
            r'I\s+need\s+(?:a|to see)\s+(?:doctor|medical|physician|hospital)',
            r'I\s+(?:had|have)\s+(?:pain|discomfort|symptoms)',
        ]

        # Patterns for family members experiencing events
        family_patterns = {
            'child': r'(?:my|our)\s+(?:child|kid|son|daughter)\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
            'spouse': r'(?:my|our)\s+(?:spouse|husband|wife|partner)\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
            'parent': r'(?:my|our)\s+(?:parent|father|mother|dad|mom)\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
            'family': r'(?:my|our)\s+(?:family member|relative|brother|sister|sibling|uncle|aunt|cousin|nephew|niece)\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
        }

        # Check for first person as affected
        for pattern in first_person_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return "policyholder (self)"

        # Check for family members as affected
        for family_type, pattern in family_patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                return f"{family_type} of policyholder"

        # Check for others as affected
        other_patterns = {
            'friend': r'(?:my|our)\s+friend\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
            'colleague': r'(?:my|our)\s+(?:colleague|coworker|co-worker)\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
            'travel companion': r'(?:my|our)\s+(?:travel companion|fellow traveler|traveling partner)\s+(?:has|had|is|was|got|became|fell|is suffering|started)',
        }

        for other_type, pattern in other_patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                return f"{other_type} of policyholder"

        # Default to None if no clear matches
        return None

    def _handle_special_case_personas(
            self,
            question: str,
            found_relationships: List[str]
    ) -> Tuple[Optional[str], Optional[List[str]], Optional[str], Optional[int], Optional[int], Optional[str]]:
        """
        Handle special case persona scenarios, now including affected person.

        Args:
            question: The question to analyze
            found_relationships: List of relationships found in the question

        Returns:
            Tuple of (policy_user, mentioned_people, relationship, total_count, claimant_count, affected_person)
            Returns None for any values that couldn't be determined by special cases
        """
        policy_user = None
        mentioned_people = None
        relationship = None
        total_count = None
        claimant_count = None
        affected_person = None

        # Define family relation terms for consistent use
        family_terms = [
            'daughter', 'son', 'child', 'children', 'wife', 'husband', 'spouse', 'partner',
            'mother', 'father', 'parent', 'parents', 'grandparent', 'grandmother', 'grandfather',
            'sister', 'brother', 'sibling', 'aunt', 'uncle', 'cousin', 'niece', 'nephew',
            'family', 'relative'
        ]

        # Create regex pattern for family relationships
        family_pattern = r'\b(' + '|'.join(family_terms) + r')\b'

        # Special case: business partner scenario
        if re.search(r'my\s+(?:business\s+)?partner\s+had', question, re.IGNORECASE):
            if re.search(r'I\s+am\s+traveling\s+alone', question, re.IGNORECASE):
                policy_user = "policyholder (individual traveler)"
                mentioned_people = ["business partner (not traveling/insured)"]
                relationship = "business relationship (but not traveling together)"
                total_count = 2
                claimant_count = 1
                affected_person = "business partner (not a claimant)"
            else:
                policy_user = "policyholder (individual)"
                mentioned_people = ["business partner"]
                relationship = "business relationship"
                total_count = 2
                claimant_count = 1
                affected_person = "business partner"

        # Special case: my family member scenarios (using any family term)
        elif re.search(r'\bmy\s+' + family_pattern, question, re.IGNORECASE):
            policy_user = "policyholder (family member)"
            relationship = "family - " + ", ".join(found_relationships)

            # Check if family member is experiencing the event
            affected_match = re.search(
                r'my\s+(' + '|'.join(family_terms) + r')\s+(?:is|was|had|got|needs|has been|suffered|experienced)',
                question, re.IGNORECASE)
            if affected_match:
                affected_person = affected_match.group(1)

            # Extract all mentioned family members
            all_family_matches = re.findall(r'my\s+(' + '|'.join(family_terms) + r')', question, re.IGNORECASE)
            mentioned_people = [match for match in all_family_matches if match]

            # Check for explicit family size
            family_size = self._count_people_from_numeric_mentions(question) if hasattr(self,
                                                                                        '_count_people_from_numeric_mentions') else 0

            if family_size > 0:
                total_count = family_size
                # Check if all family members are claiming
                if re.search(r'(?:all|everyone|each|the whole family|all \d+ of us|for all of us)', question,
                             re.IGNORECASE):
                    claimant_count = family_size
                else:
                    # Check if only family member is a claimant or multiple
                    if affected_person and not re.search(r'(?:all|everyone|each|all \d+ of us)', question,
                                                         re.IGNORECASE):
                        claimant_count = 1  # Just the affected family member
                    else:
                        claimant_count = min(family_size, len(mentioned_people) + 1)  # +1 for policyholder
            else:
                # No explicit family size
                total_count = max(2, len(mentioned_people) + 1)  # At least policyholder + mentioned people
                claimant_count = 1 if affected_person else total_count  # Either just affected or all

        # Special case: our family member scenario (using any family term)
        elif re.search(r'\bour\s+' + family_pattern, question, re.IGNORECASE):
            policy_user = "policyholder (joint policy with family)"
            relationship = "family - " + ", ".join(found_relationships)

            # Check for explicit family size
            family_size = self._count_people_from_numeric_mentions(question) if hasattr(self,
                                                                                        '_count_people_from_numeric_mentions') else 0

            # Check who is actually affected/claiming
            affected_terms = [
                'all of us', 'everyone', 'the whole family',
                r'all \d+ of us', r'for the \d+ of us', 'for all of us'
            ]

            # Extract the specific affected family member if mentioned
            affected_match = re.search(r'(?:my|our)\s+(' + '|'.join(
                family_terms) + r')\s+(?:is|was|had|got|needs|has been|suffered|experienced)', question, re.IGNORECASE)
            if affected_match:
                affected_person = affected_match.group(1)

            # If the question indicates everyone is affected/claiming
            everyone_affected = any(re.search(pattern, question, re.IGNORECASE) for pattern in affected_terms)

            if family_size > 0:
                total_count = family_size

                # Determine claimant count based on context
                if everyone_affected:
                    claimant_count = family_size
                    # If we found a specific affected person but everyone is claiming
                    if affected_person:
                        affected_person = f"{affected_person} (but whole family of {family_size} is claiming)"
                elif affected_person:
                    # Check if only the affected person is claiming
                    if re.search(r'(?:only|just)\s+(?:my|our|the)\s+' + affected_person, question, re.IGNORECASE):
                        claimant_count = 1
                    else:
                        # If reimbursement is for all
                        if re.search(r'reimburse\s+(?:the|our|all).*(trip|vacation|holiday)', question, re.IGNORECASE):
                            claimant_count = family_size
                            affected_person = f"{affected_person} (but whole family of {family_size} is claiming)"
                        else:
                            claimant_count = 1
                else:
                    # Default to all family members if no specific affected person
                    claimant_count = family_size
            else:
                # No explicit family size, use "we/our" to imply at least 2
                total_count = max(2, len(found_relationships) + 1)
                claimant_count = 1 if affected_person and not everyone_affected else total_count

        # Special case: we/family scenario without explicit "our family" phrase
        elif re.search(r'\b(we|us)\b', question, re.IGNORECASE) and any(
                term in question.lower() for term in family_terms):
            policy_user = "policyholder (family group)"
            relationship = "family group"

            # Extract all mentioned family members
            all_family_matches = re.findall(r'(?:the|my|our)\s+(' + '|'.join(family_terms) + r')', question,
                                            re.IGNORECASE)
            mentioned_people = [match for match in all_family_matches if match]

            # Check for explicit family size
            family_size = self._count_people_from_numeric_mentions(question) if hasattr(self,
                                                                                        '_count_people_from_numeric_mentions') else 0

            # Look for affected person
            affected_match = re.search(r'(?:my|our|the)\s+(' + '|'.join(
                family_terms) + r')\s+(?:is|was|had|got|needs|has been|suffered|experienced)', question, re.IGNORECASE)
            if affected_match:
                affected_person = affected_match.group(1)

            # Determine counts
            if family_size > 0:
                total_count = family_size

                # Check if everyone is claiming
                if re.search(r'(?:reimburse|refund|cover)\s+(?:the|our|all|us|everyone).*(trip|vacation|holiday|stay)',
                             question, re.IGNORECASE):
                    claimant_count = family_size
                    # If specific person affected but all claiming
                    if affected_person:
                        affected_person = f"{affected_person} (but all {family_size} travelers claiming)"
                elif affected_person:
                    # Default to just affected person claiming unless specified
                    claimant_count = 1
                else:
                    claimant_count = family_size
            else:
                # No explicit size, determine from context
                total_count = max(2, len(mentioned_people) + 1)  # At least 2 for "we"
                claimant_count = 1 if affected_person else total_count

        # Special case: traveling alone but mentioning business partner
        elif re.search(r'I\s+am\s+traveling\s+alone', question, re.IGNORECASE) and 'partner' in question.lower():
            policy_user = "policyholder (individual traveler)"
            mentioned_people = ["business partner (not traveling/insured)"]
            relationship = "business relationship"
            total_count = 2
            claimant_count = 1
            affected_person = "policyholder (self)"

        # Special case: personal item theft or loss
        elif re.search(r'(my|I).*(handbag|purse|bag|phone|document|wallet|passport|luggage)', question, re.IGNORECASE):
            policy_user = "policyholder (individual)"
            mentioned_people = []
            relationship = "none mentioned"

            # Check for group/family mentions
            family_size = self._count_people_from_numeric_mentions(question) if hasattr(self,
                                                                                        '_count_people_from_numeric_mentions') else 0
            if family_size > 0 and re.search(r'\b(we|us|our|family)\b', question, re.IGNORECASE):
                total_count = family_size

                # Check if other people's items are also affected
                if re.search(r'(?:our|all our|everyone\'s).*(handbag|purse|bag|phone|document|wallet|passport|luggage)',
                             question, re.IGNORECASE):
                    claimant_count = family_size
                    affected_person = "entire group's belongings"
                else:
                    claimant_count = 1  # Usually only the owner of the item is claiming
                    affected_person = "policyholder's belongings"
            else:
                total_count = 1
                claimant_count = 1
                affected_person = "policyholder (self)"

        # Special case: illness or medical condition
        elif re.search(r'(I had|I got|I am|I was).*?(sick|ill|poisoning|diarrhea|vomiting|stomach|pain|fever|injured)',
                       question, re.IGNORECASE):
            policy_user = "policyholder (individual)"
            mentioned_people = []
            relationship = "none mentioned"

            # Check for group/family mentions
            family_size = self._count_people_from_numeric_mentions(question) if hasattr(self,
                                                                                        '_count_people_from_numeric_mentions') else 0
            if family_size > 0 and re.search(r'\b(we|us|our|family)\b', question, re.IGNORECASE):
                total_count = family_size

                # Check if only the person is sick or everyone
                if re.search(r'(we all|all of us).*?(sick|ill|injured|affected)', question, re.IGNORECASE):
                    claimant_count = family_size
                    affected_person = "entire family/group"
                else:
                    # Check for trip cancellation for everyone
                    if re.search(
                            r'(?:reimburse|refund|cover)\s+(?:the|our|all|us|everyone).*(trip|vacation|holiday|stay)',
                            question, re.IGNORECASE):
                        claimant_count = family_size
                        affected_person = "policyholder (but whole group of " + str(family_size) + " is claiming)"
                    else:
                        claimant_count = 1
                        affected_person = "policyholder (self)"
            else:
                total_count = 1
                claimant_count = 1
                affected_person = "policyholder (self)"

        return policy_user, mentioned_people, relationship, total_count, claimant_count, affected_person

    def _determine_relationship_to_policyholder(self, policy_user: str, affected_person: str) -> str:
        """
        Determine the relationship between the affected person and policyholder.

        Args:
            policy_user: The identified policy user
            affected_person: The identified affected person

        Returns:
            String describing relationship
        """
        # If they appear to be the same person
        if policy_user.lower() in affected_person.lower() or affected_person.lower() in policy_user.lower():
            return "self"

        # Check for keywords in the affected person description
        affected_lower = affected_person.lower()
        if any(term in affected_lower for term in ["spouse", "wife", "husband", "partner"]):
            return "spouse/partner"
        elif any(term in affected_lower for term in ["child", "son", "daughter"]):
            return "child/dependent"
        elif any(term in affected_lower for term in ["parent", "father", "mother"]):
            return "parent"
        elif any(term in affected_lower for term in ["brother", "sister", "sibling"]):
            return "sibling"
        elif any(term in affected_lower for term in ["friend", "colleague", "coworker"]):
            return "non-family relation"
        elif "family" in affected_lower:
            return "family member"

        # Default if no clear relationship
        return "unknown"

    def _is_likely_covered(self, relationship: str) -> bool:
        """
        Determine if the affected person is likely covered based on their relationship.

        Args:
            relationship: Relationship to policyholder

        Returns:
            Boolean indicating likely coverage
        """
        # These relationships are typically covered
        covered_relationships = [
            "self", "spouse/partner", "child/dependent", "family member"
        ]

        # Check if relationship is in typically covered list
        return relationship in covered_relationships

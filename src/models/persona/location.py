#  src/models/persona/location.py

"""
Location detection for insurance queries.
Identifies where an event occurred.
"""
import re
from typing import Tuple


class LocationDetector:
    """
    Detects location information in insurance queries.
    """

    def identify_location(self, question: str) -> Tuple[str, bool]:
        """
        Identify location information from the insurance query.

        Args:
            question: The query to analyze

        Returns:
            Tuple of (location, is_abroad)
        """
        # Common location patterns
        airport_patterns = [
            r'(?:at|in)\s+(?:the)?\s+airport',
            r'baggage.{1,30}(?:lost|delayed|missing|derouted)',
            r'luggage.{1,30}(?:lost|delayed|missing|derouted)',
            r'airport.{1,30}(?:lost|delayed|missing)',
            r'flight.{1,30}(?:lost|delayed|missing)',
            r'check-in.{1,30}(?:lost|delayed|missing)',
        ]

        hotel_patterns = [
            r'(?:at|in)\s+(?:the|my|our)?\s+hotel',
            r'(?:at|in)\s+(?:the|a)?\s+resort',
            r'staying\s+(?:at|in)',
            r'during\s+(?:my|our)\s+stay',
            r'accommodation'
        ]

        hospital_patterns = [
            r'(?:at|in)\s+(?:the|a)?\s+hospital',
            r'medical\s+(?:facility|center|centre)',
            r'emergency\s+room',
            r'clinic',
            r'doctor',
            r'medical\s+treatment'
        ]

        abroad_patterns = [
            r'(?:abroad|overseas|internationally)',
            r'(?:in|to|from|at)\s+(?!home|my home|our home)([A-Z][a-z]+)',  # Country/city names
            r'foreign\s+(?:country|place|location)',
            r'outside\s+(?:my|the|our)\s+country',
            r'international\s+(?:trip|travel|journey)',
            r'vacation\s+(?:in|to|at)'
        ]

        domestic_patterns = [
            r'(?:at|in)\s+(?:my|our)\s+home',
            r'at\s+home',
            r'domestic',
            r'within\s+(?:my|the|our)\s+country'
        ]

        transportation_patterns = [
            r'(?:on|in)\s+(?:the|a)?\s+(?:train|bus|car|taxi|subway|metro)',
            r'driving',
            r'during\s+(?:the|my|our)?\s+(?:journey|trip|travel|transit)',
            r'(?:train|bus|subway|car|taxi)\s+(?:station|terminal|stop)'
        ]

        # First check if it's abroad (this should be checked first to combine with location)
        is_abroad = any(re.search(pattern, question, re.IGNORECASE) for pattern in abroad_patterns)

        # Check for specific locations by type
        if any(re.search(pattern, question, re.IGNORECASE) for pattern in airport_patterns):
            return "airport", is_abroad
        elif any(re.search(pattern, question, re.IGNORECASE) for pattern in hotel_patterns):
            return "hotel/resort", is_abroad
        elif any(re.search(pattern, question, re.IGNORECASE) for pattern in hospital_patterns):
            return "hospital/medical facility", is_abroad
        elif any(re.search(pattern, question, re.IGNORECASE) for pattern in transportation_patterns):
            return "during transportation", is_abroad
        elif any(re.search(pattern, question, re.IGNORECASE) for pattern in domestic_patterns):
            return "at home/domestic", False  # Domestic patterns override is_abroad

        # If no specific location but abroad is detected
        if is_abroad:
            return "abroad (unspecified location)", True

        # Default if no clear patterns
        return "unspecified location", False

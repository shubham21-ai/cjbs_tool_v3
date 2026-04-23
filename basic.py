from agent_base import SatelliteAgentBase


class BasicInfoBot(SatelliteAgentBase):
    fields = [
        ("altitude", "Orbital altitude in km (perigee/apogee or average)"),
        ("altitude_source", "Source URL for altitude"),
        ("orbital_life_years", "Orbital lifetime in years (design or operational life)"),
        ("orbital_life_source", "Source URL for orbital lifetime"),
        ("launch_orbit_classification", "Orbit type: LEO, MEO, GEO, SSO, HEO, etc."),
        ("orbit_classification_source", "Source URL for orbit classification"),
        ("number_of_payloads", "Number of payloads on the satellite"),
        ("payloads_source", "Source URL for payload count"),
    ]

    def _build_prompt(self, satellite_name: str) -> str:
        return (
            f"Find orbital and payload information for the satellite: {satellite_name}\n\n"
            "Look for: altitude (km), orbital lifetime (years), orbit classification (LEO/MEO/GEO/etc.), "
            "and number of payloads.\n\n"
            "Search nextspaceflight.com first, then Wikipedia, ESA, NASA.\n\n"
            "Call 'Complete Task' with this exact JSON when done:\n"
            f"{self._json_schema()}\n\n"
            "Use 'NA' for any missing fields."
        )
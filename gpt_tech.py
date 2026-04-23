from agent_base import SatelliteAgentBase


class TechBot(SatelliteAgentBase):
    fields = [
        ("hardware", "Main hardware or bus platform used by the satellite"),
        ("sensors", "List of sensors or payloads onboard"),
        ("breakthrough_tech", "Any breakthrough or first-of-its-kind technology, or NA if none"),
        ("tech_source_link", "Source URL for technical information"),
    ]

    def _build_prompt(self, satellite_name: str) -> str:
        return (
            f"Find the hardware, sensors, and innovative technology of the satellite: {satellite_name}\n\n"
            "Look for: main bus/platform, list of sensors/payloads, and any technological firsts or breakthroughs.\n\n"
            "Search ESA, NASA, manufacturer pages, space databases.\n\n"
            "Call 'Complete Task' with this exact JSON when done:\n"
            f"{self._json_schema()}\n\n"
            "Use 'NA' for any missing fields."
        )

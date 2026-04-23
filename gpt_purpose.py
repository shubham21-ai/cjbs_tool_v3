from agent_base import SatelliteAgentBase


class PurposeBot(SatelliteAgentBase):
    fields = [
        ("purpose", "Integer: 1=Communications, 2=Earth Observation, 3=Navigation, 4=Space Science, 5=Technology Development"),
        ("purpose_category_number", "Same integer as purpose"),
        ("purpose_description", "Description of the satellite's purpose/mission"),
        ("purpose_source_link", "Source URL for purpose information"),
        ("sdg_category", "Integer: 1=Economic, 2=Social, 3=Environmental, 4=Innovation"),
        ("sdg_category_identification_numbers", "Array of SDG goal numbers e.g. [13, 15]"),
        ("sdg_description", "Description of which SDGs the satellite serves and how"),
        ("sdg_source_link", "Source URL for SDG classification"),
    ]

    def _build_prompt(self, satellite_name: str) -> str:
        return (
            f"Find the mission purpose and SDG mapping for the satellite: {satellite_name}\n\n"
            "Purpose: 1=Communications, 2=Earth Observation, 3=Navigation, 4=Space Science, 5=Tech Development\n"
            "SDG category: 1=Economic, 2=Social, 3=Environmental, 4=Innovation\n"
            "SDG numbers: UN SDG goal numbers this satellite contributes to.\n\n"
            "Search official mission pages, ESA/NASA sites, reputable space databases.\n\n"
            "Call 'Complete Task' with this exact JSON when done:\n"
            f"{self._json_schema()}\n\n"
            "Use 'NA' for any missing fields."
        )

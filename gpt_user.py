from agent_base import SatelliteAgentBase


class UserBot(SatelliteAgentBase):
    fields = [
        ("user_category_number", "Integer: 1=Military, 2=Civil, 3=Commercial, 4=Government, 5=Mix"),
        ("user_description", "Description of the satellite's user/operator/owner"),
        ("user_source_link", "Source URL for user information"),
    ]

    def _build_prompt(self, satellite_name: str) -> str:
        return (
            f"Find who operates, owns, or uses the satellite: {satellite_name}\n\n"
            "Categories: 1=Military, 2=Civil, 3=Commercial, 4=Government, 5=Mix\n\n"
            "Search for 'operator', 'owner', or 'user' of the satellite from official sources, news articles.\n\n"
            "Call 'Complete Task' with this exact JSON when done:\n"
            f"{self._json_schema()}\n\n"
            "Use 'NA' for any missing fields."
        )
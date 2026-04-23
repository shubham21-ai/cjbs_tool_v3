from agent_base import SatelliteAgentBase


class TechAgent(SatelliteAgentBase):
    fields = [
        ("satellite_type", "Type: Communication / Earth Observation / Navigation / Science / Experimental"),
        ("satellite_type_source", "Source URL for satellite type"),
        ("satellite_application", "Detailed description of satellite's main application"),
        ("application_source", "Source URL for application description"),
        ("sensor_specs", "Sensor specifications: spectral bands and spatial resolution"),
        ("sensor_specs_source", "Source URL for sensor specs"),
        ("technological_breakthroughs", "Notable technological innovations or firsts"),
        ("breakthrough_source", "Source URL for technological breakthroughs"),
    ]

    def _build_prompt(self, satellite_name: str) -> str:
        return (
            f"Find technical specifications for the satellite: {satellite_name}\n\n"
            "Look for: satellite type, application description, sensor specs (bands/resolution), "
            "and any notable technological breakthroughs.\n\n"
            "Search nextspaceflight.com first, then ESA, NASA, Wikipedia.\n\n"
            "Call 'Complete Task' with this exact JSON when done:\n"
            f"{self._json_schema()}\n\n"
            "Use 'NA' for any missing fields."
        )
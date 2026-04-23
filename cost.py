from agent_base import SatelliteAgentBase


class CostBot(SatelliteAgentBase):
    fields = [
        ("launch_cost", "Launch cost in USD"),
        ("launch_cost_source", "Source URL for launch cost"),
        ("launch_vehicle", "Name of the launch vehicle used"),
        ("launch_vehicle_source", "Source URL for launch vehicle"),
        ("launch_date", "Launch date (YYYY-MM-DD or textual)"),
        ("launch_date_source", "Source URL for launch date"),
        ("launch_site", "Launch site / pad name"),
        ("launch_site_source", "Source URL for launch site"),
        ("launch_mass", "Satellite mass at launch in kg"),
        ("launch_mass_source", "Source URL for launch mass"),
        ("launch_success", "1 for successful launch, 0 for failure"),
        ("launch_success_source", "Source URL for launch success status"),
        ("vehicle_reusability", "1 if launch vehicle was reusable, 0 if expendable"),
        ("reusability_details", "Details about vehicle reusability"),
        ("reusability_source", "Source URL for reusability info"),
        ("mission_cost", "Total mission cost in USD (development + launch)"),
        ("mission_cost_source", "Source URL for mission cost"),
    ]

    def _build_prompt(self, satellite_name: str) -> str:
        return (
            f"Find launch and cost information for the satellite: {satellite_name}\n\n"
            "Look for: launch cost (USD), launch vehicle, launch date, launch site, "
            "satellite mass, launch success (1/0), vehicle reusability (1/0), total mission cost.\n\n"
            "Search nextspaceflight.com first, then Wikipedia, ESA, NASA, SpaceNews.\n\n"
            "Call 'Complete Task' with this exact JSON when done:\n"
            f"{self._json_schema()}\n\n"
            "Use 'NA' for any missing fields."
        )
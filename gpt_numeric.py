from agent_base import SatelliteAgentBase


class NumericBot(SatelliteAgentBase):
    fields = [
        ("return_on_investment", "Numeric ROI value e.g. 1.8 meaning 180%, or NA"),
        ("data_of_revenue_from_satellite_launch_musd", "Revenue from satellite launch in million USD, or NA"),
        ("return_on_investment_description", "Explanation and context for the ROI figure"),
        ("return_on_investment_source", "Source URL for ROI and revenue data"),
    ]

    def _build_prompt(self, satellite_name: str) -> str:
        return (
            f"Find the financial return and revenue for the satellite: {satellite_name}\n\n"
            "Look for: ROI (ratio or %), revenue from launch in million USD, financial analysis.\n\n"
            "Search financial reports, space news, government publications.\n\n"
            "Call 'Complete Task' with this exact JSON when done:\n"
            f"{self._json_schema()}\n\n"
            "Use 'NA' for any missing fields."
        )

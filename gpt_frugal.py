from agent_base import SatelliteAgentBase


class FrugalBot(SatelliteAgentBase):
    fields = [
        ("frugal", "YES or NO - is this satellite considered frugal in design/operation?"),
        ("development_cost_efficiency", "1=efficient, 0=not efficient"),
        ("development_cost_efficiency_description", "Explanation of development cost efficiency"),
        ("development_cost_efficiency_source", "Source URL"),
        ("operational_cost_efficiency", "1=efficient, 0=not efficient"),
        ("operational_cost_efficiency_description", "Explanation of operational cost efficiency"),
        ("operational_cost_efficiency_source", "Source URL"),
        ("labour_cost_efficiency", "1=efficient, 0=not efficient"),
        ("labour_cost_efficiency_description", "Explanation of labour cost efficiency"),
        ("labour_cost_efficiency_source", "Source URL"),
        ("frugal_innovation_design", "1=uses frugal innovation, 0=does not"),
        ("frugal_innovation_design_description", "Explanation of frugal innovation principles used"),
        ("frugal_innovation_design_source", "Source URL"),
    ]

    def _build_prompt(self, satellite_name: str) -> str:
        return (
            f"Evaluate cost-efficiency and frugal innovation of the satellite: {satellite_name}\n\n"
            "Is it frugal (YES/NO)? Rate development, operational, labour cost efficiency (1=yes, 0=no). "
            "Identify frugal innovation principles (COTS components, heritage tech reuse, indigenous solutions, modularity).\n\n"
            "Search budget reports, official mission pages, space news.\n\n"
            "Call 'Complete Task' with this exact JSON when done:\n"
            f"{self._json_schema()}\n\n"
            "Use 'NA' for any missing fields."
        )
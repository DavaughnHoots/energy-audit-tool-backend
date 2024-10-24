# energy_audit_tool.py

import datetime

class EnergyAuditTool:
    def __init__(self, power_consumed, duration_hours, seasonal_factor, occupancy_factor, power_factor=1.0):
        """
        Initialize the Energy Audit Tool with the required parameters.

        :param power_consumed: Power consumed in kW
        :param duration_hours: Duration in hours
        :param seasonal_factor: Seasonal adjustment factor (0.8 - 1.2)
        :param occupancy_factor: Occupancy adjustment factor (0.6 - 1.0)
        :param power_factor: Power factor (0.8 - 0.95), default is 1.0 for base calculation
        """
        self.power_consumed = power_consumed
        self.duration_hours = duration_hours
        self.seasonal_factor = seasonal_factor
        self.occupancy_factor = occupancy_factor
        self.power_factor = power_factor
        self.timestamp = datetime.datetime.now()

    def calculate_base_energy(self):
        """
        Calculate the base energy consumption.

        Ebase = P × t
        """
        Ebase = self.power_consumed * self.duration_hours
        print(f"Base Energy (Ebase): {Ebase} kWh")
        return Ebase

    def calculate_seasonal_adjusted_energy(self, Ebase):
        """
        Apply seasonal adjustment to the base energy.

        Eseasonal = Ebase × Fs
        """
        Eseasonal = Ebase * self.seasonal_factor
        print(f"Seasonal Adjusted Energy (Eseasonal): {Eseasonal} kWh")
        return Eseasonal

    def calculate_occupancy_adjusted_energy(self, Eseasonal):
        """
        Apply occupancy adjustment to the seasonal energy.

        Eoccupied = Eseasonal × Fo
        """
        Eoccupied = Eseasonal * self.occupancy_factor
        print(f"Occupancy Adjusted Energy (Eoccupied): {Eoccupied} kWh")
        return Eoccupied

    def calculate_real_energy_consumption(self, Eoccupied):
        """
        Calculate the real energy consumption considering power factor.

        Ereal = Eoccupied × PF
        """
        Ereal = Eoccupied * self.power_factor
        print(f"Real Energy Consumption (Ereal): {Ereal} kWh")
        return Ereal

    def perform_audit(self):
        """
        Perform the complete energy audit calculation workflow.
        """
        print("Starting Energy Audit Calculation...")
        Ebase = self.calculate_base_energy()
        Eseasonal = self.calculate_seasonal_adjusted_energy(Ebase)
        Eoccupied = self.calculate_occupancy_adjusted_energy(Eseasonal)
        Ereal = self.calculate_real_energy_consumption(Eoccupied)
        print(f"Final Energy Consumption: {Ereal} kWh\n")
        return {
            'Ebase': Ebase,
            'Eseasonal': Eseasonal,
            'Eoccupied': Eoccupied,
            'Ereal': Ereal,
            'Timestamp': self.timestamp
        }

def get_user_input():
    try:
        power_consumed = float(input("Enter power consumed (kW): "))
        duration_hours = float(input("Enter duration (hours): "))
        seasonal_factor = float(input("Enter seasonal adjustment factor (0.8 - 1.2): "))
        if not 0.8 <= seasonal_factor <= 1.2:
            raise ValueError("Seasonal factor must be between 0.8 and 1.2.")
        occupancy_factor = float(input("Enter occupancy adjustment factor (0.6 - 1.0): "))
        if not 0.6 <= occupancy_factor <= 1.0:
            raise ValueError("Occupancy factor must be between 0.6 and 1.0.")
        power_factor = float(input("Enter power factor (0.8 - 0.95): "))
        if not 0.8 <= power_factor <= 0.95:
            raise ValueError("Power factor must be between 0.8 and 0.95.")

        return power_consumed, duration_hours, seasonal_factor, occupancy_factor, power_factor
    except ValueError as e:
        print(f"Input error: {e}")
        return None

def main():
    print("=== Energy Audit Tool Prototype ===\n")

    if user_inputs := get_user_input():
        power_consumed, duration_hours, seasonal_factor, occupancy_factor, power_factor = user_inputs

        audit_tool = EnergyAuditTool(
            power_consumed=power_consumed,
            duration_hours=duration_hours,
            seasonal_factor=seasonal_factor,
            occupancy_factor=occupancy_factor,
            power_factor=power_factor
        )

        results = audit_tool.perform_audit()

        print("=== Audit Results ===")
        print(f"Timestamp: {results['Timestamp']}")
        print(f"Base Energy (Ebase): {results['Ebase']} kWh")
        print(f"Seasonal Adjusted Energy (Eseasonal): {results['Eseasonal']} kWh")
        print(f"Occupancy Adjusted Energy (Eoccupied): {results['Eoccupied']} kWh")
        print(f"Real Energy Consumption (Ereal): {results['Ereal']} kWh")
        print("======================\n")
    else:
        # Initialize the tool with example data
        audit_tool = EnergyAuditTool(
            power_consumed=50,         # kW
            duration_hours=10,         # hours
            seasonal_factor=1.2,       # Summer season
            occupancy_factor=0.9,      # Peak occupancy
            power_factor=0.92          # Power factor
        )

        results = audit_tool.perform_audit()

        print("=== Audit Results ===")
        print(f"Timestamp: {results['Timestamp']}")
        print(f"Base Energy (Ebase): {results['Ebase']} kWh")
        print(f"Seasonal Adjusted Energy (Eseasonal): {results['Eseasonal']} kWh")
        print(f"Occupancy Adjusted Energy (Eoccupied): {results['Eoccupied']} kWh")
        print(f"Real Energy Consumption (Ereal): {results['Ereal']} kWh")
        print("======================\n")

if __name__ == "__main__":
    main()


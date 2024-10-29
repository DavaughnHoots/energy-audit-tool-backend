# energy_audit_tool.py

import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class EnergyAuditTool:
    def __init__(self, power_consumed=0, duration_hours=0, seasonal_factor=1.0, occupancy_factor=1.0, power_factor=1.0):
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
        self.energy_results = {}
        self.heat_transfer_results = {}
        self.hvac_results = {}
        self.lighting_results = {}
        self.solar_results = {}
        self.carbon_reduction_results = {}
        self.financial_analysis_results = {}
        self.energy_efficiency_score_results = {}
        self.recommendations = []

    # Energy Consumption Analysis Methods
    def calculate_base_energy(self):
        """
        Calculate the base energy consumption.

        Ebase = P × t
        """
        Ebase = self.power_consumed * self.duration_hours
        self.energy_results['Ebase'] = Ebase
        return Ebase

    def calculate_seasonal_adjusted_energy(self, Ebase):
        """
        Apply seasonal adjustment to the base energy.

        Eseasonal = Ebase × Fs
        """
        Eseasonal = Ebase * self.seasonal_factor
        self.energy_results['Eseasonal'] = Eseasonal
        return Eseasonal

    def calculate_occupancy_adjusted_energy(self, Eseasonal):
        """
        Apply occupancy adjustment to the seasonal energy.

        Eoccupied = Eseasonal × Fo
        """
        Eoccupied = Eseasonal * self.occupancy_factor
        self.energy_results['Eoccupied'] = Eoccupied
        return Eoccupied

    def calculate_real_energy_consumption(self, Eoccupied):
        """
        Calculate the real energy consumption considering power factor.

        Ereal = Eoccupied × PF
        """
        Ereal = Eoccupied * self.power_factor
        self.energy_results['Ereal'] = Ereal
        return Ereal

    def perform_energy_audit(self):
        """
        Perform the complete energy audit calculation workflow.
        """
        Ebase = self.calculate_base_energy()
        Eseasonal = self.calculate_seasonal_adjusted_energy(Ebase)
        Eoccupied = self.calculate_occupancy_adjusted_energy(Eseasonal)
        Ereal = self.calculate_real_energy_consumption(Eoccupied)
        self.energy_results['Timestamp'] = self.timestamp
        return self.energy_results

    # Heat Transfer Analysis Methods
    def calculate_heat_transfer(self, U, A, delta_T):
        """
        Calculate heat transfer rate for a single surface.

        Q = U × A × ΔT

        :param U: Heat transfer coefficient (W/m²·K)
        :param A: Surface area (m²)
        :param delta_T: Temperature difference (K)
        :return: Heat transfer rate (W)
        """
        return U * A * delta_T

    def perform_heat_transfer_analysis(self, surfaces):
        """
        Perform heat transfer calculations for multiple surfaces.

        :param surfaces: List of dictionaries with keys 'name', 'U', 'A', 'delta_T'
        :return: Dictionary with heat transfer results
        """
        total_Q = 0
        for surface in surfaces:
            Q = self.calculate_heat_transfer(surface['U'], surface['A'], surface['delta_T'])
            self.heat_transfer_results[surface['name']] = Q
            total_Q += Q
        self.heat_transfer_results['Total_Q'] = total_Q
        self.heat_transfer_results['Timestamp'] = self.timestamp
        return self.heat_transfer_results

    # HVAC Energy Consumption Analysis Methods
    def calculate_hvac_energy_consumption(self, V, rho, Cp, delta_T, eta, Fo, Fs):
        """
        Calculate HVAC energy consumption.

        EHVAC = (V × rho × Cp × |ΔT|) / eta × Fo × Fs

        :param V: Volume of conditioned space (m³)
        :param rho: Air density (kg/m³)
        :param Cp: Specific heat capacity of air (J/kg·K)
        :param delta_T: Temperature difference (K)
        :param eta: System efficiency (0.6-0.95)
        :param Fo: Occupancy factor (0.6-1.0)
        :param Fs: Seasonal factor (0.8-1.2)
        :return: HVAC energy consumption (kWh)
        """
        delta_T_abs = abs(delta_T)
        if eta == 0:
            raise ValueError("System efficiency (η) cannot be zero.")
        Ebase_adjusted_J = V * rho * Cp * delta_T_abs / eta
        EkWh = Ebase_adjusted_J / 3_600_000  # Convert Joules to kWh
        Efinal_kWh = EkWh * Fo * Fs
        self.hvac_results['EHVAC'] = Efinal_kWh
        return Efinal_kWh

    def perform_hvac_energy_analysis(self, V, rho, Cp, delta_T, eta, Fo, Fs):
        """
        Perform HVAC energy consumption analysis.
        """
        return self.calculate_hvac_energy_consumption(V, rho, Cp, delta_T, eta, Fo, Fs)

    # Lighting Efficiency Analysis Methods
    def calculate_lighting_efficiency(self, luminous_flux, power_consumption, Fu, Fm):
        """
        Calculate lighting efficiency for a single fixture.

        Elighting = Φ / (P × Fu × Fm)

        :param luminous_flux: Luminous flux (lm)
        :param power_consumption: Power consumption (W)
        :param Fu: Utilization factor (0.4 - 0.8)
        :param Fm: Maintenance factor (0.6 - 0.8)
        :return: Lighting efficiency (lm/W)
        """
        if (power_consumption * Fu * Fm) == 0:
            raise ValueError("Power consumption, Fu, and Fm must be greater than zero.")
        return luminous_flux / (power_consumption * Fu * Fm)

    def perform_lighting_efficiency_analysis(self, fixtures):
        """
        Perform lighting efficiency calculations for multiple fixtures.
        """
        print("\nStarting Lighting Efficiency Analysis...")
        total_lighting_efficiency = 0
        lighting_efficiency_details = {}
        annual_energy_savings = 0

        for fixture in fixtures:
            name = fixture['name']
            Φ = fixture['lm']
            P = fixture['P']
            Fu = fixture['Fu']
            Fm = fixture['Fm']
            try:
                E_lighting = self.calculate_lighting_efficiency(Φ, P, Fu, Fm)
                lighting_efficiency_details[name] = E_lighting
                total_lighting_efficiency += E_lighting
                print(f"{name} Lighting Efficiency: {E_lighting:.2f} lm/W")
            except ValueError as e:
                print(f"Error calculating lighting efficiency for {name}: {e}")
                lighting_efficiency_details[name] = None

            # Calculate Annual Energy Savings if applicable
            if 'P_new' in fixture and 'Φ_new' in fixture:
                P_new = fixture['P_new']
                Φ_new = fixture['Φ_new']
                N = fixture['N']
                tannual = fixture['tannual']

                try:
                    E_new = self.calculate_lighting_efficiency(Φ_new, P_new, Fu, Fm)
                    # Energy consumption per fixture
                    Sannual = (P - P_new) * tannual * N / 1000  # Convert W to kW
                    annual_energy_savings += Sannual
                    print(f"Annual Energy Savings for {name}: {Sannual:.2f} kWh")
                except ValueError as e:
                    print(f"Error calculating annual energy savings for {name}: {e}")

        average_efficiency = total_lighting_efficiency / len(fixtures) if fixtures else 0
        self.lighting_results['lighting_efficiency_details'] = lighting_efficiency_details
        self.lighting_results['average_efficiency'] = average_efficiency
        self.lighting_results['annual_energy_savings'] = annual_energy_savings
        self.lighting_results['Timestamp'] = self.timestamp

        print(f"\nAverage Lighting Efficiency: {average_efficiency:.2f} lm/W")
        print(f"Total Annual Energy Savings: {annual_energy_savings:.2f} kWh\n")

        return self.lighting_results

    # Renewable Energy Potential Analysis Methods
    def calculate_solar_potential(self, A, H, r, PR, N):
        """
        Calculate annual solar energy potential.

        Esolar = A × H × r × PR × N

        :param A: Available installation area (m²)
        :param H: Annual solar radiation (kWh/m²/year)
        :param r: Solar panel efficiency (0.15-0.22)
        :param PR: Performance ratio (0.7-0.85)
        :param N: Shade factor (0.9-1.0)
        :return: Annual solar energy potential (kWh/year)
        """
        return A * H * r * PR * N

    def perform_solar_energy_analysis(self, solar_systems):
        """
        Perform solar energy potential calculations for multiple solar installations.
        """
        print("\nStarting Solar Energy Potential Analysis...")
        solar_potential_details = {}
        total_Esolar = 0
        total_Sannual = 0

        for system in solar_systems:
            name = system['name']
            A = system['A']
            H = system['H']
            r = system['r']
            PR = system['PR']
            N = system['N']
            Celectricity = system['Celectricity']
            Mannual = system['Mannual']

            Esolar = self.calculate_solar_potential(A, H, r, PR, N)
            solar_potential_details[name] = Esolar
            total_Esolar += Esolar
            print(f"{name} Annual Solar Energy Potential: {Esolar:.2f} kWh/year")

            # Calculate Annual Cost Savings
            Sannual = Esolar * Celectricity - Mannual
            solar_potential_details[name + '_Sannual'] = Sannual
            total_Sannual += Sannual
            print(f"{name} Annual Cost Savings: ${Sannual:.2f}")

        self.solar_results['solar_potential_details'] = solar_potential_details
        self.solar_results['total_Esolar'] = total_Esolar
        self.solar_results['total_Sannual'] = total_Sannual
        self.solar_results['Timestamp'] = self.timestamp

        print(f"\nTotal Annual Solar Energy Potential: {total_Esolar:.2f} kWh/year")
        print(f"Total Annual Cost Savings: ${total_Sannual:.2f}\n")

        return self.solar_results

    # Energy Savings and Carbon Reduction Analysis Methods
    def calculate_carbon_reduction(self, Ebaseline, Eimproved, EF, Findirect):
        """
        Calculate carbon emissions reduced.

        Creduced = (Ebaseline - Eimproved) × EF × (1 + Findirect)

        :param Ebaseline: Baseline energy consumption (kWh/year)
        :param Eimproved: Post-improvement consumption (kWh/year)
        :param EF: Grid emission factor (kg CO2e/kWh)
        :param Findirect: Indirect emissions factor (0.1-0.2)
        :return: Carbon emissions reduced (kg CO2e/year)
        """
        Senergy = Ebaseline - Eimproved
        Creduced = Senergy * EF * (1 + Findirect)
        return Creduced, Senergy

    def perform_energy_savings_carbon_reduction_analysis(self, analysis_items):
        """
        Perform energy savings and carbon reduction calculations.
        """
        print("\nStarting Energy Savings and Carbon Reduction Analysis...")
        carbon_reduction_details = {}
        total_Senergy = 0
        total_Creduced = 0
        total_Btotal = 0

        for item in analysis_items:
            name = item['name']
            Ebaseline = item['Ebaseline']
            Eimproved = item['Eimproved']
            EF = item['EF']
            Findirect = item['Findirect']
            Cenergy = item['Cenergy']
            Ccarbon = item['Ccarbon']

            Creduced, Senergy = self.calculate_carbon_reduction(Ebaseline, Eimproved, EF, Findirect)
            Btotal = (Senergy * Cenergy) + (Creduced * Ccarbon)
            carbon_reduction_details[name] = {
                'Senergy': Senergy,
                'Creduced': Creduced,
                'Btotal': Btotal
            }
            total_Senergy += Senergy
            total_Creduced += Creduced
            total_Btotal += Btotal
            print(f"{name} - Energy Savings: {Senergy:.2f} kWh/year")
            print(f"{name} - Carbon Reduction: {Creduced:.2f} kg CO2e/year")
            print(f"{name} - Total Annual Benefit: ${Btotal:.2f}\n")

        self.carbon_reduction_results['carbon_reduction_details'] = carbon_reduction_details
        self.carbon_reduction_results['total_Senergy'] = total_Senergy
        self.carbon_reduction_results['total_Creduced'] = total_Creduced
        self.carbon_reduction_results['total_Btotal'] = total_Btotal
        self.carbon_reduction_results['Timestamp'] = self.timestamp

        print(f"Total Energy Savings: {total_Senergy:.2f} kWh/year")
        print(f"Total Carbon Reduction: {total_Creduced:.2f} kg CO2e/year")
        print(f"Total Annual Benefit: ${total_Btotal:.2f}\n")

        return self.carbon_reduction_results

    # Cost-Benefit and Payback Analysis Methods
    def calculate_simple_payback_period(self, Cinitial, Sannual):
        """
        Calculate the simple payback period.

        Psimple = Cinitial / Sannual

        :param Cinitial: Initial investment ($)
        :param Sannual: Annual savings ($/year)
        :return: Simple payback period (years)
        """
        if Sannual == 0:
            raise ValueError("Annual savings cannot be zero.")
        return Cinitial / Sannual

    def calculate_npv(self, Cinitial, Sannual, r, n):
        """
        Calculate the Net Present Value (NPV).

        NPV = -Cinitial + sum_{t=1 to n} (Sannual / (1 + r)^t)

        :param Cinitial: Initial investment ($)
        :param Sannual: Annual savings ($/year)
        :param r: Discount rate (decimal, e.g., 0.05 for 5%)
        :param n: Project lifetime (years)
        :return: Net Present Value ($)
        """
        NPV = -Cinitial
        for t in range(1, n + 1):
            NPV += Sannual / ((1 + r) ** t)
        return NPV

    def calculate_life_cycle_cost(self, Cinitial, Coperating, Cmaintenance, r, n, Cdisposal=0):
        """
        Calculate the Life Cycle Cost (LCC).

        LCC = Cinitial + sum_{t=1 to n} ((Coperating + Cmaintenance) / (1 + r)^t) + (Cdisposal / (1 + r)^n)

        :param Cinitial: Initial investment ($)
        :param Coperating: Annual operating cost ($)
        :param Cmaintenance: Annual maintenance cost ($)
        :param r: Discount rate (decimal)
        :param n: Project lifetime (years)
        :param Cdisposal: End-of-life disposal cost ($)
        :return: Life Cycle Cost ($)
        """
        LCC = Cinitial
        for t in range(1, n + 1):
            LCC += (Coperating + Cmaintenance) / ((1 + r) ** t)
        LCC += Cdisposal / ((1 + r) ** n)
        return LCC

    def perform_financial_analysis(self, financial_items):
        """
        Perform financial analysis including simple payback period, NPV, and LCC.
        """
        print("\nStarting Cost-Benefit and Payback Analysis...")
        financial_analysis_details = {}

        for item in financial_items:
            name = item['name']
            Cinitial = item['Cinitial']
            Sannual = item['Sannual']
            Coperating = item.get('Coperating', 0)
            Cmaintenance = item.get('Cmaintenance', 0)
            r = item['r']
            n = item['n']
            Cdisposal = item.get('Cdisposal', 0)

            try:
                Psimple = self.calculate_simple_payback_period(Cinitial, Sannual)
                NPV = self.calculate_npv(Cinitial, Sannual, r, n)
                LCC = self.calculate_life_cycle_cost(Cinitial, Coperating, Cmaintenance, r, n, Cdisposal)

                financial_analysis_details[name] = {
                    'Psimple': Psimple,
                    'NPV': NPV,
                    'LCC': LCC
                }

                print(f"{name} - Simple Payback Period: {Psimple:.2f} years")
                print(f"{name} - Net Present Value (NPV): ${NPV:.2f}")
                print(f"{name} - Life Cycle Cost (LCC): ${LCC:.2f}\n")

            except ValueError as e:
                print(f"Error in financial analysis for {name}: {e}")

        self.financial_analysis_results['financial_analysis_details'] = financial_analysis_details
        self.financial_analysis_results['Timestamp'] = self.timestamp

        return self.financial_analysis_results

    # Energy Efficiency Scoring and Recommendations Methods
    def calculate_energy_efficiency_score(self, Eactual, Ebaseline, Fo, Fc, Fs):
        """
        Calculate the Energy Efficiency Score (EES).

        EES = (Eactual / Ebaseline) * (100 / (Fo * Fc)) * Fs

        :param Eactual: Actual energy consumption (kWh/year)
        :param Ebaseline: Baseline energy consumption (kWh/year)
        :param Fo: Occupancy factor (0.6 - 1.0)
        :param Fc: Climate factor (0.8 - 1.2)
        :param Fs: System efficiency factor (0.7 - 1.0)
        :return: Energy Efficiency Score (0 - 100)
        """
        if Ebaseline == 0 or Fo * Fc == 0:
            raise ValueError("Baseline consumption and the product of Fo and Fc cannot be zero.")
        base_ratio = Eactual / Ebaseline
        factor_adjustment = 100 / (Fo * Fc)
        EES = base_ratio * factor_adjustment * Fs
        self.energy_efficiency_score_results['EES'] = EES
        self.energy_efficiency_score_results['Eactual'] = Eactual
        self.energy_efficiency_score_results['Ebaseline'] = Ebaseline
        self.energy_efficiency_score_results['Fo'] = Fo
        self.energy_efficiency_score_results['Fc'] = Fc
        self.energy_efficiency_score_results['Fs'] = Fs
        return EES

    def interpret_energy_efficiency_score(self, EES):
        """
        Interpret the Energy Efficiency Score.

        :param EES: Energy Efficiency Score (0 - 100)
        :return: Interpretation string
        """
        if EES >= 85:
            interpretation = "Excellent (85-100): High-performance building."
        elif 70 <= EES < 85:
            interpretation = "Good (70-84): Above-average performance."
        elif 50 <= EES < 70:
            interpretation = "Fair (50-69): Average performance."
        else:
            interpretation = "Poor (<50): Below-average performance."
        self.energy_efficiency_score_results['Interpretation'] = interpretation
        return interpretation

    # Generate Audit Report Function
    def generate_audit_report(self, export=False, filename="audit_report.txt"):
        """
        Generate a comprehensive audit report compiling all analysis results.
        Optionally export the report to a text file.

        :param export: Boolean indicating whether to export the report.
        :param filename: Name of the file to export the report.
        """
        report = f"\n=== Energy Audit Report ===\n"
        report += f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Energy Consumption Results
        if self.energy_results:
            report += "---- Energy Consumption Analysis ----\n"
            report += f"Base Energy (Ebase): {self.energy_results.get('Ebase', 0):,.2f} kWh\n"
            report += f"Seasonal Adjusted Energy (Eseasonal): {self.energy_results.get('Eseasonal', 0):,.2f} kWh\n"
            report += f"Occupancy Adjusted Energy (Eoccupied): {self.energy_results.get('Eoccupied', 0):,.2f} kWh\n"
            report += f"Real Energy Consumption (Ereal): {self.energy_results.get('Ereal', 0):,.2f} kWh\n\n"

        # Heat Transfer Results
        if self.heat_transfer_results:
            report += "---- Heat Transfer Analysis ----\n"
            for surface, Q in self.heat_transfer_results.items():
                if surface not in ['Total_Q', 'Timestamp']:
                    report += f"{surface}: {Q:,.2f} W\n"
            report += f"Total Heat Transfer (Total_Q): {self.heat_transfer_results.get('Total_Q', 0):,.2f} W\n\n"

        # HVAC Energy Consumption Results
        if self.hvac_results:
            report += "---- HVAC Energy Consumption Analysis ----\n"
            report += f"Final HVAC Energy Consumption (EHVAC): {self.hvac_results.get('EHVAC', 0):,.2f} kWh\n\n"

        # Lighting Efficiency Results
        if self.lighting_results:
            report += "---- Lighting Efficiency Analysis ----\n"
            for name, efficiency in self.lighting_results.get('lighting_efficiency_details', {}).items():
                if efficiency is not None:
                    report += f"{name} Lighting Efficiency: {efficiency:,.2f} lm/W\n"
                else:
                    report += f"{name} Lighting Efficiency: Calculation Error\n"
            report += f"\nAverage Lighting Efficiency: {self.lighting_results.get('average_efficiency', 0):,.2f} lm/W\n"
            report += f"Total Annual Energy Savings: {self.lighting_results.get('annual_energy_savings', 0):,.2f} kWh\n\n"

        # Solar Energy Potential Results
        if self.solar_results:
            report += "---- Solar Energy Potential Analysis ----\n"
            for name, Esolar in self.solar_results.get('solar_potential_details', {}).items():
                if not name.endswith('_Sannual'):
                    Sannual = self.solar_results['solar_potential_details'].get(name + '_Sannual', 0)
                    report += f"{name} Annual Solar Energy Potential: {Esolar:,.2f} kWh/year\n"
                    report += f"{name} Annual Cost Savings: ${Sannual:,.2f}\n"
            report += f"\nTotal Annual Solar Energy Potential: {self.solar_results.get('total_Esolar', 0):,.2f} kWh/year\n"
            report += f"Total Annual Cost Savings: ${self.solar_results.get('total_Sannual', 0):,.2f}\n\n"

        # Energy Savings and Carbon Reduction Results
        if self.carbon_reduction_results:
            report += "---- Energy Savings and Carbon Reduction Analysis ----\n"
            for name, details in self.carbon_reduction_results.get('carbon_reduction_details', {}).items():
                report += f"{name} - Energy Savings: {details['Senergy']:,.2f} kWh/year\n"
                report += f"{name} - Carbon Reduction: {details['Creduced']:,.2f} kg CO2e/year\n"
                report += f"{name} - Total Annual Benefit: ${details['Btotal']:,.2f}\n\n"
            report += f"Total Energy Savings: {self.carbon_reduction_results.get('total_Senergy', 0):,.2f} kWh/year\n"
            report += f"Total Carbon Reduction: {self.carbon_reduction_results.get('total_Creduced', 0):,.2f} kg CO2e/year\n"
            report += f"Total Annual Benefit: ${self.carbon_reduction_results.get('total_Btotal', 0):,.2f}\n\n"

        # Financial Analysis Results
        if self.financial_analysis_results:
            report += "---- Cost-Benefit and Payback Analysis ----\n"
            for name, details in self.financial_analysis_results.get('financial_analysis_details', {}).items():
                report += f"{name} - Simple Payback Period: {details['Psimple']:,.2f} years\n"
                report += f"{name} - Net Present Value (NPV): ${details['NPV']:,.2f}\n"
                report += f"{name} - Life Cycle Cost (LCC): ${details['LCC']:,.2f}\n\n"

        # Energy Efficiency Score and Recommendations
        if self.energy_efficiency_score_results:
            report += "---- Energy Efficiency Scoring and Recommendations ----\n"
            EES = self.energy_efficiency_score_results.get('EES', 0)
            interpretation = self.energy_efficiency_score_results.get('Interpretation', '')
            report += f"Energy Efficiency Score (EES): {EES:,.2f}\n"
            report += f"Interpretation: {interpretation}\n\n"

            # Recommendations
            report += "Recommendations:\n"
            for rec in self.recommendations:
                report += f"- {rec['name']}\n"
                report += f"  Potential Savings: ${rec['Spotential']:,.2f}/year\n"
                report += f"  Implementation Cost: ${rec['Cimplementation']:,.2f}\n"
                report += f"  Priority Score: {rec['PriorityScore']:,.2f}\n\n"

            # Implementation Roadmap
            roadmap = self.energy_efficiency_score_results.get('ImplementationRoadmap', {})
            for phase, actions in roadmap.items():
                report += f"{phase}:\n"
                for action in actions:
                    report += f"- {action['name']} (Priority Score: {action['PriorityScore']:,.2f})\n"
                report += "\n"

        report += "=== End of Report ===\n"

        # Print the report to the console
        print(report)

        # Optionally export the report to a text file
        if export:
            try:
                with open(filename, 'w') as file:
                    file.write(report)
                print(f"Audit report successfully exported to {filename}")
            except Exception as e:
                print(f"Failed to export report: {e}")

    def generate_recommendations(self, potential_measures):
        """
        Generate recommendations based on potential measures.

        :param potential_measures: List of dictionaries with keys 'name', 'Spotential', 'Cimplementation',
                                   'Ffeasibility', 'Furgency'
        :return: List of recommendations with priority scores
        """
        print("\nGenerating Recommendations...")
        recommendations = []
        for measure in potential_measures:
            name = measure['name']
            Spotential = measure['Spotential']
            Cimplementation = measure['Cimplementation']
            Ffeasibility = measure['Ffeasibility']
            Furgency = measure['Furgency']
            if Cimplementation == 0:
                raise ValueError("Implementation cost cannot be zero.")
            P = (Spotential / Cimplementation) * Ffeasibility * Furgency
            measure['PriorityScore'] = P
            recommendations.append(measure)
            print(f"Recommendation: {name}")
            print(f"Priority Score: {P:.2f}\n")
        # Sort recommendations by priority score in descending order
        recommendations.sort(key=lambda x: x['PriorityScore'], reverse=True)
        self.recommendations = recommendations
        return recommendations

    def generate_implementation_roadmap(self):
        """
        Generate an implementation roadmap based on recommendations.

        :return: Dictionary with implementation phases
        """
        print("\nGenerating Implementation Roadmap...")
        immediate_actions = []
        short_term = []
        long_term = []
        for rec in self.recommendations:
            urgency = rec['Furgency']
            if urgency >= 1.8:
                immediate_actions.append(rec)
            elif 1.4 <= urgency < 1.8:
                short_term.append(rec)
            else:
                long_term.append(rec)
        roadmap = {
            'Immediate Actions (0-6 months)': immediate_actions,
            'Short-Term (6-12 months)': short_term,
            'Long-Term (1-3 years)': long_term
        }
        self.energy_efficiency_score_results['ImplementationRoadmap'] = roadmap
        for phase, actions in roadmap.items():
            print(f"\n{phase}:")
            for action in actions:
                print(f"- {action['name']} (Priority Score: {action['PriorityScore']:.2f})")
        return roadmap

    # TODO: Add Visualization Methods

    # User input functions can be added here, similar to get_user_energy_inputs(), get_user_heat_transfer_inputs(), etc.

def main():
    print("=== Energy Audit Tool Prototype ===\n")

    # Initialize the tool with example data or user input
    use_example = input("Would you like to use the example scenario for Energy Consumption Analysis? (y/n): ").strip().lower()

    if use_example == 'y':
        # Example Scenario
        print("\nUsing Example Scenario for Energy Consumption:")
        audit_tool = EnergyAuditTool(
            power_consumed=50,         # kW
            duration_hours=10,         # hours
            seasonal_factor=1.2,       # Summer season
            occupancy_factor=0.9,      # Peak occupancy
            power_factor=0.92          # Power factor
        )
    else:
        # User inputs...
        # You can implement functions to get user inputs here.
        pass  # Existing input handling code

    # Perform the energy audit
    energy_results = audit_tool.perform_energy_audit()

    # Other analyses can be performed here as needed.

    # Energy Efficiency Scoring and Recommendations
    perform_ees_analysis = input("Would you like to perform Energy Efficiency Scoring and Recommendations? (y/n): ").strip().lower()
    if perform_ees_analysis == 'y':
        use_example_ees = input("Would you like to use the example EES scenario? (y/n): ").strip().lower()
        if use_example_ees == 'y':
            example_data(audit_tool)
        else:
            # User inputs for EES and recommendations
            pass

    # Visualization (Can be skipped or retained as per user preference)
    visualize = input("Would you like to visualize the results? (y/n): ").strip().lower()
    if visualize == 'y':
        audit_tool.visualize_results()
    else:
        print("Visualization skipped.")

    # Generate the Audit Report
    audit_tool.generate_audit_report()

    # Optional: Export the report
    export_report = input("Would you like to export the audit report to a file? (y/n): ").strip().lower()
    if export_report == 'y':
        filename = input("Enter the filename (e.g., audit_report.txt): ").strip()
        audit_tool.generate_audit_report(export=True, filename=filename)
    else:
        pass  # Report already printed to console

    print("\nEnergy Audit Completed.")


# TODO Rename this here and in `main`
def example_data(audit_tool):
    # Example EES Scenario
    print("\nUsing Example Scenario for Energy Efficiency Scoring:")
    Eactual = 250000
    Ebaseline = 400000
    Fo = 0.85
    Fc = 1.1
    Fs = 0.8

    # Calculate EES
    EES = audit_tool.calculate_energy_efficiency_score(Eactual, Ebaseline, Fo, Fc, Fs)
    interpretation = audit_tool.interpret_energy_efficiency_score(EES)
    print(f"Energy Efficiency Score (EES): {EES:.2f}")
    print(f"Interpretation: {interpretation}")

    # Generate Recommendations
    potential_measures = [
        {
            'name': 'HVAC System Upgrade',
            'Spotential': 25000,          # $/year
            'Cimplementation': 150000,    # $
            'Ffeasibility': 0.9,          # 0 - 1
            'Furgency': 1.8               # 1 - 2
        },
        {
            'name': 'Lighting Retrofit',
            'Spotential': 20000,          # $/year
            'Cimplementation': 80000,     # $
            'Ffeasibility': 0.95,         # 0 - 1
            'Furgency': 1.6               # 1 - 2
        },
        {
            'name': 'Building Envelope Improvements',
            'Spotential': 16000,          # $/year
            'Cimplementation': 120000,    # $
            'Ffeasibility': 0.85,         # 0 - 1
            'Furgency': 1.4               # 1 - 2
        }
    ]
    recommendations = audit_tool.generate_recommendations(potential_measures)
    roadmap = audit_tool.generate_implementation_roadmap()

if __name__ == "__main__":
    main()
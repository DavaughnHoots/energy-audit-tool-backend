# energy_audit_tool.py

import logging
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

csv_path = os.path.join(os.path.dirname(__file__), 'energy_star_products.csv')

class ProductRecommender:
    def __init__(self):
        self.product_db = None
        self.categories = {}
        self.efficiency_metrics = {}

    def load_database(self, csv_file):
        """Load and categorize products from CSV"""
        try:
            df = pd.read_csv(csv_file)
            self.product_db = df

            # Clean efficiency values
            if 'Efficiency' in df.columns:
                # Extract numeric values from efficiency strings
                df['Efficiency_Value'] = df['Efficiency'].str.extract(r'(\d+\.?\d*)').astype(float)

            # Create category mapping
            self.categories = df['Main Category'].unique()

            # Extract efficiency metrics for each category
            for category in self.categories:
                category_df = df[df['Main Category'] == category]
                self.efficiency_metrics[category] = self._get_efficiency_metrics(category_df)

            return True
        except Exception as e:
            print(f"Error loading product database: {e}")
            return False

    def _get_efficiency_metrics(self, df):
        """Extract relevant efficiency metrics for a category"""
        metrics = {}
        if 'Efficiency_Value' in df.columns:
            metrics['efficiency'] = {
                'min': df['Efficiency_Value'].min(),
                'max': df['Efficiency_Value'].max(),
                'mean': df['Efficiency_Value'].mean()
            }
        return metrics

    def generate_product_recommendations_report(self, recommendations):
        """Generate product recommendations section of the report"""
        report = "\n---- Product Recommendations ----\n"

        if not recommendations:
            report += "No specific product recommendations at this time.\n"
            return report

        for category, products in recommendations.items():
            if products is None or products.empty:
                continue

            report += f"\n{category.title()} Recommendations:\n"
            for idx, product in products.iterrows():
                report += f"- {product['Product Name']}\n"
                if 'Efficiency' in product:
                    report += f"  Efficiency: {product['Efficiency']}\n"
                if 'Features' in product:
                    report += f"  Features: {product['Features']}\n"
                report += "\n"

        return report

    def recommend_products(self, requirements):
        """
        Recommend products based on requirements

        :param requirements: Dict containing:
            - category: Main product category
            - subcategory: Optional subcategory
            - efficiency_min: Minimum efficiency requirement
            - features: List of required features
            - budget: Maximum cost (if available)
        :return: DataFrame of recommended products
        """
        if self.product_db is None:
            return None

        # Start with all products
        recommendations = self.product_db.copy()

        # Filter by category
        if 'category' in requirements:
            recommendations = recommendations[
                recommendations['Main Category'] == requirements['category']
            ]

        # Filter by subcategory if specified
        if 'subcategory' in requirements:
            recommendations = recommendations[
                recommendations['Sub-Category'] == requirements['subcategory']
            ]

        # Filter by efficiency if specified
        if 'efficiency_min' in requirements:
            if 'Efficiency' in recommendations.columns:
                recommendations = recommendations[
                    recommendations['Efficiency'].str.extract(r'(\d+\.?\d*)').astype(float)
                    >= requirements['efficiency_min']
                ]

        # Filter by features if specified
        if 'features' in requirements:
            for feature in requirements['features']:
                recommendations = recommendations[
                    recommendations['Features'].str.contains(feature, na=False)
                ]

        # Sort by efficiency (if available)
        if 'Efficiency' in recommendations.columns:
            recommendations['Efficiency_Value'] = recommendations['Efficiency'].str.extract(r'(\d+\.?\d*)').astype(float)
            recommendations = recommendations.sort_values('Efficiency_Value', ascending=False)

        return recommendations.head(5)  # Return top 5 recommendations

    def get_category_stats(self, category):
        """Get statistics for a specific category"""
        if self.product_db is None:
            return None

        category_df = self.product_db[self.product_db['Main Category'] == category]

        return {
            'total_products': len(category_df),
            'subcategories': category_df['Sub-Category'].unique().tolist(),
            'efficiency_metrics': self.efficiency_metrics.get(category, {}),
            'features': self._extract_common_features(category_df),
        }

    def _extract_common_features(self, df):
        """Extract common features from a category"""
        return (
            df['Features'].str.split(',').explode().unique().tolist()
            if 'Features' in df.columns
            else []
        )

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
        self.product_db = None
        self.humidity_results = {}
        self.product_recommender = ProductRecommender()

    def load_product_database(self, csv_file):
        """
        Load and process the ENERGY STAR product database.
        """
        try:
            print(f"\nAttempting to load product database from: {csv_file}")
            df = pd.read_csv(csv_file)
            print(f"Successfully loaded {len(df)} products")

            # Clean and process the data
            df['Efficiency'] = df['Efficiency'].str.extract(r'(\d+\.?\d*)').astype(float)
            df['Water_Removal'] = df['Description'].str.extract(r'Capacity \(pints/day\): (\d+\.?\d*)').astype(float)

            print(f"Processed {len(df[df['Efficiency'].notna()])} products with valid efficiency data")
            print(f"Processed {len(df[df['Water_Removal'].notna()])} products with valid capacity data")

            self.product_db = df
            return True  # Return boolean instead of DataFrame
        except Exception as e:
            print(f"Error loading product database: {e}")
            self.product_db = None
            return False

    # def initialize_product_database(self, csv_file):
    #     """Initialize and prepare the product database"""
    #     # Load raw data
    #     success = self.load_product_database(csv_file)

    #     if not success:
    #         print("Failed to load product database")
    #         return False

    #     # Analyze current data coverage
    #     analysis = self.analyze_product_database()
    #     if analysis is None:
    #         print("Failed to analyze product database")
    #         return False

    #     print("\nInitial Database Analysis:")
    #     print(f"Total products: {analysis['total_products']}")
    #     print(f"Efficiency coverage: {analysis['efficiency_coverage']} products")
    #     print(f"Capacity coverage: {analysis['capacity_coverage']} products")

    #     return True

    def initialize_databases(self, energy_star_csv):
        """Initialize all necessary databases"""
        print("\nInitializing product database...")
        if self.product_recommender.load_database(energy_star_csv):
            print("Product database loaded successfully")
            # Print some statistics about the loaded data
            total_products = len(self.product_recommender.product_db)
            categories = len(self.product_recommender.categories)
            print(f"Total products: {total_products}")
            print(f"Product categories: {categories}")
            return True
        return False

    def recommend_energy_improvements(self, building_data):
        """
        Recommend energy improvement products based on audit results
        """
        if not hasattr(self, 'product_recommender') or self.product_recommender.product_db is None:
            print("Product database not initialized")
            return {}

        recommendations = {}

        try:
            # Dehumidifier recommendations
            if hasattr(self, 'humidity_results') and self.humidity_results:
                capacity_needed = self.humidity_results.get('capacity_needed', 0)
                if capacity_needed > 0:
                    dehumidifier_reqs = {
                        'category': 'Appliances',
                        'subcategory': 'Dehumidifiers',
                        'efficiency_min': 2.0,
                        'features': ['ENERGY STAR Certified: Yes'],
                        'capacity_min': capacity_needed * 0.9  # 90% of required capacity
                    }
                    recommendations['dehumidifiers'] = self.product_recommender.recommend_products(dehumidifier_reqs)

            # HVAC recommendations
            if hasattr(self, 'energy_efficiency_score_results'):
                ees = self.energy_efficiency_score_results.get('EES', 100)
                if ees < 70:
                    hvac_reqs = {
                        'category': 'Heating & Cooling',
                        'features': ['ENERGY STAR Certified: Yes']
                    }
                    recommendations['hvac'] = self.product_recommender.recommend_products(hvac_reqs)

            # Add recommendations based on energy consumption
            if self.energy_results.get('Ereal', 0) > 1000:  # High energy consumption
                lighting_reqs = {
                    'category': 'Lighting & Fans',
                    'features': ['ENERGY STAR Certified: Yes']
                }
                recommendations['lighting'] = self.product_recommender.recommend_products(lighting_reqs)

        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return {}

        return recommendations

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

        if self.humidity_results:
            report += "---- Humidity Analysis Results ----\n"
            report += f"Required Dehumidification Capacity: {self.humidity_results.get('capacity_needed', 0):,.1f} pints/day\n\n"

            if 'recommended_products' in self.humidity_results:
                report += "Recommended Dehumidification Products:\n"
                for product in self.humidity_results['recommended_products']:
                    report += f"- {product['name']}\n"
                    report += f"  Efficiency: {product['efficiency']:.2f} L/kWh\n"
                    report += f"  Capacity: {product['capacity']:.1f} pints/day\n"
                    report += f"  Total Investment: ${(product['price'] + product['installation_cost']):,.2f}\n"
                    report += f"  Annual Energy Savings: {product['energy_savings']:,.2f} kWh\n"
                    report += f"  Annual Cost Savings: ${product['cost_savings']:,.2f}\n"
                    report += f"  Payback Period: {product['payback_period']:.1f} years\n\n"

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

    # TODO: Add Visualization Methods (e.g., bar charts, pie charts, etc.)

    def calculate_dehumidification_needs(self, room_volume, relative_humidity, target_humidity):
        """
        Calculate dehumidification needs based on room conditions.

        :param room_volume: Room volume in cubic feet
        :param relative_humidity: Current relative humidity (%)
        :param target_humidity: Target relative humidity (%)
        :return: Required dehumidification capacity in pints/day
        """
        humidity_difference = relative_humidity - target_humidity
        if humidity_difference <= 0:
            return 0

        return 0.0007 * room_volume * humidity_difference

    def calculate_dehumidification_savings(self, current_efficiency, proposed_efficiency, capacity, operating_hours):
        """
        Calculate potential savings from improved dehumidification efficiency.
        """
        if current_efficiency <= 0 or proposed_efficiency <= 0:
            raise ValueError("Efficiency values must be greater than zero")

        # Convert pints/day to L/day (1 pint = 0.473176 L)
        capacity_liters = capacity * 0.473176

        # Calculate daily energy usage for both systems
        current_daily_energy = capacity_liters / current_efficiency
        proposed_daily_energy = capacity_liters / proposed_efficiency

        # Calculate annual savings (assuming not running 24/7)
        actual_operating_days = operating_hours / 24
        energy_savings = (current_daily_energy - proposed_daily_energy) * actual_operating_days

        # Apply a realistic utilization factor (system won't run at full capacity all the time)
        utilization_factor = 0.6  # 60% average utilization

        return energy_savings * utilization_factor

    def calculate_product_payback(self, product_cost, installation_cost, annual_savings):
        """
        Calculate simple payback period for a product.

        :param product_cost: Product cost ($)
        :param installation_cost: Installation cost ($)
        :param annual_savings: Annual cost savings ($/year)
        :return: Payback period (years)
        """
        total_cost = product_cost + installation_cost
        return total_cost / annual_savings if annual_savings > 0 else float('inf')

    def perform_humidity_analysis(self, building_data):
        """
        Perform humidity analysis and recommend dehumidification solutions.
        """
        results = {}

        try:
            # Debug print statements
            print("\nDebug Information:")
            print(f"Product database loaded: {'Yes' if self.product_db is not None else 'No'}")
            if self.product_db is not None:
                print(f"Number of products in database: {len(self.product_db)}")
                print(f"Number of whole-home units: {len(self.product_db[self.product_db['Description'].str.contains('Whole-home', na=False)])}")

            # Calculate dehumidification needs
            capacity_needed = self.calculate_dehumidification_needs(
                building_data['volume'],
                building_data['current_humidity'],
                building_data['target_humidity']
            )

            print(f"\nRequired capacity: {capacity_needed:.1f} pints/day")

            # Get product recommendations
            requirements = {
                'dehumidification_need': capacity_needed,
                'efficiency_threshold': building_data.get('efficiency_threshold', 2.0),
                'is_whole_home': building_data.get('is_whole_home', False)
            }

            recommended_products = self.recommend_products(requirements)

            if recommended_products is not None and not recommended_products.empty:
                best_efficiency = recommended_products.iloc[0]['Efficiency_Value']
                current_efficiency = building_data.get('current_efficiency', 1.5)
                operating_hours = building_data.get('operating_hours', 8760)
                electricity_rate = building_data.get('electricity_rate', 0.12)

                product_details = []
                print("\nRecommended Products:")

                for idx, product in recommended_products.iterrows():
                    # Calculate savings for this specific product
                    energy_savings = self.calculate_dehumidification_savings(
                        current_efficiency,
                        product['Efficiency_Value'],
                        product['Water_Removal'],
                        operating_hours
                    )

                    cost_savings = energy_savings * electricity_rate
                    payback_period = self.calculate_product_payback(
                        product['Price'],
                        product['Installation_Cost'],
                        cost_savings
                    )

                    product_info = {
                        'name': product['Product Name'],
                        'efficiency': product['Efficiency_Value'],
                        'capacity': product['Water_Removal'],
                        'price': product['Price'],
                        'installation_cost': product['Installation_Cost'],
                        'energy_savings': energy_savings,
                        'cost_savings': cost_savings,
                        'payback_period': payback_period
                    }
                    product_details.append(product_info)

                    print(f"\nProduct {idx + 1}:")
                    print(f"Name: {product['Product Name']}")
                    print(f"Efficiency: {product['Efficiency_Value']:.2f} L/kWh")
                    print(f"Capacity: {product['Water_Removal']:.1f} pints/day")
                    print(f"Price: ${product['Price']:,.2f}")
                    print(f"Installation Cost: ${product['Installation_Cost']:,.2f}")
                    print(f"Total Investment: ${(product['Price'] + product['Installation_Cost']):,.2f}")
                    print(f"Annual Energy Savings: {energy_savings:.2f} kWh")
                    print(f"Annual Cost Savings: ${cost_savings:.2f}")
                    print(f"Payback Period: {payback_period:.1f} years")

                results = {
                    'capacity_needed': capacity_needed,
                    'recommended_products': product_details,
                    'best_available_efficiency': best_efficiency,
                    'current_efficiency': current_efficiency
                }

            else:
                results = {
                    'capacity_needed': capacity_needed,
                    'error': 'No suitable products found'
                }
                print("\nNo suitable products found matching the requirements.")

        except Exception as e:
            print(f"Error in humidity analysis: {e}")
            import traceback
            traceback.print_exc()
            results = {'error': str(e)}

        self.humidity_results = results
        return results

    def analyze_product_database(self):
        """Analyze the product database for data completeness and coverage"""
        if self.product_db is None:
            return None

        try:
            return {
                'total_products': len(self.product_db),
                'categories': self.product_db['Main Category']
                .value_counts()
                .to_dict(),
                'subcategories': self.product_db['Sub-Category']
                .value_counts()
                .to_dict(),
                'efficiency_coverage': len(
                    self.product_db[self.product_db['Efficiency'].notna()]
                ),
                'capacity_coverage': len(
                    self.product_db[self.product_db['Water_Removal'].notna()]
                ),
                'description_patterns': self.analyze_description_patterns(),
            }
        except Exception as e:
            print(f"Error analyzing product database: {e}")
            return None

    def analyze_description_patterns(self):
        """Analyze patterns in the Description field to identify additional data"""
        if self.product_db is None:
            return None

        descriptions = self.product_db['Description'].dropna()
        return {
            'capacity_mentions': len(
                descriptions[descriptions.str.contains('capacity', case=False)]
            ),
            'pints_mentions': len(
                descriptions[descriptions.str.contains('pints', case=False)]
            ),
            'efficiency_mentions': len(
                descriptions[descriptions.str.contains('efficiency', case=False)]
            ),
        }

    def enhance_product_data(self):
        """Enhance product data by extracting additional information"""
        if self.product_db is None:
            return None

        # Extract capacity information using multiple patterns
        capacity_patterns = [
            r'Capacity \(pints/day\): (\d+\.?\d*)',
            r'(\d+\.?\d*)\s*pints?\s*per\s*day',
            r'(\d+\.?\d*)\s*pints?\s*capacity',
            r'Water Removal Capacity.*?(\d+\.?\d*)',
            r'removes (\d+\.?\d*) pints'
        ]

        for pattern in capacity_patterns:
            mask = self.product_db['Water_Removal'].isna()
            extracted = self.product_db.loc[mask, 'Description'].str.extract(pattern, expand=False)
            self.product_db.loc[mask & extracted.notna(), 'Water_Removal'] = extracted[extracted.notna()].astype(float)

        # Extract efficiency information from different fields
        efficiency_patterns = [
            r'Efficiency[:\s]+(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*L/kWh',
            r'Energy Factor.*?(\d+\.?\d*)',
            r'IEF.*?(\d+\.?\d*)'
        ]

        for pattern in efficiency_patterns:
            mask = self.product_db['Efficiency'].isna()
            extracted = self.product_db.loc[mask, 'Description'].str.extract(pattern, expand=False)
            self.product_db.loc[mask & extracted.notna(), 'Efficiency'] = extracted[extracted.notna()].astype(float)

    def get_alternative_recommendations(self, requirements):
        """Get alternative recommendations when exact matches aren't found"""
        if self.product_db is None:
            return []

        alternatives = []

        # Try finding products with similar capacity
        if requirements.get('dehumidification_need'):
            capacity_needed = requirements['dehumidification_need']

            # Look for products with at least 50% of required capacity
            min_capacity = capacity_needed * 0.5
            matching_products = self.product_db[
                (self.product_db['Water_Removal'] >= min_capacity) &
                (self.product_db['Water_Removal'].notna())
            ].sort_values('Water_Removal', ascending=False).head(3)

            if not matching_products.empty:
                alternatives.extend(matching_products.to_dict('records'))

                # Calculate number of units needed
                max_capacity = matching_products['Water_Removal'].max()
                if capacity_needed > max_capacity:
                    units_needed = math.ceil(capacity_needed / max_capacity)
                    for product in alternatives:
                        product['units_needed'] = units_needed
                        product['total_capacity'] = product['Water_Removal'] * units_needed

        return alternatives

    def recommend_products(self, requirements):
        """
        Recommend products based on analysis results and requirements.
        """
        if self.product_db is None:
            print("No product database loaded")
            return None

        recommendations = self.product_db.copy()
        print(f"\nInitial product count: {len(recommendations)}")

        # Debug: Show capacity range in database
        valid_capacities = recommendations['Water_Removal'].dropna()
        if not valid_capacities.empty:
            print(f"Capacity range in database: {valid_capacities.min():.1f} to {valid_capacities.max():.1f} pints/day")

        # Filter based on requirements
        if requirements.get('is_whole_home'):
            recommendations = recommendations[
                recommendations['Description'].str.contains('Whole-home', na=False)
            ]
            print(f"After whole-home filter: {len(recommendations)} products")

        if requirements.get('dehumidification_need'):
            capacity_needed = requirements['dehumidification_need']
            # Adjust the margin to be more flexible
            min_capacity = capacity_needed * 0.7  # Allow for 30% lower capacity
            recommendations = recommendations[
                recommendations['Water_Removal'] >= min_capacity
            ]
            print(f"After capacity filter (minimum {min_capacity:.1f} pints/day): {len(recommendations)} products")

        if requirements.get('efficiency_threshold'):
            recommendations = recommendations[
                recommendations['Efficiency'] >= requirements['efficiency_threshold']
            ]
            print(f"After efficiency filter: {len(recommendations)} products")

        # If no products found, try getting alternative recommendations
        if len(recommendations) == 0:
            print("\nNo exact matches found. Looking for alternatives...")
            if alternatives := self.get_alternative_recommendations(requirements):
                print(f"Found {len(alternatives)} alternative recommendations")
                # Convert alternatives list to DataFrame
                recommendations = pd.DataFrame(alternatives)
            else:
                print("No alternative recommendations found")
                return None

        # Sort by efficiency and capacity
        recommendations = recommendations.sort_values(
            by=['Efficiency', 'Water_Removal'],
            ascending=[False, False]
        )

        return recommendations.head(5)  # Return top 5 recommendations

def main():
    print("=== Energy Audit Tool Prototype ===\n")

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

        # Example building data
        building_data = {
            'volume': 20000,           # cubic feet
            'current_humidity': 65,    # %
            'target_humidity': 45,     # %
            'is_whole_home': True,
            'efficiency_threshold': 2.0,
            'operating_hours': 8760,   # full year
            'current_efficiency': 1.5,  # L/kWh
            'electricity_rate': 0.12   # $/kWh
        }

    else:
        # Get user inputs for energy audit
        power_consumed = float(input("Enter power consumed (kW): "))
        duration_hours = float(input("Enter duration (hours): "))
        seasonal_factor = float(input("Enter seasonal factor (0.8-1.2): "))
        occupancy_factor = float(input("Enter occupancy factor (0.6-1.0): "))
        power_factor = float(input("Enter power factor (0.8-0.95): "))

        audit_tool = EnergyAuditTool(
            power_consumed=power_consumed,
            duration_hours=duration_hours,
            seasonal_factor=seasonal_factor,
            occupancy_factor=occupancy_factor,
            power_factor=power_factor
        )

        # Get user inputs for humidity analysis
        print("\nEnter building data for humidity analysis:")
        volume = float(input("Enter room volume (cubic feet): "))
        current_humidity = float(input("Enter current relative humidity (%): "))
        target_humidity = float(input("Enter target relative humidity (%): "))
        is_whole_home = input("Is this a whole-home solution? (y/n): ").lower() == 'y'

        building_data = {
            'volume': volume,
            'current_humidity': current_humidity,
            'target_humidity': target_humidity,
            'is_whole_home': is_whole_home,
            'efficiency_threshold': 2.0,  # Default value
            'operating_hours': 8760,      # Default to full year
            'current_efficiency': 1.5     # Default value
        }

    # Load the product database
    audit_tool.initialize_databases(csv_path)

    # Perform humidity analysis
    humidity_results = audit_tool.perform_humidity_analysis(building_data)

    # Perform the energy audit
    energy_results = audit_tool.perform_energy_audit()


    # Get product recommendations if database was loaded
    if hasattr(audit_tool, 'product_recommender') and audit_tool.product_recommender.product_db is not None:
        recommendations = audit_tool.recommend_energy_improvements(building_data)
    else:
        recommendations = {}

    # Energy Efficiency Scoring and Recommendations
    perform_ees_analysis = input("Would you like to perform Energy Efficiency Scoring and Recommendations? (y/n): ").strip().lower()
    if perform_ees_analysis == 'y':
        use_example_ees = input("Would you like to use the example EES scenario? (y/n): ").strip().lower()
        if use_example_ees == 'y':
            example_data(audit_tool)

    # Generate the Audit Report
    audit_tool.generate_audit_report()

    # Optional: Export the report
    export_report = input("Would you like to export the audit report to a file? (y/n): ").strip().lower()
    if export_report == 'y':
        filename = input("Enter the filename (e.g., audit_report.txt): ").strip()
        audit_tool.generate_audit_report(export=True, filename=filename)

    print("\nEnergy Audit Completed.")

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
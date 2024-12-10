# energy_audit_tool.py

# At the top of energy_audit_tool.py
import datetime
import logging
import math
import io
import json
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
            if "Efficiency" in df.columns:
                # Extract numeric values from efficiency strings
                df["Efficiency_Value"] = (
                    df["Efficiency"].str.extract(r"(\d+\.?\d*)").astype(float)
                )

            # Create category mapping
            self.categories = df["Main Category"].unique()

            # Extract efficiency metrics for each category
            for category in self.categories:
                category_df = df[df["Main Category"] == category]
                self.efficiency_metrics[category] = self._get_efficiency_metrics(
                    category_df
                )

            return True
        except Exception as e:
            print(f"Error loading product database: {e}")
            return False

    def _get_efficiency_metrics(self, df):
        """Extract relevant efficiency metrics for a category"""
        metrics = {}
        if "Efficiency_Value" in df.columns:
            metrics["efficiency"] = {
                "min": df["Efficiency_Value"].min(),
                "max": df["Efficiency_Value"].max(),
                "mean": df["Efficiency_Value"].mean(),
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
                if "Efficiency" in product:
                    report += f"  Efficiency: {product['Efficiency']}\n"
                if "Features" in product:
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
        if "category" in requirements:
            recommendations = recommendations[
                recommendations["Main Category"] == requirements["category"]
            ]

        # Filter by subcategory if specified
        if "subcategory" in requirements:
            recommendations = recommendations[
                recommendations["Sub-Category"] == requirements["subcategory"]
            ]

        # Filter by efficiency if specified
        if "efficiency_min" in requirements:
            if "Efficiency" in recommendations.columns:
                recommendations = recommendations[
                    recommendations["Efficiency"]
                    .str.extract(r"(\d+\.?\d*)")
                    .astype(float)
                    >= requirements["efficiency_min"]
                ]

        # Filter by features if specified
        if "features" in requirements:
            for feature in requirements["features"]:
                recommendations = recommendations[
                    recommendations["Features"].str.contains(feature, na=False)
                ]

        # Sort by efficiency (if available)
        if "Efficiency" in recommendations.columns:
            recommendations["Efficiency_Value"] = (
                recommendations["Efficiency"].str.extract(r"(\d+\.?\d*)").astype(float)
            )
            recommendations = recommendations.sort_values(
                "Efficiency_Value", ascending=False
            )

        return recommendations.head(5)  # Return top 5 recommendations

    def get_category_stats(self, category):
        """Get statistics for a specific category"""
        if self.product_db is None:
            return None

        category_df = self.product_db[self.product_db["Main Category"] == category]

        return {
            "total_products": len(category_df),
            "subcategories": category_df["Sub-Category"].unique().tolist(),
            "efficiency_metrics": self.efficiency_metrics.get(category, {}),
            "features": self._extract_common_features(category_df),
        }

    def _extract_common_features(self, df):
        """Extract common features from a category"""
        return (
            df["Features"].str.split(",").explode().unique().tolist()
            if "Features" in df.columns
            else []
        )


class EnergyAuditTool:
    def __init__(
        self,
        power_consumed=0,
        duration_hours=0,
        seasonal_factor=1.0,
        occupancy_factor=1.0,
        power_factor=1.0,
    ):
        """Initialize the Energy Audit Tool with validation"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Energy Audit Tool")

        # Validate inputs
        self._validate_inputs(
            power_consumed,
            duration_hours,
            seasonal_factor,
            occupancy_factor,
            power_factor,
        )

        # Set instance variables
        self.power_consumed = power_consumed
        self.duration_hours = duration_hours
        self.seasonal_factor = seasonal_factor
        self.occupancy_factor = occupancy_factor
        self.power_factor = power_factor
        self.timestamp = datetime.datetime.now()

        # Initialize results containers
        self.results = {
            "energy": {},
            "hvac": {},
            "lighting": {},
            "humidity": {},
            "recommendations": {},
        }

        # Initialize product recommender
        self.product_recommender = None

    def _validate_inputs(
        self,
        power_consumed,
        duration_hours,
        seasonal_factor,
        occupancy_factor,
        power_factor,
    ):
        """Validate all input parameters"""
        if power_consumed < 0 or duration_hours < 0:
            raise ValueError("Power consumed and duration must be positive values")
        if not 0.8 <= seasonal_factor <= 1.2:
            raise ValueError("Seasonal factor must be between 0.8 and 1.2")
        if not 0.6 <= occupancy_factor <= 1.0:
            raise ValueError("Occupancy factor must be between 0.6 and 1.0")
        if not 0.8 <= power_factor <= 1.0:
            raise ValueError("Power factor must be between 0.8 and 1.0")

    def initialize_databases(self, energy_star_csv):
        """Initialize product database with error handling"""
        try:
            self.product_recommender = ProductRecommender()
            if success := self.product_recommender.load_database(energy_star_csv):
                self.logger.info("Product database initialized successfully")
                return True
            else:
                self.logger.error("Failed to initialize product database")
                return False
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            return False

    def perform_comprehensive_analysis(self, building_data):
        """Perform all analyses in one consolidated method"""
        self.logger.info("Starting comprehensive analysis")

        try:
            # Energy Analysis
            self.results["energy"] = self._perform_energy_analysis()

            # HVAC Analysis
            if "hvac_data" in building_data:
                self.results["hvac"] = self._perform_hvac_analysis(
                    building_data["hvac_data"]
                )

            # Lighting Analysis
            if "lighting_data" in building_data:
                self.results["lighting"] = self._perform_lighting_analysis(
                    building_data["lighting_data"]
                )

            # Humidity Analysis
            if self._check_humidity_data(building_data):
                self.results["humidity"] = self._perform_humidity_analysis(
                    building_data
                )

            # Generate Recommendations
            self.results["recommendations"] = (
                self._generate_comprehensive_recommendations(building_data)
            )

            # Calculate overall efficiency score
            self.results["efficiency_score"] = (
                self._calculate_overall_efficiency_score()
            )

            # Generate financial analysis
            self.results["financial_analysis"] = self._perform_financial_analysis()

            # Add timestamp and metadata
            self.results["metadata"] = {
                "timestamp": self.timestamp,
                "analysis_version": "1.0",
                "building_id": building_data.get("building_id", "Unknown"),
                "analysis_type": "comprehensive",
            }

            self.logger.info("Comprehensive analysis completed successfully")
            return self.results

        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {str(e)}")
            raise

    def _perform_energy_analysis(self):
        """Consolidated energy consumption analysis"""
        try:
            energy_results = {
                'Ebase': self.power_consumed * self.duration_hours,
                'timestamp': self.timestamp
            }

            # Calculate seasonal adjusted energy (limit the impact)
            seasonal_factor = max(0.8, min(1.2, self.seasonal_factor))
            energy_results['Eseasonal'] = energy_results['Ebase'] * seasonal_factor

            # Calculate occupancy adjusted energy (limit the impact)
            occupancy_factor = max(0.6, min(1.0, self.occupancy_factor))
            energy_results['Eoccupied'] = energy_results['Eseasonal'] * occupancy_factor

            # Calculate real energy consumption
            power_factor = max(0.8, min(1.0, self.power_factor))
            energy_results['Ereal'] = energy_results['Eoccupied'] * power_factor

            # Calculate efficiency metrics with reasonable limits
            energy_results['efficiency_metrics'] = {
                'overall_efficiency': min(100, (energy_results['Ereal'] / energy_results['Ebase']) * 100),
                'seasonal_impact': min(20, max(-20, ((energy_results['Eseasonal'] - energy_results['Ebase'])
                                    / energy_results['Ebase']) * 100)),
                'occupancy_impact': min(20, max(-20, ((energy_results['Eoccupied'] - energy_results['Eseasonal'])
                                    / energy_results['Eseasonal']) * 100))
            }

            return energy_results
        except Exception as e:
            self.logger.error(f"Energy analysis failed: {str(e)}")
            raise

    def _perform_hvac_analysis(self, hvac_data):
        """Consolidated HVAC system analysis"""
        try:
            hvac_results = {
                'timestamp': self.timestamp,
                'system_efficiency': {
                    'current_efficiency': min(100, hvac_data.get('output_capacity', 0) /
                                        max(1, hvac_data.get('input_power', 1)) * 100),
                    'target_efficiency': hvac_data.get('target_efficiency', 95),
                    'efficiency_gap': 0  # Will be calculated below
                },
                'energy_consumption': self._calculate_hvac_energy(hvac_data)
            }

            # Calculate efficiency gap
            hvac_results['system_efficiency']['efficiency_gap'] = (
                hvac_results['system_efficiency']['target_efficiency'] -
                hvac_results['system_efficiency']['current_efficiency']
            )

            return hvac_results
        except Exception as e:
            self.logger.error(f"HVAC analysis failed: {str(e)}")
            raise

    def _calculate_heat_transfer(self, U, A, delta_T):
        """Calculate heat transfer for a surface"""
        return U * A * delta_T

    def _calculate_hvac_energy(self, hvac_data):
        """Calculate HVAC energy consumption"""
        V = hvac_data.get("volume", 0)
        rho = hvac_data.get("air_density", 1.225)  # kg/m³
        Cp = hvac_data.get("specific_heat", 1005)  # J/kg·K
        delta_T = hvac_data.get("temperature_difference", 0)
        eta = hvac_data.get("system_efficiency", 0.8)

        return (V * rho * Cp * abs(delta_T)) / (eta * 3600000)

    def _perform_lighting_analysis(self, lighting_data):
        """Consolidated lighting system analysis"""
        self.logger.info("Performing lighting analysis")

        try:
            lighting_results = {
                "timestamp": self.timestamp,
                "fixtures": {},
                "total_consumption": 0,
                "efficiency_metrics": {
                    "average_efficiency": 0,
                    "total_annual_consumption": 0,
                    "total_annual_cost": 0,
                },
            }

            # Check if we have valid lighting data
            if not lighting_data or "fixtures" not in lighting_data:
                self.logger.warning("No lighting fixture data available for analysis")
                return lighting_results

            # Analyze each fixture
            for fixture in lighting_data.get("fixtures", []):
                # Validate required fields
                if not all(
                    key in fixture for key in ["name", "watts", "hours", "lumens"]
                ):
                    continue

                fixture_analysis = {
                    "consumption": fixture["watts"] * fixture["hours"] / 1000,  # kWh
                    "efficiency": fixture["lumens"] / fixture["watts"],  # lm/W
                    "annual_cost": (
                        fixture["watts"]
                        * fixture["hours"]
                        * fixture.get("electricity_rate", 0.12)
                        / 1000
                    ),
                }

                lighting_results["fixtures"][fixture["name"]] = fixture_analysis
                lighting_results["total_consumption"] += fixture_analysis["consumption"]

            # Calculate overall metrics if we have fixtures
            if lighting_results["fixtures"]:
                total_power = sum(f.get("watts", 0) for f in lighting_data["fixtures"])
                total_lumens = sum(
                    f.get("lumens", 0) for f in lighting_data["fixtures"]
                )

                lighting_results["efficiency_metrics"] = {
                    "average_efficiency": (
                        total_lumens / total_power if total_power > 0 else 0
                    ),
                    "total_annual_consumption": lighting_results["total_consumption"],
                    "total_annual_cost": sum(
                        f["annual_cost"] for f in lighting_results["fixtures"].values()
                    ),
                }

            return lighting_results

        except Exception as e:
            self.logger.error(f"Lighting analysis failed: {str(e)}")
            return {
                "timestamp": self.timestamp,
                "fixtures": {},
                "total_consumption": 0,
                "efficiency_metrics": {
                    "average_efficiency": 0,
                    "total_annual_consumption": 0,
                    "total_annual_cost": 0,
                },
            }

    def _analyze_fixture(self, fixture):
        """Analyze individual lighting fixture"""
        return {
            "consumption": fixture["power"] * fixture["hours"] / 1000,  # kWh
            "efficiency": fixture["lumens"] / fixture["power"],  # lm/W
            "annual_cost": (
                fixture["power"]
                * fixture["hours"]
                * fixture.get("electricity_rate", 0.12)
                / 1000
            ),
        }

    def _calculate_lighting_metrics(self, fixtures):
        """Calculate overall lighting metrics"""
        total_power = sum(f["power"] for f in fixtures.values())
        total_lumens = sum(f["lumens"] for f in fixtures.values())

        return {
            "average_efficiency": total_lumens / total_power if total_power > 0 else 0,
            "total_annual_consumption": sum(
                f["consumption"] for f in fixtures.values()
            ),
            "total_annual_cost": sum(f["annual_cost"] for f in fixtures.values()),
        }

    def _check_humidity_data(self, building_data):
        """Check if required humidity analysis data is present"""
        required_fields = ["volume", "current_humidity", "target_humidity"]
        return all(field in building_data for field in required_fields)

    def _calculate_overall_efficiency_score(self):
        """Calculate overall building efficiency score"""
        scores = {
            'energy_score': min(100, self._calculate_energy_score()),
            'hvac_score': min(100, self._calculate_hvac_score()),
            'lighting_score': min(100, self._calculate_lighting_score()),
            'humidity_score': min(100, self._calculate_humidity_score())
        }

        weights = {
            'energy_score': 0.4,
            'hvac_score': 0.3,
            'lighting_score': 0.2,
            'humidity_score': 0.1
        }

        total_score = 0
        applicable_weight = 0

        for component, score in scores.items():
            if score is not None:
                total_score += score * weights[component]
                applicable_weight += weights[component]

        if applicable_weight == 0:
            return None

        final_score = min(100, (total_score / applicable_weight))

        return {
            'overall_score': final_score,
            'component_scores': scores,
            'interpretation': self._interpret_efficiency_score(final_score)
        }

    def _perform_financial_analysis(self):
        """Perform financial analysis of all recommendations"""
        financial_results = {
            "total_investment_required": 0,
            "annual_savings_potential": 0,
            "simple_payback_period": 0,
            "roi": 0,
            "component_analysis": {},
        }

        try:
            # Analyze each component's financial impact
            if "recommendations" in self.results:
                recommendations = self.results["recommendations"]

                # Handle immediate actions
                if "immediate_actions" in recommendations:
                    category_financials = self._analyze_category_financials(
                        recommendations["immediate_actions"]
                    )
                    financial_results["component_analysis"][
                        "immediate"
                    ] = category_financials
                    financial_results[
                        "total_investment_required"
                    ] += category_financials["investment"]
                    financial_results[
                        "annual_savings_potential"
                    ] += category_financials["annual_savings"]

                # Handle short term actions
                if "short_term" in recommendations:
                    category_financials = self._analyze_category_financials(
                        recommendations["short_term"]
                    )
                    financial_results["component_analysis"][
                        "short_term"
                    ] = category_financials
                    financial_results[
                        "total_investment_required"
                    ] += category_financials["investment"]
                    financial_results[
                        "annual_savings_potential"
                    ] += category_financials["annual_savings"]

                # Handle long term actions
                if "long_term" in recommendations:
                    category_financials = self._analyze_category_financials(
                        recommendations["long_term"]
                    )
                    financial_results["component_analysis"][
                        "long_term"
                    ] = category_financials
                    financial_results[
                        "total_investment_required"
                    ] += category_financials["investment"]
                    financial_results[
                        "annual_savings_potential"
                    ] += category_financials["annual_savings"]

            # Calculate overall metrics
            if financial_results["annual_savings_potential"] > 0:
                financial_results["simple_payback_period"] = (
                    financial_results["total_investment_required"]
                    / financial_results["annual_savings_potential"]
                )
                financial_results["roi"] = (
                    financial_results["annual_savings_potential"]
                    / financial_results["total_investment_required"]
                ) * 100

            return financial_results

        except Exception as e:
            self.logger.error(f"Financial analysis failed: {str(e)}")
            return financial_results

    def _analyze_category_financials(self, recommendations):
        """Analyze financial metrics for a category of recommendations"""
        try:
            if not recommendations or not isinstance(recommendations, list):
                return {"investment": 0, "annual_savings": 0, "average_payback": 0}

            valid_recommendations = [
                rec for rec in recommendations if isinstance(rec, dict)
            ]

            investment = sum(
                rec.get("implementation_cost", 0) for rec in valid_recommendations
            )

            annual_savings = sum(
                rec.get("estimated_savings", 0) for rec in valid_recommendations
            )

            return {
                "investment": investment,
                "annual_savings": annual_savings,
                "average_payback": self._calculate_average_payback(
                    valid_recommendations
                ),
            }

        except Exception as e:
            self.logger.error(f"Category financial analysis failed: {str(e)}")
            return {"investment": 0, "annual_savings": 0, "average_payback": 0}

    def _calculate_average_payback(self, recommendations):
        """Calculate average payback period for a set of recommendations"""
        try:
            if not recommendations:
                return 0

            valid_paybacks = []
            for rec in recommendations:
                if not isinstance(rec, dict):
                    continue

                implementation_cost = rec.get("implementation_cost", 0)
                annual_savings = rec.get("annual_savings", 0)

                if annual_savings > 0:
                    valid_paybacks.append(implementation_cost / annual_savings)

            return sum(valid_paybacks) / len(valid_paybacks) if valid_paybacks else 0

        except Exception as e:
            self.logger.error(f"Payback calculation failed: {str(e)}")
            return 0

    def _interpret_efficiency_score(self, score):
        """Interpret the overall efficiency score"""
        if score >= 90:
            return "Excellent - High-performance building"
        elif score >= 80:
            return "Very Good - Above average performance"
        elif score >= 70:
            return "Good - Meeting standard requirements"
        elif score >= 60:
            return "Fair - Room for improvement"
        else:
            return "Poor - Significant improvements needed"

    def _generate_comprehensive_recommendations(self, building_data):
        """Generate comprehensive recommendations based on all analyses"""
        self.logger.info("Generating comprehensive recommendations")

        recommendations = {
            "immediate_actions": [],
            "short_term": [],
            "long_term": [],
            "product_recommendations": {},
            "estimated_savings": {},
        }

        try:
            # Energy recommendations
            if self.results.get("energy"):
                energy_recs = self._generate_energy_recommendations()
                if energy_recs:
                    self._categorize_recommendations(energy_recs, recommendations)

            # HVAC recommendations
            if self.results.get("hvac"):
                hvac_recs = self._generate_hvac_recommendations()
                if hvac_recs:
                    self._categorize_recommendations(hvac_recs, recommendations)

            # Lighting recommendations
            if self.results.get("lighting"):
                lighting_recs = self._generate_lighting_recommendations()
                if lighting_recs:
                    self._categorize_recommendations(lighting_recs, recommendations)

            # Humidity recommendations
            if self.results.get("humidity"):
                humidity_recs = self._generate_humidity_recommendations()
                if humidity_recs:
                    self._categorize_recommendations(humidity_recs, recommendations)

            # Product recommendations
            if self.product_recommender:
                recommendations["product_recommendations"] = (
                    self.product_recommender.recommend_products(building_data)
                )

            # Calculate estimated savings
            recommendations["estimated_savings"] = self._calculate_total_savings(
                recommendations
            )

            return recommendations

        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
            return recommendations

    def _generate_energy_recommendations(self):
        """Generate energy-specific recommendations"""
        recommendations = []
        energy_results = self.results["energy"]

        # Check overall efficiency
        if energy_results["efficiency_metrics"]["overall_efficiency"] < 80:
            recommendations.append(
                {
                    "category": "energy",
                    "priority": "high",
                    "title": "Improve Overall Energy Efficiency",
                    "description": "Implementation of energy management system recommended",
                    "estimated_savings": self._estimate_energy_savings(),
                    "implementation_cost": self._estimate_implementation_cost(
                        "energy_management"
                    ),
                    "payback_period": None,  # Will be calculated later
                }
            )

        # Check seasonal impact
        if abs(energy_results["efficiency_metrics"]["seasonal_impact"]) > 20:
            recommendations.append(
                {
                    "category": "energy",
                    "priority": "medium",
                    "title": "Optimize Seasonal Energy Usage",
                    "description": "Implement seasonal adjustment strategies",
                    "estimated_savings": self._estimate_seasonal_savings(),
                    "implementation_cost": self._estimate_implementation_cost(
                        "seasonal_optimization"
                    ),
                    "payback_period": None,
                }
            )

        return recommendations

    def _generate_hvac_recommendations(self):
        """Generate HVAC-specific recommendations"""
        recommendations = []
        hvac_results = self.results["hvac"]

        # Check system efficiency
        if hvac_results["system_efficiency"]["efficiency_gap"] > 10:
            recommendations.append(
                {
                    "category": "hvac",
                    "priority": "high",
                    "title": "HVAC System Upgrade Required",
                    "description": "Current system operating below optimal efficiency",
                    "estimated_savings": self._estimate_hvac_savings(),
                    "implementation_cost": self._estimate_implementation_cost(
                        "hvac_upgrade"
                    ),
                    "payback_period": None,
                }
            )

        return recommendations

    def _generate_lighting_recommendations(self):
        """Generate lighting-specific recommendations"""
        recommendations = []
        lighting_results = self.results["lighting"]

        # Check lighting efficiency
        if lighting_results["efficiency_metrics"]["average_efficiency"] < 80:
            recommendations.append(
                {
                    "category": "lighting",
                    "priority": "medium",
                    "title": "Lighting System Upgrade",
                    "description": "Upgrade to more efficient lighting systems",
                    "estimated_savings": self._estimate_lighting_savings(),
                    "implementation_cost": self._estimate_implementation_cost(
                        "lighting_upgrade"
                    ),
                    "payback_period": None,
                }
            )

        return recommendations

    def _perform_humidity_analysis(self, building_data):
        """Perform comprehensive humidity analysis"""
        self.logger.info("Performing humidity analysis")

        try:
            humidity_results = {
                "requirements": {},
                "recommendations": [],
                "product_needs": {},
                "current_status": self._analyze_current_humidity(building_data),
            }

            # Determine requirements
            humidity_results["requirements"] = self._determine_humidity_requirements(
                building_data, humidity_results["current_status"]
            )

            # Calculate dehumidification needs
            if humidity_results["requirements"]["needs_dehumidification"]:
                humidity_results["product_needs"] = (
                    self._calculate_dehumidification_needs(
                        building_data,
                        humidity_results["current_status"],
                        humidity_results["requirements"],
                    )
                )

            return humidity_results

        except Exception as e:
            self.logger.error(f"Humidity analysis failed: {str(e)}")
            raise

    def _analyze_current_humidity(self, building_data):
        """Analyze current humidity conditions"""
        return {
            "current_humidity": building_data["current_humidity"],
            "humidity_ratio": self._calculate_humidity_ratio(building_data),
            "dew_point": self._calculate_dew_point(building_data),
            "vapor_pressure": self._calculate_vapor_pressure(building_data),
        }

    def _determine_humidity_requirements(self, building_data, current_status):
        """Determine humidity control requirements"""
        target_humidity = building_data["target_humidity"]
        current_humidity = current_status["current_humidity"]

        return {
            "target_humidity": target_humidity,
            "humidity_gap": current_humidity - target_humidity,
            "needs_dehumidification": current_humidity > target_humidity,
            "needs_humidification": current_humidity < target_humidity,
            "control_priority": self._determine_humidity_priority(
                current_humidity, target_humidity
            ),
        }

    def _calculate_dehumidification_needs(
        self, building_data, current_status, requirements
    ):
        """Calculate specific dehumidification requirements"""
        volume = building_data["volume"]
        humidity_gap = requirements["humidity_gap"]

        capacity_needed = self.calculate_dehumidification_needs(
            volume, current_status["current_humidity"], requirements["target_humidity"]
        )

        return {
            "capacity_needed": capacity_needed,
            "recommended_capacity": capacity_needed * 1.2,  # 20% safety factor
            "unit_size": self._determine_unit_size(capacity_needed),
            "estimated_runtime": self._estimate_runtime(capacity_needed, humidity_gap),
        }

    def generate_visualizations(self):
        """Generate all analysis visualizations"""
        self.logger.info("Generating analysis visualizations")

        visualizations = {}

        # Energy visualizations
        if 'energy' in self.results:
            visualizations['energy_consumption'] = self.results['energy']
            visualizations['efficiency_metrics'] = self.results['energy']

        # HVAC visualizations
        if 'hvac' in self.results:
            visualizations['hvac_performance'] = self.results['hvac']

        # Lighting visualizations
        if 'lighting' in self.results:
            visualizations['lighting_efficiency'] = self.results['lighting']

        # Humidity visualizations
        if 'humidity' in self.results:
            visualizations['humidity_analysis'] = self.results['humidity']

        return visualizations

    def _create_energy_visualizations(self):
        """Create energy analysis visualizations"""
        try:
            # Energy consumption breakdown
            fig, ax = plt.subplots(figsize=(10, 6))
            energy_values = [
                self.results["energy"]["Ebase"],
                self.results["energy"]["Eseasonal"],
                self.results["energy"]["Eoccupied"],
                self.results["energy"]["Ereal"],
            ]
            labels = ["Base", "Seasonal", "Occupied", "Real"]

            bars = ax.bar(labels, energy_values)
            ax.set_title("Energy Consumption Breakdown")
            ax.set_ylabel("Energy (kWh)")

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:,.0f}",
                    ha="center",
                    va="bottom",
                )

            energy_viz = {"consumption_breakdown": fig}
            # Efficiency metrics radar chart
            metrics = self.results["energy"]["efficiency_metrics"]
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
            values = list(metrics.values())

            ax.plot(angles, values)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles)
            ax.set_xticklabels(list(metrics.keys()))
            ax.set_title("Efficiency Metrics")

            energy_viz["efficiency_radar"] = fig

            return energy_viz

        except Exception as e:
            self.logger.error(f"Energy visualization creation failed: {str(e)}")
            return None

    def _create_hvac_visualizations(self):
        """Create HVAC analysis visualizations"""
        try:
            if "hvac" not in self.results:
                return None

            # System efficiency comparison
            fig, ax = plt.subplots(figsize=(8, 6))
            efficiency_data = self.results["hvac"]["system_efficiency"]

            x = ["Current", "Target"]
            y = [
                efficiency_data["current_efficiency"],
                efficiency_data["target_efficiency"],
            ]

            bars = ax.bar(x, y)
            ax.set_title("HVAC System Efficiency Comparison")
            ax.set_ylabel("Efficiency (%)")

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                )

            hvac_viz = {"efficiency_comparison": fig}
            # Heat transfer distribution
            if "heat_transfer" in self.results["hvac"]:
                fig, ax = plt.subplots(figsize=(10, 6))
                heat_data = self.results["hvac"]["heat_transfer"]

                plt.pie(heat_data.values(), labels=heat_data.keys(), autopct="%1.1f%%")
                plt.title("Heat Transfer Distribution")

                hvac_viz["heat_transfer"] = fig

            return hvac_viz

        except Exception as e:
            self.logger.error(f"HVAC visualization creation failed: {str(e)}")
            return None

    def _create_lighting_visualizations(self):
        """Create lighting analysis visualizations"""
        try:
            if "lighting" not in self.results:
                return None

            # Fixture efficiency comparison
            fig, ax = plt.subplots(figsize=(12, 6))
            fixtures = self.results["lighting"]["fixtures"]

            names = list(fixtures.keys())
            efficiencies = [f["efficiency"] for f in fixtures.values()]

            bars = ax.bar(names, efficiencies)
            ax.set_title("Lighting Fixture Efficiency Comparison")
            ax.set_ylabel("Efficiency (lm/W)")
            plt.xticks(rotation=45)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                )

            lighting_viz = {"fixture_efficiency": fig}
            # Energy consumption distribution
            fig, ax = plt.subplots(figsize=(8, 8))
            consumption = [f["consumption"] for f in fixtures.values()]

            plt.pie(consumption, labels=names, autopct="%1.1f%%")
            plt.title("Lighting Energy Consumption Distribution")

            lighting_viz["consumption_distribution"] = fig

            return lighting_viz

        except Exception as e:
            self.logger.error(f"Lighting visualization creation failed: {str(e)}")
            return None

    def _create_humidity_visualizations(self):
        """Create humidity analysis visualizations"""
        try:
            if "humidity" not in self.results:
                return None

            # Current vs Target Humidity
            fig, ax = plt.subplots(figsize=(8, 6))
            current = self.results["humidity"]["current_status"]["current_humidity"]
            target = self.results["humidity"]["requirements"]["target_humidity"]

            bars = ax.bar(["Current", "Target"], [current, target])
            ax.set_title("Humidity Levels Comparison")
            ax.set_ylabel("Relative Humidity (%)")

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                )

            return {"humidity_comparison": fig}
        except Exception as e:
            self.logger.error(f"Humidity visualization creation failed: {str(e)}")
            return None

    def _create_savings_visualizations(self):
        """Create savings and recommendations visualizations"""
        try:
            if "recommendations" not in self.results:
                return None

            # Potential savings by category
            fig, ax = plt.subplots(figsize=(10, 6))
            savings = self.results["recommendations"]["estimated_savings"]

            categories = list(savings.keys())
            values = list(savings.values())

            bars = ax.bar(categories, values)
            ax.set_title("Estimated Annual Savings by Category")
            ax.set_ylabel("Savings ($)")
            plt.xticks(rotation=45)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"${height:,.0f}",
                    ha="center",
                    va="bottom",
                )

            return {"savings_by_category": fig}
        except Exception as e:
            self.logger.error(f"Savings visualization creation failed: {str(e)}")
            return None

    def generate_comprehensive_report(self, include_visualizations=True, export_format='pdf'):
        """Generate comprehensive analysis report"""
        self.logger.info(f"Generating comprehensive report in {export_format} format")

        try:
            report = {
                'metadata': self._generate_report_metadata(),
                'executive_summary': self._generate_executive_summary(),
                'detailed_analysis': self._generate_detailed_analysis(),
                'recommendations': self._generate_recommendations_summary(),
                'financial_analysis': self._generate_financial_summary()
            }

            if include_visualizations:
                report['visualizations'] = self.generate_visualizations()

            # Export report in specified format
            exported_file = self._export_report(report, export_format)

            return {
                'report_data': report,
                'exported_file': exported_file
            }

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise

    def _generate_report_metadata(self):
        """Generate report metadata"""
        return {
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "report_version": "1.0",
            "analysis_type": "comprehensive",
            "generated_by": "Energy Audit Tool",
            "report_id": f"EAT-{self.timestamp.strftime('%Y%m%d%H%M%S')}",
        }

    def _generate_executive_summary(self):
        """Generate executive summary of findings"""
        return {
            'overview': {
                'total_energy_consumption': self.results['energy']['Ereal'],
                'overall_efficiency_score': self.results['efficiency_score']['overall_score'],
                'potential_annual_savings': sum(
                    self.results['recommendations']['estimated_savings'].values()
                ) if 'recommendations' in self.results else 0
            },
            'key_findings': self._generate_key_findings(),
            'priority_recommendations': self._get_priority_recommendations(),
            'financial_highlights': self._generate_financial_summary()
        }

    def _generate_detailed_analysis(self):
        """Generate detailed analysis section"""
        return {
            "energy_analysis": self._format_energy_analysis(),
            "hvac_analysis": self._format_hvac_analysis(),
            "lighting_analysis": self._format_lighting_analysis(),
            "humidity_analysis": self._format_humidity_analysis(),
        }

    def _format_energy_analysis(self):
        """Format energy analysis results for report"""
        energy_results = self.results["energy"]

        return {
            "consumption_breakdown": {
                "base_consumption": energy_results["Ebase"],
                "seasonal_adjusted": energy_results["Eseasonal"],
                "occupancy_adjusted": energy_results["Eoccupied"],
                "real_consumption": energy_results["Ereal"],
            },
            "efficiency_metrics": energy_results["efficiency_metrics"],
            "analysis_notes": self._generate_energy_analysis_notes(),
        }

    def _format_hvac_analysis(self):
        """Format HVAC analysis results for report"""
        if "hvac" not in self.results:
            return {
                "system_efficiency": {},
                "energy_consumption": 0,
                "heat_transfer": {},
                "notes": [],
            }

        hvac_results = self.results["hvac"]

        formatted_results = {
            "system_efficiency": hvac_results.get("system_efficiency", {}),
            "energy_consumption": hvac_results.get("energy_consumption", 0),
            "heat_transfer": hvac_results.get("heat_transfer", {}),
            "notes": self._generate_hvac_analysis_notes(),
        }

        # Add efficiency ratings
        if "system_efficiency" in hvac_results:
            efficiency = hvac_results["system_efficiency"]
            formatted_results["efficiency_ratings"] = {
                "current": efficiency.get("current_efficiency", 0),
                "target": efficiency.get("target_efficiency", 0),
                "gap": efficiency.get("efficiency_gap", 0),
            }

        # Add recommendations if efficiency gap is significant
        if formatted_results["efficiency_ratings"]["gap"] > 10:
            formatted_results["recommendations"] = [
                {
                    "type": "efficiency_improvement",
                    "description": "System efficiency below target - upgrade recommended",
                    "priority": (
                        "high"
                        if formatted_results["efficiency_ratings"]["gap"] > 20
                        else "medium"
                    ),
                }
            ]

        return formatted_results

    def _generate_hvac_analysis_notes(self):
        """Generate analysis notes for HVAC system"""
        notes = []

        if "hvac" not in self.results:
            return notes

        hvac_results = self.results["hvac"]

        # Check system efficiency
        if "system_efficiency" in hvac_results:
            efficiency = hvac_results["system_efficiency"]
            if efficiency.get("efficiency_gap", 0) > 20:
                notes.append("Critical: System efficiency significantly below target")
            elif efficiency.get("efficiency_gap", 0) > 10:
                notes.append("Warning: System efficiency below optimal range")

        # Check energy consumption
        if "energy_consumption" in hvac_results:
            consumption = hvac_results["energy_consumption"]
            if consumption > 100:  # Example threshold
                notes.append("High energy consumption detected")

        # Check heat transfer
        if "heat_transfer" in hvac_results:
            heat_transfer = hvac_results["heat_transfer"]
            for surface, value in heat_transfer.items():
                if value > 1000:  # Example threshold
                    notes.append(f"High heat transfer through {surface}")

        return notes

    def _format_humidity_analysis(self):
        """Format humidity analysis results for report"""
        if "humidity" not in self.results:
            return {
                "current_status": {},
                "requirements": {},
                "recommendations": [],
                "notes": [],
            }

        humidity_results = self.results["humidity"]

        formatted_results = {
            "current_status": humidity_results.get("current_status", {}),
            "requirements": humidity_results.get("requirements", {}),
            "product_needs": humidity_results.get("product_needs", {}),
            "notes": self._generate_humidity_analysis_notes(humidity_results),
        }

        return formatted_results

    def _generate_humidity_analysis_notes(self, humidity_results):
        """Generate analysis notes for humidity control"""
        notes = []

        if not humidity_results:
            return notes

        # Check humidity levels
        current_status = humidity_results.get("current_status", {})
        requirements = humidity_results.get("requirements", {})

        if requirements.get("needs_dehumidification"):
            humidity_gap = requirements.get("humidity_gap", 0)
            if humidity_gap > 20:
                notes.append("Critical: Humidity levels significantly above target")
            elif humidity_gap > 10:
                notes.append("Warning: Humidity levels above comfortable range")

        # Check capacity needs
        product_needs = humidity_results.get("product_needs", {})
        if product_needs.get("capacity_needed", 0) > 0:
            notes.append(
                f"Required dehumidification capacity: "
                f"{product_needs['capacity_needed']:.1f} pints/day"
            )

        return notes

    def _format_lighting_analysis(self):
        """Format lighting analysis results for report"""
        if "lighting" not in self.results:
            return {
                "fixtures": {},
                "total_consumption": 0,
                "efficiency_metrics": {},
                "notes": [],
            }

        lighting_results = self.results["lighting"]

        formatted_results = {
            "fixtures": self._format_fixture_details(
                lighting_results.get("fixtures", {})
            ),
            "total_consumption": lighting_results.get("total_consumption", 0),
            "efficiency_metrics": lighting_results.get("efficiency_metrics", {}),
            "notes": self._generate_lighting_analysis_notes(lighting_results),
        }

        # Add summary statistics
        formatted_results["summary"] = {
            "total_fixtures": len(lighting_results.get("fixtures", {})),
            "average_efficiency": self._calculate_average_lighting_efficiency(
                lighting_results.get("fixtures", {})
            ),
            "annual_cost": lighting_results.get("efficiency_metrics", {}).get(
                "total_annual_cost", 0
            ),
        }

        # Add recommendations if needed
        formatted_results["recommendations"] = (
            self._generate_lighting_recommendations_summary(formatted_results)
        )

        return formatted_results

    def _format_fixture_details(self, fixtures):
        """Format detailed fixture information"""
        formatted_fixtures = {}

        for name, fixture in fixtures.items():
            formatted_fixtures[name] = {
                "consumption": fixture.get("consumption", 0),
                "efficiency": fixture.get("efficiency", 0),
                "annual_cost": fixture.get("annual_cost", 0),
                "performance_rating": self._rate_fixture_performance(fixture),
            }

        return formatted_fixtures

    def _rate_fixture_performance(self, fixture):
        """Rate fixture performance based on efficiency"""
        efficiency = fixture.get("efficiency", 0)

        if efficiency >= 100:
            return "Excellent"
        elif efficiency >= 80:
            return "Good"
        elif efficiency >= 60:
            return "Fair"
        else:
            return "Poor"

    def _calculate_average_lighting_efficiency(self, fixtures):
        """Calculate average lighting efficiency across all fixtures"""
        if not fixtures:
            return 0

        efficiencies = [f.get("efficiency", 0) for f in fixtures.values()]
        return sum(efficiencies) / len(efficiencies) if efficiencies else 0

    def _generate_lighting_analysis_notes(self, lighting_results):
        """Generate analysis notes for lighting system"""
        notes = []

        if not lighting_results:
            return notes

        # Check overall efficiency
        efficiency_metrics = lighting_results.get("efficiency_metrics", {})
        avg_efficiency = efficiency_metrics.get("average_efficiency", 0)

        if avg_efficiency < 60:
            notes.append("Critical: Overall lighting efficiency is poor")
        elif avg_efficiency < 80:
            notes.append("Warning: Lighting efficiency could be improved")

        # Check consumption
        total_consumption = lighting_results.get("total_consumption", 0)
        if total_consumption > 10000:  # Example threshold
            notes.append("High annual energy consumption for lighting")

        # Check individual fixtures
        fixtures = lighting_results.get("fixtures", {})
        poor_fixtures = [
            name
            for name, fixture in fixtures.items()
            if fixture.get("efficiency", 0) < 60
        ]

        if poor_fixtures:
            notes.append(
                f"Low efficiency fixtures identified: {', '.join(poor_fixtures)}"
            )

        return notes

    def _generate_lighting_recommendations_summary(self, formatted_results):
        """Generate summary of lighting recommendations"""
        recommendations = []

        # Check overall efficiency
        avg_efficiency = formatted_results["summary"]["average_efficiency"]
        if avg_efficiency < 80:
            recommendations.append(
                {
                    "type": "efficiency_improvement",
                    "description": "Upgrade to more efficient lighting fixtures",
                    "priority": "high" if avg_efficiency < 60 else "medium",
                }
            )

        # Check annual cost
        annual_cost = formatted_results["summary"]["annual_cost"]
        if annual_cost > 5000:  # Example threshold
            recommendations.append(
                {
                    "type": "cost_reduction",
                    "description": "Implement cost reduction measures",
                    "priority": "medium",
                }
            )

        # Check individual fixtures
        poor_fixtures = [
            name
            for name, details in formatted_results["fixtures"].items()
            if details["performance_rating"] == "Poor"
        ]

        if poor_fixtures:
            recommendations.append(
                {
                    "type": "fixture_replacement",
                    "description": f'Replace inefficient fixtures: {", ".join(poor_fixtures)}',
                    "priority": "high",
                }
            )

        return recommendations

    def _format_recommendations_pdf(self, recommendations, styles):
        """Format recommendations for PDF report"""
        elements = []

        # Immediate actions
        if recommendations.get('immediate_actions'):
            elements.append(Paragraph("Immediate Actions", styles['Heading2']))
            for rec in recommendations['immediate_actions']:
                elements.append(
                    Paragraph(
                        f"• {rec['title']}: {rec['description']}",
                        styles['Normal']
                    )
                )
                elements.append(
                    Paragraph(
                        f"  Estimated Savings: ${rec.get('estimated_savings', 0):,.2f}",
                        styles['Normal']
                    )
                )
            elements.append(Spacer(1, 12))

        # Short-term actions
        if recommendations.get('short_term'):
            elements.append(Paragraph("Short-term Actions", styles['Heading2']))
            for rec in recommendations['short_term']:
                elements.append(
                    Paragraph(
                        f"• {rec['title']}: {rec['description']}",
                        styles['Normal']
                    )
                )
            elements.append(Spacer(1, 12))

        # Long-term actions
        if recommendations.get('long_term'):
            elements.append(Paragraph("Long-term Actions", styles['Heading2']))
            for rec in recommendations['long_term']:
                elements.append(
                    Paragraph(
                        f"• {rec['title']}: {rec['description']}",
                        styles['Normal']
                    )
                )

        return elements

    def _format_detailed_analysis_pdf(self, analysis, styles):
        """Format detailed analysis for PDF report"""
        elements = []

        # Energy Analysis
        if 'energy_analysis' in analysis:
            elements.append(Paragraph("Energy Analysis", styles['Heading2']))
            energy = analysis['energy_analysis']
            data = [
                ['Metric', 'Value'],
                ['Base Consumption', f"{energy['consumption_breakdown']['base_consumption']:,.2f} kWh"],
                ['Real Consumption', f"{energy['consumption_breakdown']['real_consumption']:,.2f} kWh"],
                ['Overall Efficiency', f"{energy['efficiency_metrics']['overall_efficiency']:.1f}%"]
            ]

            table = Table(data, colWidths=[3*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(table)
            elements.append(Spacer(1, 12))

        # HVAC Analysis
        if 'hvac_analysis' in analysis:
            elements.append(Paragraph("HVAC Analysis", styles['Heading2']))
            hvac = analysis['hvac_analysis']
            data = [
                ['Metric', 'Value'],
                ['Energy Consumption', f"{hvac['energy_consumption']:,.2f} kWh"],
                ['Current Efficiency', f"{hvac['system_efficiency'].get('current_efficiency', 0):.1f}%"],
                ['Target Efficiency', f"{hvac['system_efficiency'].get('target_efficiency', 0):.1f}%"]
            ]

            table = Table(data, colWidths=[3*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(table)
            elements.append(Spacer(1, 12))

        # Lighting Analysis
        if 'lighting_analysis' in analysis:
            elements.append(Paragraph("Lighting Analysis", styles['Heading2']))
            lighting = analysis['lighting_analysis']
            data = [
                ['Metric', 'Value'],
                ['Total Consumption', f"{lighting['total_consumption']:,.2f} kWh"],
                ['Average Efficiency', f"{lighting['efficiency_metrics'].get('average_efficiency', 0):.1f} lm/W"]
            ]

            table = Table(data, colWidths=[3*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(table)
            elements.append(Spacer(1, 12))

        # Humidity Analysis
        if 'humidity_analysis' in analysis:
            elements.append(Paragraph("Humidity Analysis", styles['Heading2']))
            humidity = analysis['humidity_analysis']
            data = [
                ['Metric', 'Value'],
                ['Current Humidity', f"{humidity['current_status'].get('current_humidity', 0):.1f}%"],
                ['Target Humidity', f"{humidity['requirements'].get('target_humidity', 0):.1f}%"]
            ]

            table = Table(data, colWidths=[3*inch, 3*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            elements.append(table)
            elements.append(Spacer(1, 12))

        return elements

    def _format_visualizations_pdf(self, visualizations):
        """Format visualizations for PDF report"""
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Image, Paragraph
        import io

        styles = getSampleStyleSheet()
        elements = []
        index = 0
        for viz_type, viz_data in visualizations.items():
            try:
                if index % 2 == 0 and index > 2:
                    # Add page break after every two visualizations
                    elements.append(PageBreak())
                elements.append(Paragraph(f"{viz_type.replace('_', ' ').title()}", styles['Heading2']))

                if fig := self._create_visualization_figure(viz_type, viz_data):
                    # Save the matplotlib figure to a bytes buffer
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
                    img_buffer.seek(0)

                    # Add the image to the PDF
                    img = Image(img_buffer)
                    imgadjust = .1
                    img.drawHeight = 300 - (300*imgadjust)
                    img.drawWidth = 400 - (400*imgadjust)
                    elements.append(img)

                    # Close the figure to free memory
                    plt.close(fig)
                    index += 1

            except Exception as e:
                self.logger.error(f"Error formatting visualization {viz_type}: {str(e)}")
                continue

        return elements

    def _create_visualization_figure(self, viz_type, viz_data):
        """Create a matplotlib figure based on visualization type and data"""
        try:
            if viz_type == 'energy_consumption':
                return self._create_energy_consumption_figure(viz_data)
            elif viz_type == 'efficiency_metrics':
                return self._create_efficiency_metrics_figure(viz_data)
            elif viz_type == 'hvac_performance':
                return self._create_hvac_performance_figure(viz_data)
            elif viz_type == 'lighting_efficiency':
                return self._create_lighting_efficiency_figure(viz_data)
            elif viz_type == 'humidity_analysis':
                return self._create_humidity_analysis_figure(viz_data)
            else:
                self.logger.warning(f"Unknown visualization type: {viz_type}")
                return None
        except Exception as e:
            self.logger.error(f"Error creating visualization figure: {str(e)}")
            return None

    def _create_energy_consumption_figure(self, data):
        """Create energy consumption visualization"""
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['Base', 'Seasonal', 'Occupied', 'Real']
        values = [
            data.get('Ebase', 0),
            data.get('Eseasonal', 0),
            data.get('Eoccupied', 0),
            data.get('Ereal', 0)
        ]

        bars = ax.bar(categories, values)
        ax.set_title('Energy Consumption Breakdown')
        ax.set_ylabel('Energy (kWh)')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def _create_efficiency_metrics_figure(self, data):
        """Create efficiency metrics visualization"""
        if not data:
            return None

        metrics = data.get('efficiency_metrics', {})
        if not metrics:
            return None

        # Create radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())

        # Calculate angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)

        # Close the plot by appending first value
        values.append(values[0])
        angles = np.concatenate((angles, [angles[0]]))

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Efficiency Metrics')

        plt.tight_layout()
        return fig

    def _create_hvac_performance_figure(self, data):
        """Create HVAC performance visualization"""
        if not data:
            return None

        efficiency = data.get('system_efficiency', {})
        if not efficiency:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        categories = ['Current', 'Target']
        values = [
            efficiency.get('current_efficiency', 0),
            efficiency.get('target_efficiency', 0)
        ]

        bars = ax.bar(categories, values)
        ax.set_title('HVAC System Efficiency')
        ax.set_ylabel('Efficiency (%)')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def _create_lighting_efficiency_figure(self, data):
        """Create lighting efficiency visualization"""
        if not data:
            return None

        fixtures = data.get('fixtures', {})
        if not fixtures:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(fixtures.keys())
        efficiencies = [f.get('efficiency', 0) for f in fixtures.values()]

        bars = ax.bar(names, efficiencies)
        ax.set_title('Lighting Fixture Efficiency')
        ax.set_ylabel('Efficiency (lm/W)')
        plt.xticks(rotation=45)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def _create_humidity_analysis_figure(self, data):
        """Create humidity analysis visualization"""
        if not data:
            return None

        current_status = data.get('current_status', {})
        requirements = data.get('requirements', {})

        if not current_status or not requirements:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        categories = ['Current', 'Target']
        values = [
            current_status.get('current_humidity', 0),
            requirements.get('target_humidity', 0)
        ]

        bars = ax.bar(categories, values)
        ax.set_title('Humidity Levels')
        ax.set_ylabel('Relative Humidity (%)')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def _generate_energy_analysis_notes(self):
        """Generate notes for energy analysis"""
        notes = []

        if "energy" not in self.results:
            return notes

        energy_results = self.results["energy"]

        # Check efficiency metrics
        if "efficiency_metrics" in energy_results:
            metrics = energy_results["efficiency_metrics"]

            # Overall efficiency
            overall_efficiency = metrics.get("overall_efficiency", 0)
            if overall_efficiency < 70:
                notes.append(
                    "Critical: Overall energy efficiency below acceptable range"
                )
            elif overall_efficiency < 85:
                notes.append("Warning: Energy efficiency could be improved")

            # Seasonal impact
            seasonal_impact = abs(metrics.get("seasonal_impact", 0))
            if seasonal_impact > 20:
                notes.append("Significant seasonal variation in energy consumption")

            # Occupancy impact
            occupancy_impact = abs(metrics.get("occupancy_impact", 0))
            if occupancy_impact > 15:
                notes.append("Notable occupancy-related energy variation")

        return notes

    def _generate_financial_highlights(self):
        """Generate financial analysis highlights"""
        if "financial_analysis" not in self.results:
            return {}

        financial = self.results["financial_analysis"]

        return {
            "total_investment": financial.get("total_investment_required", 0),
            "annual_savings": financial.get("annual_savings_potential", 0),
            "simple_payback": financial.get("simple_payback_period", 0),
            "roi": financial.get("roi", 0),
            "summary": self._generate_financial_summary(financial),
        }

    def _generate_recommendations_summary(self):
        """Generate summary of all recommendations"""
        if "recommendations" not in self.results:
            return {
                "immediate_actions": [],
                "short_term": [],
                "long_term": [],
                "total_savings": 0,
                "total_investment": 0,
                "priority_actions": [],
            }

        recommendations = self.results["recommendations"]

        summary = {
            "immediate_actions": self._summarize_action_category(
                recommendations.get("immediate_actions", [])
            ),
            "short_term": self._summarize_action_category(
                recommendations.get("short_term", [])
            ),
            "long_term": self._summarize_action_category(
                recommendations.get("long_term", [])
            ),
            "product_recommendations": self._summarize_product_recommendations(
                recommendations.get("product_recommendations", {})
            ),
        }

        # Calculate totals
        summary["total_savings"] = sum(
            action.get("estimated_savings", 0)
            for category in ["immediate_actions", "short_term", "long_term"]
            for action in summary[category]
        )

        summary["total_investment"] = sum(
            action.get("implementation_cost", 0)
            for category in ["immediate_actions", "short_term", "long_term"]
            for action in summary[category]
        )

        # Get priority actions
        summary["priority_actions"] = self._get_priority_actions(summary)

        return summary

    def _summarize_action_category(self, actions):
        """Summarize actions within a category"""
        summarized_actions = []

        for action in actions:
            if not isinstance(action, dict):
                continue

            summarized_action = {
                "title": action.get("title", "Unnamed Action"),
                "description": action.get("description", ""),
                "estimated_savings": action.get("estimated_savings", 0),
                "implementation_cost": action.get("implementation_cost", 0),
                "priority": action.get("priority", "medium"),
            }

            # Calculate payback period if possible
            if summarized_action["estimated_savings"] > 0:
                summarized_action["payback_period"] = (
                    summarized_action["implementation_cost"]
                    / summarized_action["estimated_savings"]
                )
            else:
                summarized_action["payback_period"] = float("inf")

            summarized_actions.append(summarized_action)

        # Sort by estimated savings
        return sorted(
            summarized_actions, key=lambda x: x["estimated_savings"], reverse=True
        )

    def _summarize_product_recommendations(self, product_recommendations):
        """Summarize product recommendations"""
        summary = {}

        for category, products in product_recommendations.items():
            if isinstance(products, pd.DataFrame):
                summary[category] = {
                    "count": len(products),
                    "avg_efficiency": (
                        products["Efficiency_Value"].mean()
                        if "Efficiency_Value" in products.columns
                        else None
                    ),
                    "top_products": (
                        products.head(3)["Product Name"].tolist()
                        if "Product Name" in products.columns
                        else []
                    ),
                }

        return summary

    def _get_priority_actions(self, summary):
        """Get top priority actions across all categories"""
        all_actions = []

        # Collect all actions with their category
        for category in ["immediate_actions", "short_term", "long_term"]:
            for action in summary[category]:
                action["category"] = category
                all_actions.append(action)

        # Sort by priority and savings
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_actions = sorted(
            all_actions,
            key=lambda x: (
                priority_order.get(x["priority"], 999),
                -x["estimated_savings"],
            ),
        )

        # Return top 5 priority actions
        return sorted_actions[:5]

    def _format_recommendations_summary(self, recommendations_summary):
        """Format recommendations summary for the report"""
        formatted_summary = {
            "overview": {
                "total_actions": sum(
                    len(recommendations_summary[cat])
                    for cat in ["immediate_actions", "short_term", "long_term"]
                ),
                "total_savings": recommendations_summary["total_savings"],
                "total_investment": recommendations_summary["total_investment"],
                "roi": (
                    (
                        recommendations_summary["total_savings"]
                        / recommendations_summary["total_investment"]
                        * 100
                    )
                    if recommendations_summary["total_investment"] > 0
                    else 0
                ),
            },
            "priority_actions": [
                {
                    "title": action["title"],
                    "category": action["category"],
                    "savings": action["estimated_savings"],
                    "cost": action["implementation_cost"],
                }
                for action in recommendations_summary["priority_actions"]
            ],
            "category_summaries": {
                "immediate": self._summarize_category(
                    recommendations_summary["immediate_actions"]
                ),
                "short_term": self._summarize_category(
                    recommendations_summary["short_term"]
                ),
                "long_term": self._summarize_category(
                    recommendations_summary["long_term"]
                ),
            },
        }

        if "product_recommendations" in recommendations_summary:
            formatted_summary["product_recommendations"] = recommendations_summary[
                "product_recommendations"
            ]

        return formatted_summary

    def _summarize_category(self, actions):
        """Summarize a category of actions"""
        if not actions:
            return {
                "count": 0,
                "total_savings": 0,
                "total_cost": 0,
                "average_payback": 0,
            }

        total_savings = sum(action["estimated_savings"] for action in actions)
        total_cost = sum(action["implementation_cost"] for action in actions)

        valid_paybacks = [
            action["payback_period"]
            for action in actions
            if action["payback_period"] != float("inf")
        ]

        return {
            "count": len(actions),
            "total_savings": total_savings,
            "total_cost": total_cost,
            "average_payback": (
                sum(valid_paybacks) / len(valid_paybacks)
                if valid_paybacks
                else float("inf")
            ),
        }

    def _generate_financial_summary(self):
        """Generate financial analysis summary"""
        if 'financial_analysis' not in self.results:
            return {
                'total_investment': 0,
                'annual_savings': 0,
                'simple_payback': 0,
                'roi': 0,
                'summary_points': ['No financial analysis data available']
            }

        financial_data = self.results['financial_analysis']

        summary = {
            'total_investment': financial_data.get('total_investment_required', 0),
            'annual_savings': financial_data.get('annual_savings_potential', 0),
            'simple_payback': financial_data.get('simple_payback_period', 0),
            'roi': financial_data.get('roi', 0),
            'summary_points': self._generate_financial_summary_points(financial_data)
        }

        # Add component analysis if available
        if 'component_analysis' in financial_data:
            summary['component_analysis'] = self._summarize_component_financials(
                financial_data['component_analysis']
            )

        return summary

    def _summarize_component_financials(self, component_analysis):
        """Summarize financial analysis by component"""
        return {
            component: {
                'investment': data.get('investment', 0),
                'annual_savings': data.get('annual_savings', 0),
                'payback_period': (
                    data.get('investment', 0) / data.get('annual_savings', 1)
                    if data.get('annual_savings', 0) > 0
                    else float('inf')
                ),
            }
            for component, data in component_analysis.items()
        }

    def _generate_financial_summary_points(self, financial_data):
        """Generate summary points from financial analysis"""
        summary_points = []

        # Investment assessment
        total_investment = financial_data.get('total_investment_required', 0)
        if total_investment > 10000:
            summary_points.append("Major investment required")
        else:
            summary_points.append("Moderate investment required")

        # Payback assessment
        payback = financial_data.get('simple_payback_period', 0)
        if payback < 2:
            summary_points.append("Excellent payback period")
        elif payback < 5:
            summary_points.append("Good payback period")
        else:
            summary_points.append("Long-term investment")

        # ROI assessment
        roi = financial_data.get('roi', 0)
        if roi > 50:
            summary_points.append("High return on investment")
        elif roi > 20:
            summary_points.append("Good return on investment")
        else:
            summary_points.append("Moderate return on investment")

        return summary_points

    def _generate_key_findings(self):
        """Generate key findings from analysis results"""
        findings = []

        # Energy findings
        if "energy" in self.results:
            energy_efficiency = self.results["energy"]["efficiency_metrics"][
                "overall_efficiency"
            ]
            findings.append(
                {
                    "category": "Energy",
                    "finding": f"Overall energy efficiency is {energy_efficiency:.1f}%",
                    "impact": "high" if energy_efficiency < 80 else "medium",
                }
            )

        # HVAC findings
        if "hvac" in self.results:
            hvac_gap = self.results["hvac"]["system_efficiency"]["efficiency_gap"]
            findings.append(
                {
                    "category": "HVAC",
                    "finding": f"System efficiency gap is {hvac_gap:.1f}%",
                    "impact": "high" if hvac_gap > 10 else "medium",
                }
            )

        # Humidity findings
        if "humidity" in self.results:
            humidity_reqs = self.results["humidity"]["requirements"]
            if humidity_reqs.get("needs_dehumidification"):
                findings.append(
                    {
                        "category": "Humidity",
                        "finding": "Dehumidification needed",
                        "impact": humidity_reqs.get("control_priority", "medium"),
                    }
                )

        return findings

    def _export_report(self, report_data, export_format):
        """Export report in specified format"""
        try:
            if export_format.lower() == "pdf":
                return self._export_pdf_report(report_data)
            elif export_format.lower() == "excel":
                return self._export_excel_report(report_data)
            elif export_format.lower() == "json":
                return self._export_json_report(report_data)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")

        except Exception as e:
            self.logger.error(f"Report export failed: {str(e)}")
            raise

    def _export_pdf_report(self, report_data):
        """Export report as PDF"""
        try:
            filename = f"energy_audit_report_{self.timestamp.strftime('%Y%m%d')}.pdf"
            doc = SimpleDocTemplate(
                filename,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )

            # Initialize styles
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=30
            ))
            styles.add(ParagraphStyle(
                name='CustomHeading1',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12
            ))

            # Create document elements
            elements = []

            # Add title
            elements.append(Paragraph("Energy Audit Report", styles['CustomTitle']))
            elements.append(Spacer(1, 12))

            # Add metadata
            elements.extend(self._format_metadata_pdf(report_data['metadata'], styles))
            elements.append(Spacer(1, 12))

            # Add executive summary
            elements.append(Paragraph("Executive Summary", styles['CustomHeading1']))
            elements.extend(self._format_executive_summary_pdf(report_data['executive_summary'], styles))
            elements.append(Spacer(1, 12))

            # Add detailed analysis
            elements.append(Paragraph("Detailed Analysis", styles['CustomHeading1']))
            elements.extend(self._format_detailed_analysis_pdf(report_data['detailed_analysis'], styles))
            elements.append(Spacer(1, 12))

            # Add recommendations
            elements.append(Paragraph("Recommendations", styles['CustomHeading1']))
            elements.extend(self._format_recommendations_pdf(report_data['recommendations'], styles))
            elements.append(Spacer(1, 12))

            # Add visualizations if available
            if 'visualizations' in report_data:
                # add a page break before visualizations
                elements.append(PageBreak())
                elements.append(Paragraph("Analysis Visualizations", styles['CustomHeading1']))
                if viz_elements := self._format_visualizations_pdf(
                    report_data['visualizations']
                ):
                    elements.extend(viz_elements)

            # Build PDF
            doc.build(elements)
            return filename

        except Exception as e:
            self.logger.error(f"PDF export failed: {str(e)}")
            raise

    def _format_metadata_pdf(self, metadata, styles):
        """Format metadata for PDF report"""
        elements = []

        # Create metadata table
        data = [
            ['Report Date:', metadata.get('timestamp', '')],
            ['Report ID:', metadata.get('report_id', '')],
            ['Analysis Type:', metadata.get('analysis_type', '')],
            ['Version:', metadata.get('report_version', '')]
        ]

        # Create table with imported Table class
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey)
        ]))

        elements.append(table)
        return elements

    def _export_excel_report(self, report_data):
        """Export report as Excel workbook"""
        try:
            filename = f"energy_audit_report_{self.timestamp.strftime('%Y%m%d')}.xlsx"

            with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
                # Create summary sheet
                pd.DataFrame([report_data["executive_summary"]]).to_excel(
                    writer, sheet_name="Executive Summary", index=False
                )

                # Create detailed analysis sheets
                for analysis_type, data in report_data["detailed_analysis"].items():
                    pd.DataFrame([data]).to_excel(
                        writer,
                        sheet_name=analysis_type.replace("_", " ").title(),
                        index=False,
                    )

                # Create recommendations sheet
                pd.DataFrame(report_data["recommendations"]).to_excel(
                    writer, sheet_name="Recommendations", index=False
                )

                # Add visualizations if available
                if "visualizations" in report_data:
                    self._add_visualizations_to_excel(
                        writer, report_data["visualizations"]
                    )

            return filename

        except Exception as e:
            self.logger.error(f"Excel export failed: {str(e)}")
            raise

    def _export_json_report(self, report_data):
        """Export report as JSON"""
        try:
            filename = f"energy_audit_report_{self.timestamp.strftime('%Y%m%d')}.json"

            # Convert visualizations to base64 if present
            if "visualizations" in report_data:
                report_data["visualizations"] = self._convert_visualizations_to_base64(
                    report_data["visualizations"]
                )

            with open(filename, "w") as f:
                json.dump(report_data, f, indent=4, default=str)

            return filename

        except Exception as e:
            self.logger.error(f"JSON export failed: {str(e)}")
            raise

    def export_data_to_csv(self, data_type="all"):
        """Export specific analysis data to CSV"""
        try:
            if data_type == "all":
                return self._export_all_data_csv()
            elif data_type in self.results:
                return self._export_specific_data_csv(data_type)
            else:
                raise ValueError(f"Invalid data type: {data_type}")

        except Exception as e:
            self.logger.error(f"CSV export failed: {str(e)}")
            raise

    def _format_executive_summary_pdf(self, summary, styles):
        """Format executive summary for PDF report"""
        elements = []

        # Overview section
        overview = summary.get('overview', {})
        data = [
            ['Metric', 'Value'],
            ['Total Energy Consumption', f"{overview.get('total_energy_consumption', 0):,.2f} kWh"],
            ['Overall Efficiency Score', f"{overview.get('overall_efficiency_score', 0):.1f}"],
            ['Potential Annual Savings', f"${overview.get('potential_annual_savings', 0):,.2f}"]
        ]

        table = Table(data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(table)
        elements.append(Spacer(1, 12))

        # Key findings
        elements.append(Paragraph("Key Findings", styles['Heading2']))
        for finding in summary.get('key_findings', []):
            elements.append(
                Paragraph(
                    f"• {finding.get('category')}: {finding.get('finding')}",
                    styles['Normal']
                )
            )

        return elements

    def _add_visualizations_to_excel(self, writer, visualizations):
        """Add visualizations to Excel workbook"""
        workbook = writer.book
        worksheet = workbook.add_worksheet("Visualizations")

        row = 0
        for viz_type, fig in visualizations.items():
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png")
            worksheet.insert_image(row, 0, viz_type, {"image_data": img_buffer})
            row += 40  # Leave space for next image

    def _convert_visualizations_to_base64(self, visualizations):
        """Convert matplotlib figures to base64 strings"""
        converted = {}
        for viz_type, fig in visualizations.items():
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png")
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            converted[viz_type] = img_str
        return converted

    def run_analysis(
        self, building_data, export_format="pdf", include_visualizations=True
    ):
        """Main method to run the complete analysis and generate report"""
        self.logger.info("Starting complete analysis run")

        try:
            # Perform comprehensive analysis
            analysis_results = self.perform_comprehensive_analysis(building_data)

            # Generate and export report
            report = self.generate_comprehensive_report(
                include_visualizations=include_visualizations,
                export_format=export_format,
            )

            return {"analysis_results": analysis_results, "report": report}

        except Exception as e:
            self.logger.error(f"Analysis run failed: {str(e)}")
            raise

    def _calculate_energy_score(self):
        """Calculate energy efficiency score"""
        try:
            energy_results = self.results["energy"]
            base_efficiency = (energy_results["Ereal"] / energy_results["Ebase"]) * 100
            return min(100, max(0, 100 - (100 - base_efficiency) * 1.5))
        except Exception:
            return None

    def _calculate_hvac_score(self):
        """Calculate HVAC efficiency score"""
        try:
            hvac_results = self.results.get("hvac", {})
            if not hvac_results:
                return None
            efficiency = hvac_results["system_efficiency"]["current_efficiency"]
            target = hvac_results["system_efficiency"]["target_efficiency"]
            return min(100, max(0, (efficiency / target) * 100))
        except Exception:
            return None

    def _calculate_lighting_score(self):
        """Calculate lighting efficiency score"""
        try:
            lighting_results = self.results.get("lighting", {})
            if not lighting_results:
                return None
            metrics = lighting_results["efficiency_metrics"]
            return min(100, max(0, metrics["average_efficiency"]))
        except Exception:
            return None

    def _calculate_humidity_score(self):
        """Calculate humidity control score"""
        try:
            humidity_results = self.results.get("humidity", {})
            if not humidity_results:
                return None
            current = humidity_results["current_status"]["current_humidity"]
            target = humidity_results["requirements"]["target_humidity"]
            deviation = abs(current - target)
            return min(100, max(0, 100 - deviation * 2))
        except Exception:
            return None

    def _generate_key_findings(self):
        """Generate key findings from analysis results"""
        findings = []

        # Energy findings
        if "energy" in self.results:
            energy_efficiency = self.results["energy"]["efficiency_metrics"][
                "overall_efficiency"
            ]
            findings.append(
                {
                    "category": "Energy",
                    "finding": f"Overall energy efficiency is {energy_efficiency:.1f}%",
                    "impact": "high" if energy_efficiency < 80 else "medium",
                }
            )

        # HVAC findings
        if "hvac" in self.results:
            hvac_gap = self.results["hvac"]["system_efficiency"]["efficiency_gap"]
            findings.append(
                {
                    "category": "HVAC",
                    "finding": f"System efficiency gap is {hvac_gap:.1f}%",
                    "impact": "high" if hvac_gap > 10 else "medium",
                }
            )

        return findings

    def _get_priority_recommendations(self):
        """Get priority recommendations"""
        if "recommendations" not in self.results:
            return []

        return [
            {
                "title": rec["title"],
                "savings": rec.get("estimated_savings", 0),
                "payback": rec.get("payback_period", "N/A"),
            }
            for rec in self.results["recommendations"].get("immediate_actions", [])
        ]

    def _generate_financial_highlights(self):
        """Generate financial analysis highlights"""
        if "financial_analysis" not in self.results:
            return {}

        financial = self.results["financial_analysis"]
        return {
            "total_investment": financial["total_investment_required"],
            "annual_savings": financial["annual_savings_potential"],
            "simple_payback": financial["simple_payback_period"],
            "roi": financial["roi"],
        }

    def _generate_energy_analysis_notes(self):
        """Generate notes for energy analysis"""
        energy_results = self.results["energy"]
        notes = []

        # Check seasonal impact
        seasonal_impact = energy_results["efficiency_metrics"]["seasonal_impact"]
        if abs(seasonal_impact) > 20:
            notes.append(f"High seasonal variation detected ({seasonal_impact:.1f}%)")

        # Check occupancy impact
        occupancy_impact = energy_results["efficiency_metrics"]["occupancy_impact"]
        if abs(occupancy_impact) > 15:
            notes.append(f"Significant occupancy impact ({occupancy_impact:.1f}%)")

        return notes

    def _calculate_humidity_ratio(self, building_data):
        """Calculate humidity ratio"""
        # Simplified calculation
        return building_data["current_humidity"] / 100

    def _calculate_dew_point(self, building_data):
        """Calculate dew point"""
        # Simplified calculation - Magnus formula
        T = building_data.get("temperature", 20)  # Default to 20°C if not provided
        RH = building_data["current_humidity"]

        a = 17.27
        b = 237.7

        alpha = ((a * T) / (b + T)) + np.log(RH / 100.0)
        return (b * alpha) / (a - alpha)

    def _calculate_vapor_pressure(self, building_data):
        """Calculate vapor pressure"""
        # Simplified calculation
        T = building_data.get("temperature", 20)  # Default to 20°C if not provided
        RH = building_data["current_humidity"]

        # Saturation vapor pressure
        es = 6.112 * np.exp((17.67 * T) / (T + 243.5))

        # Actual vapor pressure
        return (RH / 100.0) * es

    def calculate_dehumidification_needs(
        self, volume, current_humidity, target_humidity
    ):
        """Calculate dehumidification needs"""
        # Simplified calculation - returns pints per day
        humidity_difference = current_humidity - target_humidity
        return 0 if humidity_difference <= 0 else 0.0007 * volume * humidity_difference

    def _determine_unit_size(self, capacity_needed):
        """Determine appropriate dehumidifier unit size"""
        if capacity_needed <= 30:
            return "small"
        elif capacity_needed <= 50:
            return "medium"
        else:
            return "large"

    def _estimate_runtime(self, capacity_needed, humidity_gap):
        """Estimate dehumidifier runtime"""
        # Simplified calculation - returns hours per day
        base_runtime = 8
        if humidity_gap > 20:
            return base_runtime * 1.5
        elif humidity_gap > 10:
            return base_runtime * 1.2
        return base_runtime

    def _determine_humidity_priority(self, current_humidity, target_humidity):
        """
        Determine the priority level for humidity control based on the difference
        between current and target humidity levels.

        Returns:
        - 'high': difference > 20%
        - 'medium': difference between 10-20%
        - 'low': difference < 10%
        """
        humidity_difference = abs(current_humidity - target_humidity)

        if humidity_difference > 20:
            return "high"
        elif humidity_difference > 10:
            return "medium"
        else:
            return "low"

    def _calculate_humidity_ratio(self, building_data):
        """Calculate humidity ratio from relative humidity"""
        # Simplified calculation - assumes standard temperature and pressure
        current_humidity = building_data["current_humidity"]
        temperature = building_data.get(
            "temperature", 20
        )  # Default to 20°C if not provided

        # Simplified calculation of humidity ratio
        saturation_pressure = 6.112 * np.exp(
            (17.67 * temperature) / (temperature + 243.5)
        )
        partial_pressure = (current_humidity / 100.0) * saturation_pressure
        return 0.622 * (partial_pressure / (101.325 - partial_pressure))

    def _calculate_dew_point(self, building_data):
        """Calculate dew point temperature"""
        current_humidity = building_data["current_humidity"]
        temperature = building_data.get(
            "temperature", 20
        )  # Default to 20°C if not provided

        # Magnus formula
        alpha = np.log(current_humidity / 100.0) + (
            (17.67 * temperature) / (temperature + 243.5)
        )
        return (243.5 * alpha) / (17.67 - alpha)

    def _calculate_vapor_pressure(self, building_data):
        """Calculate vapor pressure"""
        current_humidity = building_data["current_humidity"]
        temperature = building_data.get(
            "temperature", 20
        )  # Default to 20°C if not provided

        # Calculate saturation vapor pressure
        saturation_pressure = 6.112 * np.exp(
            (17.67 * temperature) / (temperature + 243.5)
        )

        return (current_humidity / 100.0) * saturation_pressure

    def calculate_dehumidification_needs(
        self, volume, current_humidity, target_humidity
    ):
        """
        Calculate dehumidification capacity needed based on room conditions.
        Returns capacity needed in pints per day.
        """
        if current_humidity <= target_humidity:
            return 0

        humidity_difference = current_humidity - target_humidity
        return 0.0007 * volume * humidity_difference

    def _determine_unit_size(self, capacity_needed):
        """
        Determine appropriate dehumidifier unit size based on capacity needed.
        """
        if capacity_needed <= 30:
            return "small"
        elif capacity_needed <= 50:
            return "medium"
        else:
            return "large"

    def _estimate_runtime(self, capacity_needed, humidity_gap):
        """
        Estimate daily runtime hours based on capacity needed and humidity gap.
        """
        base_runtime = 8  # Base runtime hours

        if humidity_gap > 20:
            return base_runtime * 1.5
        elif humidity_gap > 10:
            return base_runtime * 1.2

        return base_runtime

    def _categorize_recommendations(self, recommendations_list, recommendations_dict):
        """
        Categorize recommendations into immediate, short-term, and long-term actions
        based on priority and estimated savings.
        """
        if not recommendations_list:
            return

        for rec in recommendations_list:
            priority = rec.get("priority", "medium").lower()

            if priority == "high":
                recommendations_dict["immediate_actions"].append(rec)
            elif priority == "medium":
                recommendations_dict["short_term"].append(rec)
            else:
                recommendations_dict["long_term"].append(rec)

    def _estimate_energy_savings(self):
        """Estimate potential energy savings from energy efficiency improvements"""
        try:
            current_consumption = self.results["energy"]["Ereal"]
            potential_improvement = 0.15  # Assume 15% potential improvement

            return (
                current_consumption * potential_improvement * 0.12
            )  # Assuming $0.12/kWh
        except Exception:
            return 0

    def _estimate_seasonal_savings(self):
        """Estimate potential savings from seasonal optimization"""
        try:
            seasonal_impact = abs(
                self.results["energy"]["efficiency_metrics"]["seasonal_impact"]
            )
            current_consumption = self.results["energy"]["Ereal"]

            return (
                (seasonal_impact / 100) * current_consumption * 0.12 * 0.5
            )  # 50% of potential improvement
        except Exception:
            return 0

    def _estimate_hvac_savings(self):
        """Estimate potential savings from HVAC improvements"""
        try:
            if "hvac" not in self.results:
                return 0

            efficiency_gap = self.results["hvac"]["system_efficiency"]["efficiency_gap"]
            current_consumption = self.results["hvac"]["energy_consumption"]

            return (efficiency_gap / 100) * current_consumption * 0.12
        except Exception:
            return 0

    def _estimate_lighting_savings(self):
        """Estimate potential savings from lighting improvements"""
        try:
            if "lighting" not in self.results:
                return 0

            current_consumption = self.results["lighting"]["total_consumption"]
            potential_improvement = 0.30  # Assume 30% potential improvement with LED

            return current_consumption * potential_improvement * 0.12
        except Exception:
            return 0

    def _estimate_implementation_cost(self, improvement_type):
        """
        Estimate implementation cost for different types of improvements.
        These are rough estimates and should be adjusted based on actual market data.
        """
        cost_estimates = {
            "energy_management": 5000,
            "seasonal_optimization": 2000,
            "hvac_upgrade": 10000,
            "lighting_upgrade": 5000,
            "humidity_control": 3000,
        }

        return cost_estimates.get(improvement_type, 1000)

    def _calculate_total_savings(self, recommendations):
        """Calculate total estimated savings across all recommendations"""
        total_savings = {
            "energy": sum(
                rec.get("estimated_savings", 0)
                for rec in recommendations.get("immediate_actions", [])
            ),
            "hvac": sum(
                rec.get("estimated_savings", 0)
                for rec in recommendations.get("short_term", [])
            ),
            "lighting": sum(
                rec.get("estimated_savings", 0)
                for rec in recommendations.get("long_term", [])
            ),
        }

        # Add product-specific savings if available
        if "product_recommendations" in recommendations:
            for product_type, products in recommendations[
                "product_recommendations"
            ].items():
                if isinstance(products, pd.DataFrame):
                    # Calculate potential savings based on efficiency improvements
                    total_savings[product_type] = self._calculate_product_savings(
                        products
                    )

        return total_savings

    def _calculate_product_savings(self, products_df):
        """Calculate potential savings from recommended products"""
        try:
            if "Efficiency_Value" not in products_df.columns:
                return 0

            # Calculate average efficiency improvement
            avg_efficiency = products_df["Efficiency_Value"].mean()
            baseline_efficiency = 1.0  # Baseline efficiency

            # Estimate annual energy savings
            improvement_factor = (
                avg_efficiency - baseline_efficiency
            ) / baseline_efficiency
            estimated_annual_usage = 1000  # Assumed annual energy usage in kWh

            return (
                estimated_annual_usage * improvement_factor * 0.12
            )  # Assuming $0.12/kWh
        except Exception:
            return 0

    def _generate_humidity_recommendations(self):
        """Generate humidity-specific recommendations"""
        recommendations = []

        try:
            if "humidity" not in self.results:
                return recommendations

            humidity_results = self.results["humidity"]
            current_status = humidity_results.get("current_status", {})
            requirements = humidity_results.get("requirements", {})

            if requirements.get("needs_dehumidification"):
                capacity_needed = humidity_results.get("product_needs", {}).get(
                    "capacity_needed", 0
                )

                recommendations.append(
                    {
                        "category": "humidity",
                        "priority": requirements.get("control_priority", "medium"),
                        "title": "Install Dehumidification System",
                        "description": f"Install dehumidification system with {capacity_needed:.1f} pints/day capacity",
                        "estimated_savings": self._estimate_humidity_savings(
                            humidity_results
                        ),
                        "implementation_cost": self._estimate_implementation_cost(
                            "humidity_control"
                        ),
                        "payback_period": None,
                    }
                )

        except Exception as e:
            self.logger.error(f"Error generating humidity recommendations: {e}")

        return recommendations

    def _estimate_humidity_savings(self, humidity_results):
        """Estimate potential savings from humidity control improvements"""
        try:
            # Estimate energy savings from improved humidity control
            # This is a simplified calculation
            capacity_needed = humidity_results.get("product_needs", {}).get(
                "capacity_needed", 0
            )
            runtime_hours = humidity_results.get("product_needs", {}).get(
                "estimated_runtime", 8
            )

            # Assume 1000W average power consumption per 50 pints/day capacity
            power_consumption = (capacity_needed / 50) * 1000  # Watts
            annual_energy = power_consumption * runtime_hours * 365 / 1000  # kWh/year

            return annual_energy * 0.12  # Assuming $0.12/kWh
        except Exception:
            return 0
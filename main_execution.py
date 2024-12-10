# main.py

import logging
import json
from energy_audit_tool import EnergyAuditTool

def load_building_data(filename):
    """Load building data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize the audit tool
        audit_tool = EnergyAuditTool(
            power_consumed=50,  # Example values
            duration_hours=10,
            seasonal_factor=1.2,
            occupancy_factor=0.9,
            power_factor=0.92
        )
        
        # Load product database
        if not audit_tool.initialize_databases('energy_star_products.csv'):
            logger.error("Failed to initialize product database")
            return
        
        # Example building data
        building_data = {
            'building_id': 'BLDG001',
            'volume': 20000,  # cubic feet
            'current_humidity': 65,  # %
            'target_humidity': 45,  # %
            'hvac_data': {
                'output_capacity': 60000,  # BTU
                'input_power': 5000,  # W
                'target_efficiency': 95,
                'surfaces': [
                    {
                        'name': 'North Wall',
                        'U': 0.5,
                        'A': 200,
                        'delta_T': 20
                    }
                ]
            },
            'lighting_data': {
                'fixtures': [
                    {
                        'name': 'Office Lights',
                        'watts': 1000,  # W
                        'hours': 3000,  # hours/year
                        'lumens': 80000,
                        'electricity_rate': 0.12  # $/kWh
                    },
                    {
                        'name': 'Task Lighting',
                        'watts': 500,
                        'hours': 2000,
                        'lumens': 40000,
                        'electricity_rate': 0.12
                    }
                ]
            }
        }
        
        # Run analysis
        results = audit_tool.run_analysis(
            building_data,
            export_format='pdf',
            include_visualizations=True
        )
        
        logger.info("Analysis completed successfully")
        logger.info(f"Report exported to: {results['report']['exported_file']}")
        
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
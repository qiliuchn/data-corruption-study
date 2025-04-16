# Generating demand for single intersection
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np
import random
import os

# Set simulation parameters
NUM_ROUTES = 12
# route 0-2: West L-T-R traffic
# route 3-5: East L-T-R traffic
# route 6-8: south L-T-R traffic
# route 9-11: north L-T-R traffic
SIMULATION_STEPS = 3600  # Total number of simulation steps (e.g., 1 hour)

# Binomial distribution parameters
# Note: Adjust these based on desired traffic flow characteristics
b = [2] * 12  # Maximum number of arriving vehicles per second (2 to 5)
p_max = [0.01, 0.03, 0.02] * 4  # Probability for binomial distribution
p_max = np.asarray(p_max)
p_max = p_max * 3.22

# Output path for the generated route file
output_path = "./generated_demand.rou.xml"

# Generate time-varying probabilities for east-west (routes 0-5) and north-south (routes 6-11) traffic
def get_time_varying_probabilities(step, sim_steps, p_max):
    """
    Generate time-varying probabilities for traffic demand.
    East-West (routes 0-5): sine curve.
    North-South (routes 6-11): cosine curve.
    """
    time_fraction = step / sim_steps  # Normalize time to [0, 1]
    multiplier = np.pi / 2 * time_fraction  # Scale time to [0, π/2]

    probabilities = np.zeros_like(p_max)

    # East-West traffic (routes 0-5) follow sine curve, first small then large
    probabilities[0:6] = p_max[0:6] * np.sin(multiplier)

    # North-South traffic (routes 6-11) follow cosine curve, first large then small
    probabilities[6:12] = p_max[6:12] * np.cos(multiplier)

    return probabilities


# Function to generate stochastic demand based on the binomial distribution
def generate_vehicle_flows():
    flows = []
    
    for step in range(SIMULATION_STEPS):
        # Get time-varying probabilities for the current step
        p_t = get_time_varying_probabilities(step, SIMULATION_STEPS, p_max)
        
        for route_id in range(NUM_ROUTES):
            route_id_str = f"r_{route_id}"
        
            # Generate vehicle arrivals for each second
            # Number of vehicles arriving in this second
            arrivals = np.random.binomial(b[route_id], p_t[route_id])
            
            # Create vehicles with random intervals within the current second
            for i in range(arrivals):
                departure_time = step + random.uniform(0, 1)  # Departure time within the current second
                vehicle_id = f"veh_{route_id}_{step}_{i}"
                
                flows.append({
                    "vehicle_id": vehicle_id,
                    "departure_time": departure_time,
                    "route_id": route_id_str
                })
                
    # Sort flows by departure time before returning
    flows.sort(key=lambda x: x["departure_time"])
    return flows

# Function to write flows to a SUMO-compatible XML route file with pretty formatting
def write_to_route_file(flows):
    root = ET.Element("routes")
    
    # Define routes for each route ID
    #for route_id in range(NUM_ROUTES):
    #    ET.SubElement(root, "route", id=f"r_{route_id}", edges=f"edge_{route_id}_start edge_{route_id}_end")
    
    # Create vehicles
    for flow in flows:
        ET.SubElement(
            root, "vehicle",
            id=flow["vehicle_id"],
            depart=str(flow["departure_time"]),
            route=flow["route_id"]
        )
    
    # Convert ElementTree to a string
    rough_string = ET.tostring(root, 'utf-8')
    # Use minidom to pretty print the XML
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    
    # Save pretty-printed XML to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)
    
    print(f"Generated route file saved at: {output_path}")

if __name__ == "__main__":
    # Generate demand flows using binomial distribution
    flows = generate_vehicle_flows()
    # Write to SUMO route file with pretty formatting
    write_to_route_file(flows)
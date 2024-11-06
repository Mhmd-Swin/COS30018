import pathlib
import folium
import datetime
import webbrowser
import pandas as pd
from traffic_predict import scats_network, calculate_routes_with_dijkstra


def create_scats_location_map(display_output=False):
    """
    Creates a dictionary mapping SCATS sites to their respective locations based on the file names
    in the specified directory. Optionally displays the location mapping in the console.

    Args:
        display_output (bool): If True, prints the SCATS site mappings to the console. Defaults to False.
    
    Returns:
        dict: A dictionary with SCATS codes as keys and lists of location names as values.
    """
    location_map = {}

    # Iterate over each traffic file in the specified directory
    for traffic_file in pathlib.Path("C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/TrafficFlowPrediction-master/data/TRAFFIC_FLOW").iterdir():
        scats_code = traffic_file.name[:4]  # Extract SCATS code from the file name
        location_name = traffic_file.name[5:-4]  # Extract location name from the file name

        # If the SCATS code isn't already in location_map, add it with the location name
        if scats_code not in location_map:
            location_map[scats_code] = [location_name]
        else:
            location_map[scats_code].append(location_name)  # Append the location if SCATS code exists

    # Display the location map if display_output is set to True
    if display_output:
        print("List of locations for each SCATS site:")
        for scats_code in location_map:
            print(str(scats_code))
            for location in location_map[scats_code]:
                print("\t" + location)
    else:
        return location_map


def check_scats_network_consistency():
    """
    Validates the integrity and consistency of SCATS adjacency data by performing:
    - A check for missing SCATS codes in either scats_network or location_map.
    - A symmetry check for SCATS adjacencies.
    - A consistency check for location names across scats_network and location_map.
    """
    location_map = create_scats_location_map()  # Generate the SCATS location map
    
    # Create sets of SCATS codes for comparison
    scats_network_codes = set(scats_network.keys())
    location_map_codes = set(location_map.keys())
    
    # Check for SCATS codes present in location_map but missing from scats_network
    missing_in_scats_network = location_map_codes - scats_network_codes
    for scats_code in missing_in_scats_network:
        print(f"SCATS {scats_code} exists in location_map but is missing from scats_network")

    # Check for SCATS codes present in scats_network but missing from location_map
    missing_in_location_map = scats_network_codes - location_map_codes
    for scats_code in missing_in_location_map:
        print(f"SCATS {scats_code} exists in scats_network but is missing from location_map")
    
    # Validate adjacency symmetry and location name consistency within scats_network
    for scats_code, adjacents in scats_network.items():
        for adj_code, adj_location in adjacents.items():
            # Check if adjacency is symmetric (A -> B should imply B -> A)
            if adj_code in scats_network:
                if scats_code not in scats_network[adj_code]:
                    print(f"Adjacency missing: SCATS {scats_code} is adjacent to {adj_code}, but not vice versa.")
            
            # Validate that location names in scats_network match the names in location_map
            if str(scats_code) in location_map:
                if adj_location not in location_map[str(scats_code)]:
                    print(f"Location mismatch for SCATS {scats_code}: [{adj_location}] is in scats_network but not in location_map")

    print("SCATS network validation complete.")


def fetch_mean_coordinates(mean_coords_df, scats_code):
    """Retrieve the mean latitude and longitude for a given SCATS site."""
    location_data = mean_coords_df[mean_coords_df['SCATS Number'] == scats_code]
    if not location_data.empty:
        return [location_data.iloc[0]['NB_LATITUDE'], location_data.iloc[0]['NB_LONGITUDE']]
    return None


def generate_route_map(route_data):
    # Load SCATS location and mean coordinates data
    scats_coords_df = pd.read_csv('C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/Scats Location Points.csv')
    mean_coords_df = pd.read_csv('C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/Scats Mean Points.csv')

    # Define map center using average latitude and longitude values
    avg_lat = scats_coords_df['NB_LATITUDE'].mean()
    avg_lon = scats_coords_df['NB_LONGITUDE'].mean()
    map_center = [avg_lat, avg_lon]
    scats_map = folium.Map(location=map_center, zoom_start=12.5)

    # Color palette for routes
    colors = ['blue', 'purple', 'orange', 'green', 'red']
    
    # Mark each SCATS site on the map using the mean coordinates, with SCATS number as label
    for _, row in mean_coords_df.iterrows():
        folium.CircleMarker(
            location=[row['NB_LATITUDE'], row['NB_LONGITUDE']],
            radius=8,
            color='#7cc4ff',
            fill=True,
            fill_color='#7cc4ff',
            fill_opacity=1
        ).add_to(scats_map)
        
        # Display SCATS number as a label above each marker
        folium.map.Marker(
            [row['NB_LATITUDE'], row['NB_LONGITUDE']],
            icon=folium.DivIcon(html=f"""
                <div style="font-size: 13.5px; color: black; font-weight: bold;">{int(row['SCATS Number'])}</div>
                """)
        ).add_to(scats_map)

    # Draw each route with a unique color and include travel time in the legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 20px; right: 20px; width: 280px; height: auto; 
                background-color: rgba(255, 255, 255, 0.9); z-index:9990; font-size:14px;
                padding: 15px; border-radius: 8px; box-shadow: 0px 0px 10px rgba(0,0,0,0.3);">
    <b>Route Legend:</b><br>
    """

    for idx, (route, duration) in enumerate(route_data[:5]):  # Limit to 5 routes for clarity
        route_color = colors[idx % len(colors)]  # Cycle through colors if there are more than 5 routes
        route_coords = [fetch_mean_coordinates(mean_coords_df, scats) for scats in route if fetch_mean_coordinates(mean_coords_df, scats) is not None]
        
        if route_coords:
            folium.PolyLine(
                route_coords,
                color=route_color,
                weight=6 if idx == 0 else 4,  # Thicker line for the main route
                opacity=0.8,
                tooltip=f"Route {idx + 1} Duration: {duration:.2f} min"
            ).add_to(scats_map)

            # Add legend entry with color and travel time
            legend_html += f"<i style='color:{route_color}; font-weight: bold;'>&#9679;</i> Route {idx + 1} - Duration: {duration:.2f} min<br>"

            # Mark the start and end points of the primary route only
            if idx == 0:
                start_coords = route_coords[0]
                end_coords = route_coords[-1]
                
                # Clear and styled start point marker
                folium.Marker(
                    location=start_coords,
                    icon=folium.Icon(color='darkgreen', icon='play-circle', prefix='fa'),
                    tooltip='Starting Point',
                    popup=folium.Popup('<b>Starting Point</b>', max_width=300)
                ).add_to(scats_map)

                # Clear and styled end point marker
                folium.Marker(
                    location=end_coords,
                    icon=folium.Icon(color='darkred', icon='flag-checkered', prefix='fa'),
                    tooltip='Destination',
                    popup=folium.Popup('<b>Destination</b>', max_width=300)
                ).add_to(scats_map)

    legend_html += "</div>"
    
    # Add the legend to the map
    scats_map.get_root().html.add_child(folium.Element(legend_html))

    # Save the generated map to an HTML file and open it in the browser
    scats_map.save('scats_route_map.html')
    webbrowser.open_new_tab('scats_route_map.html')


if __name__ == "__main__":
    # Generate and display the SCATS location map and validate SCATS network consistency
    create_scats_location_map(True)
    check_scats_network_consistency()
    
    # Sample route calculation
    sample_routes = calculate_routes_with_dijkstra('lstm', 2000, 4034, datetime.datetime(2024, 10, 18, 2, 52))
    generate_route_map(sample_routes)

import copy
import os
import datetime
import heapq
import math
import numpy as np
import pandas as pd
from keras.api.models import load_model
from keras.api.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic


# SCATS network mapping
scats_network = {
     970: {3685: 'WARRIGAL_RD N of HIGH STREET_RD', 2846: 'HIGH STREET_RD W of WARRIGAL_RD'},
    2000: {3685: 'WARRIGAL_RD S of BURWOOD_HWY', 3682: 'WARRIGAL_RD N of TOORAK_RD', 4043: 'TOORAK_RD W of WARRIGAL_RD'},
    2200: {3126: 'UNION_RD S of MAROONDAH_HWY', 4063: 'MAROONDAH_HWY W of UNION_RD'},
    2820: {3662: 'PRINCESS_ST S of CHANDLER_HWY', 4321: 'EARL_ST SE of PRINCESS_ST'},
    2825: {4030: 'BURKE_RD S of EASTERN_FWY'},
    2827: {4051: 'BULLEEN_RD S of EASTERN_FWY_W_BD_RAMPS'},
    2846: {970: 'HIGH_ST E OF S.E.ARTERIAL'},
    3001: {3002: 'BARKERS_RD E of HIGH_ST', 3662: 'HIGH_ST NE of BARKERS_RD', 4821: 'BARKERS_RD W of CHURCH_ST', 4262: 'CHURCH_ST SW of BARKERS_RD'},
    3002: {3001: 'BARKERS_RD W of DENMARK_ST', 3662: 'DENMARK_ST N of BARKERS_RD', 4263: 'POWER_ST S of BARKERS_RD', 4035: 'BARKERS_RD E of DENMARK_ST'},
    3120: {4035: 'BURKE_RD N of CANTERBURY_RD', 3122: 'CANTERBURY_RD E of BURKE_RD', 4040: 'BURKE_RD S of CANTERBURY_RD'},
    3122: {3120: 'CANTERBURY_RD W of STANHOPE_GV', 3127: 'CANTERBURY_RD E of STANHOPE_GV', 3804: 'STANHOPE_GV S of CANTERBURY_RD'},
    3126: {3127: 'CANTERBURY_RD W of WARRIGAL_RD', 3682: 'WARRIGAL_RD S of CANTERBURY_RD'},
    3127: {3122: 'CANTERBURY_RD W of BALWYN_RD', 4063: 'BALWYN_RD N of CANTERBURY_RD', 3126: 'CANTERBURY_RD E of BALWYN_RD'},
    3180: {4051: 'DONCASTER_RD W of BALWYN_RD', 4057: 'BALWYN_RD S of DONCASTER_RD'},
    3662: {3001: 'HIGH_ST SW of DENMARK_ST', 4324: 'HIGH_ST NE of PRINCESS_ST', 4335: 'HIGH_ST NE of PRINCESS_ST', 2820: 'PRINCESS_ST N of HIGH_ST'},
    3682: {3126: 'WARRIGAL_RD N OF RIVERSDALE_RD', 3804: 'RIVERSDALE_RD W OF WARRIGAL_RD', 2000: 'WARRIGAL_RD S OF RIVERSDALE_RD'},
    3685: {2000: 'WARRIGAL_RD N of HIGHBURY_RD', 970: 'WARRIGAL_RD S of HIGHBURY_RD'},
    3804: {4040: 'RIVERSDALE_RD W of TRAFALGAR_RD', 3122: 'TRAFALGAR_RD N of RIVERSDALE_RD', 3812: 'TRAFALGAR_RD S of RIVERSDALE_RD', 3682: 'RIVERSDALE_RD E of TRAFALGAR_RD'},
    3812: {3804: 'TRAFALGAR_RD NE of CAMBERWELL_RD', 4040: 'CAMBERWELL_RD NW of TRAFALGAR_RD'},
    4030: {4321: 'HIGH_ST SW of BURKE_RD', 4032: 'BURKE_RD S of DONCASTER_RD'},
    4032: {4321: 'HARP_RD W of BURKE_RD', 4030: 'BURKE_RD N of HARP_RD', 4057: 'BELMORE_RD E of BURKE_RD', 4034: 'BURKE_RD S of HARP_RD'},
    4034: {4324: 'COTHAM_RD W OF BURKE_RD', 4032: 'BURKE_RD N OF WHITEHORSE_RD', 4063: 'WHITEHORSE_RD E OF BURKE_RD', 4035: 'BURKE_RD S OF WHITEHORSE_RD'},
    4035: {3002: 'BARKERS_RD W of BURKE_RD', 4034: 'BURKE_RD N of MONT ALBERT_RD', 3120: 'BURKE_RD S of BARKERS_RD'},
    4040: {4272: 'RIVERSDALE_RD W of BURKE_RD', 3120: 'BURKE_RD N of RIVERSDALE_RD', 3804: 'RIVERSDALE_RD E of BURKE_RD', 3812: 'CAMBERWELL_RD SE of BURKE_RD', 4043: 'BURKE_RD S of RIVERSDALE_RD', 4266: 'CAMBERWELL_RD NW of BURKE_RD'},
    4043: {4273: 'TOORAK_RD W OF BURKE_RD', 4040: 'BURKE_RD N of TOORAK_RD', 2000: 'TOORAK_RD E of BURKE_RD'},
    4051: {4030: 'DONCASTER_RD SW of SEVERN_ST', 3180: 'DONCASTER_RD E of BULLEEN_RD'},
    4057: {4032: 'BELMORE_RD W OF BALWYN_RD', 3180: 'BALWYN_RD N OF BELMORE_RD', 4063: 'BALWYN_RD S OF BELMORE_RD'},
    4063: {4034: 'WHITEHORSE_RD W OF BALWYN_RD', 4057: 'BALWYN_RD N OF WHITEHORSE_RD', 2200: 'WHITEHORSE_RD W OF BALWYN_RD', 3127: 'BALWYN_RD S OF WHITEHORSE_RD'},
    4263: {4262: 'BURWOOD_RD W of POWER_ST', 3002: 'POWER_ST N of BURWOOD_RD', 4264: 'BURWOOD_RD E of POWER_ST'},
    4264: {4263: 'BURWOOD_RD W of GLENFERRIE_RD', 4324: 'GLENFERRIE_RD N of BURWOOD_RD', 4266: 'BURWOOD_RD E of GLENFERRIE_RD', 4270: 'GLENFERRIE_RD S of BURWOOD_RD'},
    4266: {4264: 'BURWOOD_RD W of AUBURN_RD', 4040: 'BURWOOD_RD E of AUBURN_RD'},
    4270: {4812: 'RIVERSDALE_RD W of GLENFERRIE_RD', 4264: 'GLENFERRIE_RD N of RIVERSDALE_RD', 4272: 'RIVERSDALE_RD E of GLENFERRIE_RD'},
    4272: {4270: 'RIVERSDALE_RD W of TOORONGA_RD', 4040: 'RIVERSDALE_RD E of TOORONGA_RD', 4273: 'TOORONGA_RD S of RIVERSDALE_RD'},
    4273: {4272: 'TOORONGA_RD N of TOORAK_RD', 4043: 'TOORAK_RD E of TOORONGA_RD'},
    4321: {4335: 'HIGH_ST SW OF HARP_ST', 4032: 'HARP_RD E OF HIGH_ST', 2820: 'VALERIE_ST W OF HIGH_ST', 4030: 'HIGH_ST NE OF HARP_ST'},
    4324: {4034: 'COTHAM_RD E OF GLENFERRIE_RD', 3662: 'COTHAM_RD W OF GLENFERRIE_RD', 4264: 'GLENFERRIE_RD S OF COTHAM_RD'},
    4335: {4321: 'HIGH_ST NE of CHARLES_ST'},
    4812: {4270: 'SWAN_ST NE of MADDEN_GV'},
    4821: {3001: 'VICTORIA_ST E OF BURNLEY_ST'},
}

def calculate_geodesic_distance(site_a, site_b, coord_data):
    """Calculates the geodesic distance in kilometers between two SCATS sites."""
    lat_a, lon_a = coord_data.loc[site_a, ['NB_LATITUDE', 'NB_LONGITUDE']]
    lat_b, lon_b = coord_data.loc[site_b, ['NB_LATITUDE', 'NB_LONGITUDE']]
    return geodesic((lat_a, lon_a), (lat_b, lon_b)).kilometers

def dijkstra_search(graph, start_node, end_node, coordinates, exclusion_set=set()):
    """Performs Dijkstra's search to find the shortest path between nodes."""
    priority_queue = []
    heapq.heappush(priority_queue, (0, start_node))
    came_from = {start_node: None}
    travel_cost = {start_node: 0}

    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)

        if current_node == end_node:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = came_from[current_node]
            return path[::-1]

        if current_node not in graph:
            continue

        for neighbor in graph[current_node]:
            if neighbor in exclusion_set:
                continue

            if neighbor not in travel_cost:
                travel_cost[neighbor] = float('inf')

            new_cost = travel_cost[current_node] + calculate_geodesic_distance(current_node, neighbor, coordinates)
            if new_cost < travel_cost[neighbor]:
                came_from[neighbor] = current_node
                travel_cost[neighbor] = new_cost
                heapq.heappush(priority_queue, (travel_cost[neighbor], neighbor))

    return None

def calculate_routes_with_dijkstra(origin, destination, coordinates, max_routes=5):
    """
    Generate up to five routes using Dijkstra's algorithm, progressively excluding nodes to find alternative routes.
    
    Parameters:
        origin (int): The SCATS code for the origin.
        destination (int): The SCATS code for the destination.
        coordinates (DataFrame): DataFrame containing SCATS site coordinates.
        max_routes (int): Maximum number of routes to find.
    
    Returns:
        list: A list of routes, each represented as a list of SCATS site codes.
    """
    routes_list = []
    exclusion_set = set()

    while len(routes_list) < max_routes:
        # Find the shortest path using Dijkstra's algorithm with the current exclusion set
        route = dijkstra_search(scats_network, origin, destination, coordinates, exclusion_set)
        if route is None:
            break  # Stop if no more routes can be found

        routes_list.append(route)

        # Exclude the second-last node in the route to encourage finding a different path
        if len(route) > 1:
            exclusion_set.add(route[-2])

    print("Route calculation complete!")
    return routes_list


if __name__ == '__main__':
    # Load coordinates data
    coord_data = pd.read_csv('C:/Users/hasan/Downloads/mhmd/TrafficFlowPrediction-master/Scats Mean Points.csv').set_index('SCATS Number')

    # Example usage to find routes
    origin = 3001
    destination = 3120
    routes = calculate_routes_with_dijkstra(origin, destination, coord_data)
    print("Found Routes:")
    for idx, route in enumerate(routes, 1):
        print(f"Route {idx}: {route}")
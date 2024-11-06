import tkinter as tk
from tkinter import messagebox, ttk
from tkcalendar import DateEntry
from datetime import datetime
import traffic_predict
from file import generate_route_map
from file import create_scats_location_map
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='absl')

# Function to initiate traffic prediction
def generate_traffic_prediction():
    start_location = origin_dropdown.get()[:4]
    end_location = destination_dropdown.get()[:4]
    selected_model = model_dropdown.get().lower()
    selected_date = date_selector.get()
    selected_time = time_entry.get()

    try:
        hours_minutes = [f"{int(part):02}" for part in selected_time.split(":")]
        if len(hours_minutes) != 2:
            raise ValueError("Invalid time format")
    except ValueError as e:
        messagebox.showwarning("Input Error", str(e))
        return
    selected_time = ":".join(hours_minutes)

    full_datetime = f"{selected_date.replace('/', '-')} {selected_time}"
    try:
        full_datetime = datetime.strptime(full_datetime, '%d-%m-%Y %H:%M')
    except ValueError:
        messagebox.showwarning("Input Error", "Date format should be dd-mm-yyyy.")
        return

    if not start_location or not end_location or not selected_model:
        messagebox.showwarning("Input Error", "All fields are required.")
        return

    try:
        calculated_routes = traffic_predict.calculate_routes_with_dijkstra(selected_model, int(start_location), int(end_location), full_datetime)
        if not calculated_routes:
            result_display.config(text="No routes available.")
            return

        result_text = "Predicted Routes:\n"
        for idx, (route, duration) in enumerate(calculated_routes[:5]):
            result_text += f"Route {idx + 1}: {route}, Duration: {duration:.2f} min\n"

        result_display.config(text=result_text)
        generate_route_map(calculated_routes[:5])
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

# Initialize application window
app = tk.Tk()
app.title("Traffic Prediction")
app.geometry("700x500")
app.resizable(False, False)
app.configure(bg="#3e3eff")

# Create style for widgets
style = ttk.Style()
style.theme_use('clam')
style.configure('TLabel', background='#3e3e42', font=('Helvetica', 11), foreground='white')
style.configure('TButton', font=('Helvetica', 12), padding=8, background='#528945', foreground='white')
style.map('TButton', background=[('active', '#457f99')])
style.configure('TEntry', padding=5, font=('Helvetica', 11))
style.configure('TFrame', background='#3e3e42')
style.configure('TCombobox', font=('Helvetica', 10))

# Sidebar for input
sidebar = ttk.Frame(app, width=250, padding=20)
sidebar.pack(side='left', fill='y')

header_label = ttk.Label(sidebar, text="Traffic Prediction", font=('Helvetica', 16, 'bold'), foreground='#ffbbbb')
header_label.pack(anchor='w', pady=(0, 20))

# Input fields on sidebar
ttk.Label(sidebar, text="SELECT ORIGIN").pack(anchor='w', pady=5)
origin_dropdown = ttk.Combobox(sidebar, values=[], state='readonly', width=25)
origin_dropdown.pack(anchor='w', pady=5)

ttk.Label(sidebar, text="SELECT DESTINATION").pack(anchor='w', pady=5)
destination_dropdown = ttk.Combobox(sidebar, values=[], state='readonly', width=25)
destination_dropdown.pack(anchor='w', pady=5)

ttk.Label(sidebar, text="CHOOSE MODEL").pack(anchor='w', pady=5)
model_dropdown = ttk.Combobox(sidebar, values=["SRNN", "SAEs-ext", "LSTM"], state='readonly', width=25)
model_dropdown.pack(anchor='w', pady=5)
model_dropdown.set("SRNN")

# Date and time selection
ttk.Label(sidebar, text="ENTER DATE (DD-MM-YYYY):").pack(anchor='w', pady=5)
date_selector = DateEntry(sidebar, width=23, background='darkblue', foreground='white', borderwidth=2, date_pattern='dd-mm-yyyy')
date_selector.set_date(datetime.strptime("01-10-2006", "%d-%m-%Y"))
date_selector.pack(anchor='w', pady=5)

ttk.Label(sidebar, text="ENTER TIME (HH:MM):").pack(anchor='w', pady=5)
time_entry = ttk.Entry(sidebar, width=10, justify='center')
time_entry.pack(anchor='w', pady=5)
time_entry.insert(0, "14:30")

# Predict button
predict_button = ttk.Button(sidebar, text="Predict Traffic", command=generate_traffic_prediction)
predict_button.pack(anchor='w', pady=(10, 0))

# Main content area for results
main_content = ttk.Frame(app, padding=20)
main_content.pack(side='right', fill='both', expand=True)

result_display = ttk.Label(main_content, text="Results will be displayed here", font=('Helvetica', 12), foreground='#eeeeee', background='#3e3e42', wraplength=300)
result_display.pack(anchor='center', pady=20)

# Populate dropdowns with SCATS data
destination_options = sorted(set([key for value in traffic_predict.scats_network.values() for key in value]))
scats_locations = create_scats_location_map()
start_options = []
end_options = []

for scats_code in traffic_predict.scats_network.keys():
    intersections = []
    roads = []
    for location in scats_locations[f"{scats_code:04}"]:
        road = location.split("_")[0] + " " + "".join(location.split("_")[1]).split(" ")[0]
        if road == "S.E.ARTERIAL N OF HIGH ST":
            roads.append("S.E.ARTERIAL")
        elif road == "OFFRAMP EASTERN":
            roads.append("OFFRAMP EASTERN FWY")
        elif road not in roads:
            roads.append(road)
    
    start_options.append(f"{scats_code} ({' x '.join(roads)})")
    if scats_code in destination_options:
        end_options.append(f"{scats_code} ({' x '.join(roads)})")

origin_dropdown['values'] = start_options
destination_dropdown['values'] = end_options

app.mainloop()

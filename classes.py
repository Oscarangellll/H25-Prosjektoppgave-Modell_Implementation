import pandas as pd


class Vessel:
    def __init__(self, name, vessel_type):
        self.name = name
        self.vessel_type = vessel_type
        
class VesselType:
    def __init__(self, name, multiday, n_teams, max_wind, max_wave, shift_length, day_rate, mob_rate, speed, cost_per_km, periodic_return, usage_cost_per_day):
        self.name = name
        self.multiday = multiday
        self.n_teams = n_teams
        self.max_wind = max_wind
        self.max_wave = max_wave
        self.shift_length = shift_length
        self.day_rate = day_rate
        self.mob_rate = mob_rate
        self.speed = speed
        self.cost_per_km = cost_per_km
        self.periodic_return = periodic_return
        self.usage_cost_per_day = usage_cost_per_day
        
    def calculate_ST(self, days):
        return self.day_rate * days + self.mob_rate   
    
    def calculate_LT(self, days, n_periods):
        return self.day_rate * days * n_periods + self.mob_rate

class Location:
    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates

class WindFarm(Location):
    def __init__(self, name, coordinates, n_turbines, weather_data_file, turbine_model):
        super().__init__(name, coordinates)
        self.n_turbines = n_turbines
        self.weather_data_file = weather_data_file    

        self._set_power_curve(turbine_model)
    
    def _set_power_curve(self, turbine_model):
        power_curve_df = pd.read_csv(f"Power Curves/{turbine_model}.csv")
        speed = power_curve_df["speed"].values
        power = power_curve_df["power"].values

        self.power_curve = {
            "speed": speed,
            "power": power
        }
        

    
class Base(Location):
    def __init__(self, name, coordinates):
        super().__init__(name, coordinates)
        
class MaintenanceCategory:
    def __init__(self, name, failure_rate, duration, vessel_types):
        self.name = name
        self.failure_rate = failure_rate  # failures per year
        self.duration = duration # in hours
        self.vessel_types = vessel_types

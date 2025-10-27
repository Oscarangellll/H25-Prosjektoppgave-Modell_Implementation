
class VesselType:
    def __init__(self, name, n_teams, max_wind, max_wave, shift_length, day_rate, mob_rate):
        self.name = name
        self.n_teams = n_teams
        self.max_wind = max_wind
        self.max_wave = max_wave
        self.shift_length = shift_length
        self.day_rate = day_rate
        self.mob_rate = mob_rate
        
    def calculate_ST(self, days):
        return self.day_rate * days + self.mob_rate   
    
    def calculate_LT(self, days, n_periods):
        return self.day_rate * days * n_periods + self.mob_rate

class WindFarm:
    def __init__(self, name, n_turbines, location, distance_to_base, weather_data_file):
        self.name = name
        self.n_turbines = n_turbines
        self.location = location
        self.distance_to_base = distance_to_base
        self.weather_data_file = weather_data_file

class MaintenanceCategory:
    def __init__(self, name, failure_rate, duration, vessel_types):
        self.name = name
        self.failure_rate = failure_rate  # failures per year
        self.duration = duration # in hours
        self.vessel_types = vessel_types




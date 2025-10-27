# import weather_windows
# import failure_generation

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
    def __init__(self, name, n_turbines, location, distance_to_base):
        self.name = name
        self.n_turbines = n_turbines
        self.location = location
        self.distance_to_base = distance_to_base

class MaintenanceCategory:
    def __init__(self, name, failure_rate, duration, vessel_types):
        self.name = name
        self.failure_rate = failure_rate  # failures per year
        self.duration = duration # in hours
        self.vessel_types = vessel_types

vessels = [
    VesselType("CTV", n_teams=2, max_wind=15, max_wave=1.5, shift_length=12, day_rate=3000, mob_rate=50_000),
]

maintenance_categories = [
    MaintenanceCategory("Annual Service", failure_rate=5.0, duration=2, vessel_types=["CTV", "SOV"])
]

days = 30
T = ["Jan"]

C_ST = {(v, t): v.calculate_ST(days) for v in vessels for t in T}
C_LT = {(v): v.calculate_LT(days, len(T)) for v in vessels}


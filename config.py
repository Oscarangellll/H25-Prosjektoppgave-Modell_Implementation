#config file for parameters

class Config:
    
    ######################
    # Sets
    ######################
    VESSELS = ["Crew Transfer Vessel", "Service Operation Vessel"]
    WINDFARMS = ["Wind Farm A"]
    
    ######################
    # Parameters
    ######################
    
    HOURS = 24
    DAYS = 30
    MONTHS = 12
    SYNTHETIC_YEARS = 100  # number of weather scenarios/samples
    
    # vessel operability limits
    WEATHER_LIMITS = {
        "Crew Transfer Vessel": {"max_wind": 25.0, "max_wave": 1.5},
        "Service Operation Vessel": {"max_wind": 30.0, "max_wave": 2.0},
    }
    
    #min window hours for operability
    MIN_WINDOW_HOURS = 5
    MAX_WINDOW_HOURS = 12
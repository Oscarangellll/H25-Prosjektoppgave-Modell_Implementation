# H25-Prosjektoppgave-Modell_Implementation

Day_rate = {
    0: 2940,  # Daily charter cost of Crew Transfer Vessel
    1: 11765,  # Daily charter cost of Service Operation Vessel
}

Mobilization_rate = {
    0: 58825,  # Mobilization cost for Crew Transfer Vessel
    1: 235295,  # Mobilization cost for Service Operation Vessel
}

months = 12  # Number of months in a year
days = 30  # Number of days in a month

C_ST = {
    0: Day_rate[0] * days + Mobilization_rate[0],  # Cost per month for short-term chartering of Crew Transfer Vessel
    1: Day_rate[1] * days + Mobilization_rate[1],  # Cost per month for short-term chartering of Service Operation Vessel
}

C_LT = {
    0: Day_rate[0] * days * months + Mobilization_rate[0],  # Cost for long-term chartering of Crew Transfer Vessel
    1: Day_rate[1] * days * months + Mobilization_rate[1],  # Cost for long-term chartering of Service Operation Vessel
}

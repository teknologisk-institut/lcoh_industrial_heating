# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:34:46 2024

@author: WBM
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


input_params = {
    "Gas_price": 39.3, #[EUR/MWh] # ENS AF24
    "Biomass_price": 43.6, # [EUR/MWh]      
    "CO2_tax": 100, #[EUR/tons]
    "CO2_emission_factor": 0.198, #[tons/MWh]    
    # Heat pumps
    "SpecificInvestment_HP80": 0.741, #[mio. EUR/MWthermal] heat pump delivering process heat at 80°C. i.e delivering the source
    "SpecificInvestment_HTHP": 1.07, #[mio. EUR/MWthermal] heat pump delivering process heat at 150°C. 
    "O_and_M_fixed_HP80": 2280, #[EUR/MW/year]
    "O_and_M_fixed_HTHP": 991.8,  #[EUR/MW/year]
    "O_and_M_var_HP80": 3.2, #[EUR/MWh]
    "O_and_M_var_HTHP": 3.2,  #[EUR/MWh]
    "T_source_in": 80, #[°C]
    "T_source_out": 60, #[°C]
    "T_sink_in": 140, #[°C]
    "T_sink_out": 150, #[°C]
    "eta_Lorenz": 0.5, #[%]	
    "T_source_sourceHP_in": 20, #[°C] inlet temperature of the heat source for the source heat pump
    "T_source_sourceHP_out": 10, #[°C] outlet temperature of the heat source for the source heat pump
    # Electric boiler
    "SpecificInvestment_ElectricBoiler": 0.1, #[mio. EUR/MWthermal] 
    "O_and_M_fixed_ElectricBoiler": 1203.5, #[EUR/MW/year]
    "O_and_M_var_ElectricBoiler": 0.99, #[EUR/MWh]
    "Eta_electricBoiler": 0.99, #[%]	
    # Natural-/biogas boiler			
    "SpecificInvestment_NGBoiler" : 0.05, #[mio. EUR/MWthermal]	
    "O_and_M_fixed_NGBoiler": 2166, #[EUR/MW/year]
    "O_and_M_var_NGBoiler": 1.14,#[EUR/MWh]
    "Eta_NGboiler": 0.94, #[%]	
    # Biomass boiler
    "SpecificInvestment_BiomassBoiler": 0.674, #[mio. EUR/MWthermal]	
    "O_and_M_fixed_BiomassBoiler": 40926, #[EUR/MW/year]
    "O_and_M_var_BiomassBoiler": 3.22, #[EUR/MWh]
    "Eta_biomassBoiler": 0.90, #[%]	
    # H2-boiler		
    "SpecificInvestment_H2Boiler": 0.05, #[mio. EUR/MWthermal]
    "O_and_M_fixed_H2Boiler": 2166, #[EUR/MW/year]
    "O_and_M_var_H2Boiler": 1.14, #[EUR/MWh]
    "Eta_H2boiler": 0.94, #[%]	
    # "Assumptions for LCOH 			
    "lifetime": 20, #[years]
    "interest_rate": 0.05, #[-]	
    "inflation_rate": 0.02, #[-]
    # Hydrogen production and transport cost input
    "LHV_H2": 0.03333, #[MWh/kg] lower heating value of hydrogen
    "electrolyser_efficiency": 0.622, # from Technology catalogue year 2035
    "lifetime_electrolyzer": 20, #[years]
    "specific_investment_electrolyzer": 621.5*1000, #[EUR/MW electricity input]
    "OandM_electrolyzer": 4, #[% of specific investment/year]
    "operating_hours_electrolyzer": 8765, #[h] in the non-optimized operation 
    # Battery Electricity storage 
    "specific_investment_electricity_storage": 0.77, #[mio. EUR/MWh]
    "O_and_M_fixed_El_storage": 0.015, # [-] fraction of total investment
    "O_and_M_var_El_storage": 2.24, #[EUR/MWh]
    # Thermal storage
    "specific_investment_thermal_storage": 0.05, #[mio. EUR/MWh]
    "o_and_m_thermal_storage": 0.02, # [-] fraction of total investment
    "o_and_m_var_thermal_storage": 0, #[EUR/MWh]
    "operating_hours_peak_shaving": 8765, #[h]
    "efficiency_storage": 1, #[-]    
    # variability of electricity price, daily - defined as a linear function
    "reduced_el_price_linear_func_a": 0.56, #these are for DK2 linear regression on relative prices
    "reduced_el_price_linear_func_b": 0.44,
    # variability of electricity price, yearly - defined as a linear function
    "electrolyser_linear_func_a":0.0105,
    "electrolyser_linear_func_b":-0.0489,
    # factors for calculating grid connection costs
    "grid_connection_factor_HP": 1,
    "grid_connection_factor_EB": 3,
    "grid_connection_factor_fuelBoiler": 0,
    # misc
    "days_per_year": 365
    }


# Helper function to calculate effective interest rate and CRF
def calculate_crf(interest, inflation, lifetime):
    i_eff = (1 + interest) / (1 + inflation) - 1  # Effective interest rate
    crf = i_eff * np.power((1 + i_eff), lifetime) / (np.power((1 + i_eff), lifetime) - 1)  # Capital Recovery Factor
    return crf

class Technology:
    def __init__(
        self, 
        name, 
        specific_investment_tech, 
        efficiency, 
        o_and_m_fixed, 
        o_and_m_var, 
        capacity, 
        grid_cost_factor, 
        lifetime, 
        specific_investment_storage=0,  # Default to 0 (no storage investment)
        capacity_storage=0,  # Default to 0 (no storage capacity)
        o_and_m_thermal_storage=0,  # Default to 0 (no storage O&M)
        efficiency_storage=1  # Default to 1 (no efficiency loss)
    ):
        self.name = name
        self.specific_investment_tech = specific_investment_tech
        self.efficiency = efficiency
        self.o_and_m_fixed = o_and_m_fixed
        self.o_and_m_var = o_and_m_var
        self.capacity = capacity
        self.grid_cost_factor = grid_cost_factor
        self.lifetime = lifetime
        self.specific_investment_storage = specific_investment_storage
        self.capacity_storage = capacity_storage
        self.o_and_m_thermal_storage = o_and_m_thermal_storage
        self.efficiency_storage = efficiency_storage

    def calculate_investment(self):
        """Calculate total investment based on specific investment given in in mil.€/MW heating capacity ."""
        return self.specific_investment_tech * self.capacity * 1e6 + self.specific_investment_storage * self.capacity_storage * 1e6

    def calculate_o_and_m(self, yearly_production):
        """Calculate total O&M costs."""
        return self.o_and_m_fixed * self.capacity + self.o_and_m_var * yearly_production + self.o_and_m_thermal_storage * self.capacity_storage

    def calculate_grid_connection_cost(self):
        """Calculate grid connection cost, based on correlation from Pieper et al paper"""
        return (0.12908 * self.capacity + 0.01085)*1.15 * 1e6 * self.grid_cost_factor    

    def calculate_fuel_consumption(self, yearly_production):
        """Placeholder method for fuel consumption calculation, as this differs for different technologies"""
        raise NotImplementedError("Subclasses must implement the fuel_consumption method.")      
        
    def calculate_heat_source_cost(self, yearly_production, heat_source_price):
        """Default behavior: return 0 for technologies that don't use a heat source."""
        return 0

    def calculate_CO2_tax(self, yearly_heat_production, CO2_emission_factor, specific_CO2_tax):
        """Calculate the yearly expences for CO2-tax based on CO2 emmision of the fuel."""
        fuel_consumption = self.calculate_fuel_consumption(yearly_heat_production)
        CO2_tax = fuel_consumption*CO2_emission_factor*specific_CO2_tax      
        return CO2_tax 


class HeatPump(Technology):
    def __init__(
            self, 
            name, 
            specific_investment_tech, 
            eta_lorenz, 
            t_source_in, 
            t_source_out, 
            t_sink_in, 
            t_sink_out, 
            o_and_m_fixed, 
            o_and_m_var, 
            capacity, 
            grid_cost_factor,
            lifetime, 
            specific_investment_storage=0, 
            capacity_storage=0, 
            o_and_m_thermal_storage=0, 
            efficiency_storage=1
        ):
        super().__init__(
            name, 
            specific_investment_tech, 
            None, 
            o_and_m_fixed, 
            o_and_m_var, 
            capacity, 
            grid_cost_factor,
            lifetime,
            specific_investment_storage, 
            capacity_storage, 
            o_and_m_thermal_storage, 
            efficiency_storage
            )
        
        self.eta_lorenz = eta_lorenz
        self.t_source_in = t_source_in
        self.t_source_out = t_source_out
        self.t_sink_in = t_sink_in
        self.t_sink_out = t_sink_out

    def calculate_cop(self):
        """Calculate the Coefficient of Performance (COP) for the heat pump."""
        T_source_in_K = self.t_source_in + 273.15 #[K]
        T_source_out_K = self.t_source_out + 273.15 #
        T_sink_in_K = self.t_sink_in + 273.15 #[K]
        T_sink_out_K = self.t_sink_out+ 273.15 #[K]
    
        T_LM_source = (T_source_in_K-T_source_out_K)/(np.log(T_source_in_K/T_source_out_K))
        T_LM_sink = (T_sink_in_K-T_sink_out_K)/(np.log(T_sink_in_K/T_sink_out_K))
    
        COP_Lorenz = T_LM_sink/(T_LM_sink - T_LM_source) 
        COP = self.eta_lorenz*COP_Lorenz
        
        return COP

    def calculate_heat_source_capacity(self):
        """Calculate heat source consumption based on the COP."""
        COP = self.calculate_cop()
        heat_source_capacity = self.capacity * (COP-1)/COP
        return heat_source_capacity

    def calculate_fuel_consumption(self, yearly_heat_production):
        """Calculate fuel consumption based on COP ."""
        COP = self.calculate_cop()
        fuel_consumption = yearly_heat_production / COP / self.efficiency_storage
        return fuel_consumption

    def calculate_fuel_cost(self, yearly_heat_production, electricity_price):
        """Calculate fuel cost based on COP and electricity price."""
        fuel_consumption = self.calculate_fuel_consumption(yearly_heat_production)
        fuel_cost = fuel_consumption * electricity_price
        return fuel_cost
    
    def calculate_heat_source_cost(self, yearly_heat_production, heat_source_price):
        heat_source_capacity = self.calculate_heat_source_capacity()
        heat_source_cost = heat_source_capacity * yearly_heat_production/self.capacity * heat_source_price
        return heat_source_cost
    

class Boiler(Technology):
    def __init__(
            self, 
            name,
            specific_investment_tech,
            efficiency,
            o_and_m_fixed, 
            o_and_m_var, 
            capacity,
            grid_cost_factor,
            lifetime, 
            specific_investment_storage=0, 
            capacity_storage=0, 
            o_and_m_thermal_storage=0, 
            efficiency_storage=1
        ):
        super().__init__(
            name, 
            specific_investment_tech,
            efficiency,
            o_and_m_fixed, 
            o_and_m_var, 
            capacity, 
            grid_cost_factor,
            lifetime, 
            specific_investment_storage, 
            capacity_storage, 
            o_and_m_thermal_storage, 
            efficiency_storage
            )


    def calculate_fuel_consumption(self, yearly_heat_production):
        """Calculate fuel cost based on efficiency and fuel price."""
        fuel_consumption = yearly_heat_production / self.efficiency / self.efficiency_storage
        return fuel_consumption

    def calculate_fuel_cost(self, yearly_heat_production, fuel_price):
        """Calculate fuel cost based on efficiency and fuel price."""
        fuel_consumption = self.calculate_fuel_consumption(yearly_heat_production)
        fuel_cost = fuel_consumption * fuel_price
        return fuel_cost
    


def calculate_LCOH_for_technology(tech,yearly_heat_production,fuel_price, heat_source_price, specific_CO2_tax, CO2_emission_factor,interest, inflation):
    
    crf = calculate_crf(interest, inflation,tech.lifetime)
    investment = tech.calculate_investment()
    fuel_cost = tech. calculate_fuel_cost(yearly_heat_production, fuel_price)
    O_and_M = tech.calculate_o_and_m(yearly_heat_production)
    heat_source = tech.calculate_heat_source_cost(yearly_heat_production, heat_source_price)
    grid_connection = tech.calculate_grid_connection_cost()
    CO2_tax = tech.calculate_CO2_tax(yearly_heat_production, CO2_emission_factor, specific_CO2_tax)
    
    LCO_investment = (investment*crf)/(yearly_heat_production)
    LCO_fuel = fuel_cost/yearly_heat_production
    LCO_heatsource = heat_source/yearly_heat_production
    LCO_maintenance = O_and_M/yearly_heat_production
    LCO_grid = grid_connection*crf/yearly_heat_production
    LCO_CO2 = CO2_tax/yearly_heat_production
    LCOH = LCO_investment + LCO_fuel + LCO_maintenance + LCO_heatsource + LCO_grid + LCO_CO2 
    
    return LCOH, LCO_investment, LCO_grid, LCO_fuel, LCO_heatsource, LCO_maintenance, LCO_CO2 

def calculate_hydrogen_price(LHV_H2, electricity_price, electrolyser_efficiency, specific_investment_electrolyzer, OandM_electrolyzer, interest, inflation,lifetime, operating_hours):
    crf = calculate_crf(interest, inflation,lifetime)
    specific_energy_demand = LHV_H2/electrolyser_efficiency
    LCOHydrogen = specific_energy_demand*((crf + OandM_electrolyzer/100)*specific_investment_electrolyzer/operating_hours + electricity_price)
    LCOHydrogen_OPEX = specific_energy_demand*((OandM_electrolyzer/100)*specific_investment_electrolyzer/operating_hours + electricity_price)
    LCOHydrogen_CAPEX = specific_energy_demand*((crf)*specific_investment_electrolyzer/operating_hours)
    hydrogen_price = LCOHydrogen/LHV_H2 #[EUR/MWh]
    return hydrogen_price, LCOHydrogen, LCOHydrogen_OPEX, LCOHydrogen_CAPEX 





def LCOH_for_technologies(parameters, electricity_price, demand_hours, yearly_heat_production):
    """this is the main function, where all technologies are defined. LCOH is found for each technology and storages and capacities are optimized"""
    # Extract common parameters
    CO2_tax = parameters["CO2_tax"]
    interest_rate = parameters["interest_rate"]
    inflation_rate = parameters["inflation_rate"]
    days_per_year = parameters["days_per_year"]

    # # Initialize results dictionary
    results = {}

    # Calculate LCOH for heat source heat pumps

    # Initialize technologies and calculate LCOH for each technology
    # Heat Pump without storage. This heat pump is used to calculate LCOH with and without heat source costs.
    HP1 = HeatPump(
        name="HTHP 150°C",
        specific_investment_tech=parameters["SpecificInvestment_HTHP"],
        eta_lorenz=parameters["eta_Lorenz"],
        t_source_in=parameters["T_source_in"],
        t_source_out=parameters["T_source_out"],
        t_sink_in=parameters["T_sink_in"],
        t_sink_out=parameters["T_sink_out"],
        o_and_m_fixed=parameters["O_and_M_fixed_HTHP"],
        o_and_m_var=parameters["O_and_M_var_HTHP"],
        capacity=yearly_heat_production / demand_hours,  # Capacity based on demand hours
        grid_cost_factor=parameters["grid_connection_factor_HP"],
        lifetime=parameters["lifetime"]
    )
 
    # Heat source heat pump for HP1
    heat_source_HP1 = HeatPump(
        name="Heat Source HP for HP1",
        specific_investment_tech=parameters["SpecificInvestment_HP80"],
        eta_lorenz=parameters["eta_Lorenz"],
        t_source_in=parameters["T_source_sourceHP_in"],
        t_source_out=parameters["T_source_sourceHP_out"],
        t_sink_in=parameters["T_source_in"],  # Heat source pump sinks into the main heat pump's source
        t_sink_out=parameters["T_source_out"],
        o_and_m_fixed=parameters["O_and_M_fixed_HP80"],
        o_and_m_var=parameters["O_and_M_var_HP80"],
        capacity=HP1.calculate_heat_source_capacity(),
        grid_cost_factor=parameters["grid_connection_factor_HP"],
        lifetime=parameters["lifetime"]
    )

    heat_source_price_HP1, _, _, _, _, _, _ = calculate_LCOH_for_technology(
        tech=heat_source_HP1,
        yearly_heat_production=heat_source_HP1.capacity*demand_hours,
        fuel_price=electricity_price,
        heat_source_price=0,
        specific_CO2_tax=CO2_tax,
        CO2_emission_factor=0,
        interest=interest_rate,
        inflation=inflation_rate,
    )
    
    # Calculate LCOH for HP1 using heat source price
    LCOH_HP1, LCO_investment, LCO_grid, LCO_fuel, LCO_heatsource, LCO_maintenance, LCO_CO2 = calculate_LCOH_for_technology(
        tech=HP1,
        yearly_heat_production=yearly_heat_production,
        fuel_price=electricity_price,
        heat_source_price=heat_source_price_HP1,
        specific_CO2_tax=CO2_tax,
        CO2_emission_factor=0,
        interest=interest_rate,
        inflation=inflation_rate,
    )
    results["HP1"] = {
        "LCOH": LCOH_HP1,
        "LCO_investment": LCO_investment,
        "LCO_grid": LCO_grid,
        "LCO_fuel": LCO_fuel,
        "LCO_heatsource": LCO_heatsource,
        "LCO_maintenance": LCO_maintenance,
        "LCO_CO2": LCO_CO2,
    }

    # Calculate LCOH for HP1 with free heat source
    LCOH_HP1_free_source, LCO_investment, LCO_grid, LCO_fuel, LCO_heatsource, LCO_maintenance, LCO_CO2 = calculate_LCOH_for_technology(
        tech=HP1,
        yearly_heat_production=yearly_heat_production,
        fuel_price=electricity_price,
        heat_source_price=0,
        specific_CO2_tax=CO2_tax,
        CO2_emission_factor=0,
        interest=interest_rate,
        inflation=inflation_rate,
    )
    results["HP1_free_source"] = {
        "LCOH": LCOH_HP1_free_source,
        "LCO_investment": LCO_investment,
        "LCO_grid": LCO_grid,
        "LCO_fuel": LCO_fuel,
        "LCO_heatsource": LCO_heatsource,
        "LCO_maintenance": LCO_maintenance,
        "LCO_CO2": LCO_CO2,
    }
    
   ########################## Heat Pump with storage. This heat pump is optimized in capacity for operation with heat source costs.
    
   # Initialize lists for tracking LCOH and operating hours
    tech_daily_operating_hours_list = []
    LCOH_HP_temp_list = []
    LCOH_components_list = []  # To store all LCOH components for each iteration
    
    # Loop through daily operating hours to find the minimum LCOH
    for i in range(1, 25):
        tech_daily_operation_hours = i   
        capacity_HP_temp, thermal_storage_capacity_temp, operating_hours_temp = determine_capacities_storage(
            tech_daily_operation_hours, yearly_heat_production, demand_hours, days_per_year)
            
        electricity_price_HP_temp = electricity_price * calculate_factor_reduced_electricity_price(
            operating_hours_temp, parameters["reduced_el_price_linear_func_a"], parameters["reduced_el_price_linear_func_b"]
        )
    
        # Define the heat pump with temporary capacities
        HP_temp = HeatPump(
            name="HTHP 150° + thermal storage,temp",
            specific_investment_tech=parameters["SpecificInvestment_HTHP"],
            eta_lorenz=parameters["eta_Lorenz"],
            t_source_in=parameters["T_source_in"],
            t_source_out=parameters["T_source_out"],
            t_sink_in=parameters["T_sink_in"],
            t_sink_out=parameters["T_sink_out"],
            o_and_m_fixed=parameters["O_and_M_fixed_HTHP"],
            o_and_m_var=parameters["O_and_M_var_HTHP"],
            capacity=capacity_HP_temp,
            grid_cost_factor=parameters["grid_connection_factor_HP"],
            lifetime=parameters["lifetime"],
            specific_investment_storage=parameters["specific_investment_thermal_storage"],
            capacity_storage=thermal_storage_capacity_temp,
            o_and_m_thermal_storage=parameters["o_and_m_thermal_storage"], 
            efficiency_storage=parameters["efficiency_storage"]
        )
        
        # Define the heat source heat pump
        heat_source_HP_temp = HeatPump(
            name="Heat Source HP for HP_temp",
            specific_investment_tech=parameters["SpecificInvestment_HP80"],
            eta_lorenz=parameters["eta_Lorenz"],
            t_source_in=parameters["T_source_sourceHP_in"],
            t_source_out=parameters["T_source_sourceHP_out"],
            t_sink_in=parameters["T_source_in"],  # Heat source pump sinks into the main heat pump's source
            t_sink_out=parameters["T_source_out"],
            o_and_m_fixed=parameters["O_and_M_fixed_HP80"],
            o_and_m_var=parameters["O_and_M_var_HP80"],
            capacity=HP_temp.calculate_heat_source_capacity(),
            grid_cost_factor=parameters["grid_connection_factor_HP"],
            lifetime=parameters["lifetime"]
        )
        
        # Calculate heat source price for the temporary heat pump
        heat_source_price_HP_temp, _, _, _, _, _, _ = calculate_LCOH_for_technology(
            tech=heat_source_HP_temp,
            yearly_heat_production=heat_source_HP_temp.capacity * operating_hours_temp,
            fuel_price=electricity_price_HP_temp,
            heat_source_price=0,
            specific_CO2_tax=CO2_tax,
            CO2_emission_factor=0,
            interest=interest_rate,
            inflation=inflation_rate,
        )
       
        # Calculate LCOH for HP2 using the heat source price
        LCOH_HP_temp, LCO_investment, LCO_grid, LCO_fuel, LCO_heatsource, LCO_maintenance, LCO_CO2 = calculate_LCOH_for_technology(
            tech=HP_temp,
            yearly_heat_production=yearly_heat_production,
            fuel_price=electricity_price_HP_temp,
            heat_source_price=heat_source_price_HP_temp,
            specific_CO2_tax=CO2_tax,
            CO2_emission_factor=0,
            interest=interest_rate,
            inflation=inflation_rate,
        )
        
        # Append results to lists for tracking
        tech_daily_operating_hours_list.append(tech_daily_operation_hours)
        LCOH_HP_temp_list.append(LCOH_HP_temp)
        LCOH_components_list.append({
            "LCO_investment": LCO_investment,
            "LCO_grid": LCO_grid,
            "LCO_fuel": LCO_fuel,
            "LCO_heatsource": LCO_heatsource,
            "LCO_maintenance": LCO_maintenance,
            "LCO_CO2": LCO_CO2
        })
    
    # Find the minimum LCOH and corresponding daily operating hours
    min_LCOH_HP_temp = min(LCOH_HP_temp_list)
    optimal_index = LCOH_HP_temp_list.index(min_LCOH_HP_temp)
    optimal_daily_operating_hours_HP2 = tech_daily_operating_hours_list[optimal_index]
    
    # Extract LCOH components for the optimal solution
    optimal_LCOH_components = LCOH_components_list[optimal_index]
    
    # Store all results in the results dictionary
    results["HP2"] = {
        "LCOH": min_LCOH_HP_temp,
        "optimal_daily_operating_hours": optimal_daily_operating_hours_HP2,
        "LCO_investment": optimal_LCOH_components["LCO_investment"],
        "LCO_grid": optimal_LCOH_components["LCO_grid"],
        "LCO_fuel": optimal_LCOH_components["LCO_fuel"],
        "LCO_heatsource": optimal_LCOH_components["LCO_heatsource"],
        "LCO_maintenance": optimal_LCOH_components["LCO_maintenance"],
        "LCO_CO2": optimal_LCOH_components["LCO_CO2"]
        }

    ########################## Heat Pump with storage and free source. This heat pump is optimized in capacity.
    
    # Initialize lists for tracking LCOH and operating hours
    tech_daily_operating_hours_list = []
    LCOH_HP_temp_list = []
    LCOH_components_list = []  # To store all LCOH components for each iteration
    
    # Loop through daily operating hours to find the minimum LCOH
    for i in range(1, 25):
        tech_daily_operation_hours = i   
        capacity_HP_temp_free, thermal_storage_capacity_temp_free, operating_hours_temp_free = determine_capacities_storage(
            tech_daily_operation_hours, yearly_heat_production, demand_hours, days_per_year)
            
        electricity_price_HP_temp_free = electricity_price * calculate_factor_reduced_electricity_price(
            operating_hours_temp_free, parameters["reduced_el_price_linear_func_a"], parameters["reduced_el_price_linear_func_b"]
        )
    
        # Define the heat pump with temporary capacities
        HP_temp_free = HeatPump(
            name="HTHP 150° + thermal storage,temp_free",
            specific_investment_tech=parameters["SpecificInvestment_HTHP"],
            eta_lorenz=parameters["eta_Lorenz"],
            t_source_in=parameters["T_source_in"],
            t_source_out=parameters["T_source_out"],
            t_sink_in=parameters["T_sink_in"],
            t_sink_out=parameters["T_sink_out"],
            o_and_m_fixed=parameters["O_and_M_fixed_HTHP"],
            o_and_m_var=parameters["O_and_M_var_HTHP"],
            capacity=capacity_HP_temp_free,
            grid_cost_factor=parameters["grid_connection_factor_HP"],
            lifetime=parameters["lifetime"],
            specific_investment_storage=parameters["specific_investment_thermal_storage"],
            capacity_storage=thermal_storage_capacity_temp_free,
            o_and_m_thermal_storage=parameters["o_and_m_thermal_storage"], 
            efficiency_storage=parameters["efficiency_storage"]
        )
        
       
        # Calculate LCOH for HP2 using the heat source price
        LCOH_HP_temp_free, LCO_investment, LCO_grid, LCO_fuel, LCO_heatsource, LCO_maintenance, LCO_CO2 = calculate_LCOH_for_technology(
            tech=HP_temp_free,
            yearly_heat_production=yearly_heat_production,
            fuel_price=electricity_price_HP_temp_free,
            heat_source_price=0,
            specific_CO2_tax=CO2_tax,
            CO2_emission_factor=0,
            interest=interest_rate,
            inflation=inflation_rate,
        )
        
        # Append results to lists for tracking
        tech_daily_operating_hours_list.append(tech_daily_operation_hours)
        LCOH_HP_temp_list.append(LCOH_HP_temp_free)
        LCOH_components_list.append({
            "LCO_investment": LCO_investment,
            "LCO_grid": LCO_grid,
            "LCO_fuel": LCO_fuel,
            "LCO_heatsource": LCO_heatsource,
            "LCO_maintenance": LCO_maintenance,
            "LCO_CO2": LCO_CO2
        })
    
    # Find the minimum LCOH and corresponding daily operating hours
    min_LCOH_HP_temp_free = min(LCOH_HP_temp_list)
    optimal_index = LCOH_HP_temp_list.index(min_LCOH_HP_temp_free)
    optimal_daily_operating_hours_HP2_free = tech_daily_operating_hours_list[optimal_index]
    
    # Extract LCOH components for the optimal solution
    optimal_LCOH_components = LCOH_components_list[optimal_index]
   
    # Store all results in the results dictionary
    results["HP2_free_source"] = {
        "LCOH": min_LCOH_HP_temp_free,
        "optimal_daily_operating_hours": optimal_daily_operating_hours_HP2_free,
        "LCO_investment": optimal_LCOH_components["LCO_investment"],
        "LCO_grid": optimal_LCOH_components["LCO_grid"],
        "LCO_fuel": optimal_LCOH_components["LCO_fuel"],
        "LCO_heatsource": optimal_LCOH_components["LCO_heatsource"],
        "LCO_maintenance": optimal_LCOH_components["LCO_maintenance"],
        "LCO_CO2": optimal_LCOH_components["LCO_CO2"]
        }

    # Electric Boiler
    EB1 = Boiler(
        name="Electric boiler",
        specific_investment_tech=parameters["SpecificInvestment_ElectricBoiler"],
        efficiency=parameters["Eta_electricBoiler"],
        o_and_m_fixed=parameters["O_and_M_fixed_ElectricBoiler"],
        o_and_m_var=parameters["O_and_M_var_ElectricBoiler"],
        capacity=yearly_heat_production / demand_hours,
        grid_cost_factor=parameters["grid_connection_factor_EB"],
        lifetime=parameters["lifetime"]
    )

    # Calculate LCOH for Electric boiler 
    LCOH_EB1, LCO_investment, LCO_grid, LCO_fuel, LCO_heatsource, LCO_maintenance, LCO_CO2 = calculate_LCOH_for_technology(
        tech=EB1,
        yearly_heat_production=yearly_heat_production,
        fuel_price=electricity_price,
        heat_source_price=0,
        specific_CO2_tax=CO2_tax,
        CO2_emission_factor=0,
        interest=interest_rate,
        inflation=inflation_rate,
    )
    results["EB1"] = {
        "LCOH": LCOH_EB1,
        "LCO_investment": LCO_investment,
        "LCO_grid": LCO_grid,
        "LCO_fuel": LCO_fuel,
        "LCO_heatsource": LCO_heatsource,
        "LCO_maintenance": LCO_maintenance,
        "LCO_CO2": LCO_CO2,
    }
    
    # Initialize lists for tracking LCOH and operating hours
    tech_daily_operating_hours_list = []
    LCOH_EB_temp_list = []
    LCOH_components_list = []  # To store all LCOH components for each iteration
    
    # Loop through daily operating hours to find the minimum LCOH
    
    for i in range(1, 25):
        tech_daily_operation_hours = i   
        capacity_EB_temp, thermal_storage_capacity_temp_EB, operating_hours_temp_EB = determine_capacities_storage(
            tech_daily_operation_hours, yearly_heat_production, demand_hours, days_per_year)
            
        electricity_price_EB_temp = electricity_price * calculate_factor_reduced_electricity_price(
            operating_hours_temp_EB, parameters["reduced_el_price_linear_func_a"], parameters["reduced_el_price_linear_func_b"]
        )
    
        # Define the heat pump with temporary capacities
        EB_temp = Boiler(
            name="Electric boiler + thermal storage",
            specific_investment_tech=parameters["SpecificInvestment_ElectricBoiler"],
            efficiency=parameters["Eta_electricBoiler"],
            o_and_m_fixed=parameters["O_and_M_fixed_ElectricBoiler"],
            o_and_m_var=parameters["O_and_M_var_ElectricBoiler"],
            capacity=capacity_EB_temp,
            grid_cost_factor=parameters["grid_connection_factor_EB"],
            lifetime=parameters["lifetime"],
            specific_investment_storage=parameters["specific_investment_thermal_storage"],
            capacity_storage=thermal_storage_capacity_temp_EB,
            o_and_m_thermal_storage = parameters["o_and_m_thermal_storage"], 
            efficiency_storage=parameters["efficiency_storage"]
        )
        
       
        # Calculate LCOH for EB_temp using the heat source price
        LCOH_EB_temp, LCO_investment, LCO_grid, LCO_fuel, LCO_heatsource, LCO_maintenance, LCO_CO2 = calculate_LCOH_for_technology(
            tech=EB_temp,
            yearly_heat_production=yearly_heat_production,
            fuel_price=electricity_price_EB_temp,
            heat_source_price=0,
            specific_CO2_tax=CO2_tax,
            CO2_emission_factor=0,
            interest=interest_rate,
            inflation=inflation_rate,
        )
        
        
 
        # Append results to lists for tracking
        tech_daily_operating_hours_list.append(tech_daily_operation_hours)
        LCOH_EB_temp_list.append(LCOH_EB_temp)
        LCOH_components_list.append({
            "LCO_investment": LCO_investment,
            "LCO_grid": LCO_grid,
            "LCO_fuel": LCO_fuel,
            "LCO_heatsource": LCO_heatsource,
            "LCO_maintenance": LCO_maintenance,
            "LCO_CO2": LCO_CO2
        })
    
    # Find the minimum LCOH and corresponding daily operating hours
    min_LCOH_EB_temp = min(LCOH_EB_temp_list)
    optimal_index = LCOH_EB_temp_list.index(min_LCOH_EB_temp)
    optimal_daily_operating_hours_EB2 = tech_daily_operating_hours_list[optimal_index]
    
    # Extract LCOH components for the optimal solution
    optimal_LCOH_components = LCOH_components_list[optimal_index]
    
    # Store all results in the results dictionary
    results["EB2"] = {
        "LCOH": min_LCOH_EB_temp,
        "optimal_daily_operating_hours": optimal_daily_operating_hours_EB2,
        "LCO_investment": optimal_LCOH_components["LCO_investment"],
        "LCO_grid": optimal_LCOH_components["LCO_grid"],
        "LCO_fuel": optimal_LCOH_components["LCO_fuel"],
        "LCO_heatsource": optimal_LCOH_components["LCO_heatsource"],
        "LCO_maintenance": optimal_LCOH_components["LCO_maintenance"],
        "LCO_CO2": optimal_LCOH_components["LCO_CO2"]
        }
    


    # Biomass Boiler
    BB = Boiler(
        name="Biomass boiler",
        specific_investment_tech=parameters["SpecificInvestment_BiomassBoiler"],
        efficiency=parameters["Eta_biomassBoiler"],
        o_and_m_fixed=parameters["O_and_M_fixed_BiomassBoiler"],
        o_and_m_var=parameters["O_and_M_var_BiomassBoiler"],
        capacity=yearly_heat_production / demand_hours,
        grid_cost_factor=parameters["grid_connection_factor_fuelBoiler"],
        lifetime=parameters["lifetime"]
    )

    # Calculate LCOH for biomass boiler
    LCOH_Biomass, LCO_investment, LCO_grid, LCO_fuel, _, LCO_maintenance, LCO_CO2 = calculate_LCOH_for_technology(
        tech=BB,
        yearly_heat_production=yearly_heat_production,
        fuel_price=parameters["Biomass_price"],
        heat_source_price=0,  # No heat source for boilers
        specific_CO2_tax=CO2_tax,
        CO2_emission_factor=0,
        interest=interest_rate,
        inflation=inflation_rate,
    )
    results["Biomass Boiler"] = {
        "LCOH": LCOH_Biomass,
        "LCO_investment": LCO_investment,
        "LCO_grid": LCO_grid,
        "LCO_fuel": LCO_fuel,
        "LCO_heatsource": 0,  # No heat source cost for boilers
        "LCO_maintenance": LCO_maintenance,
        "LCO_CO2": LCO_CO2,
    }

    # Hydrogen boiler
    HB1 = Boiler(
        name="Hydrogen boiler",
        specific_investment_tech=parameters["SpecificInvestment_H2Boiler"],
        efficiency=parameters["Eta_H2boiler"],
        o_and_m_fixed=parameters["O_and_M_fixed_H2Boiler"],
        o_and_m_var=parameters["O_and_M_var_H2Boiler"],
        capacity=yearly_heat_production / demand_hours,
        grid_cost_factor=parameters["grid_connection_factor_fuelBoiler"],
        lifetime=parameters["lifetime"]
    )
    
    hydrogen_price,_,_,_ = calculate_hydrogen_price(
        LHV_H2=parameters["LHV_H2"],
        electricity_price=electricity_price, 
        electrolyser_efficiency=parameters["electrolyser_efficiency"],
        specific_investment_electrolyzer = parameters["specific_investment_electrolyzer"],
        OandM_electrolyzer= parameters["OandM_electrolyzer"],
        interest=interest_rate,
        inflation=inflation_rate,
        lifetime=parameters["lifetime_electrolyzer"],
        operating_hours = parameters["operating_hours_electrolyzer"]) 
    
    # Calculate LCOH for Hydrogen boiler 
    LCOH_HB1, LCO_investment, LCO_grid, LCO_fuel, LCO_heatsource, LCO_maintenance, LCO_CO2 = calculate_LCOH_for_technology(
        tech=HB1,
        yearly_heat_production=yearly_heat_production,
        fuel_price=hydrogen_price,
        heat_source_price=0,
        specific_CO2_tax=CO2_tax,
        CO2_emission_factor=0,
        interest=interest_rate,
        inflation=inflation_rate,
    )
    results["HB1"] = {
        "LCOH": LCOH_HB1,
        "LCO_investment": LCO_investment,
        "LCO_grid": LCO_grid,
        "LCO_fuel": LCO_fuel,
        "LCO_heatsource": LCO_heatsource,
        "LCO_maintenance": LCO_maintenance,
        "LCO_CO2": LCO_CO2,
    }

    # Initialize lists for tracking LCOH and operating hours
    electrolyzer_utilization_list = []
    LCOH_HB_temp_list = []
    LCOH_components_list = []  # To store all LCOH components for each iteration

    for i in range(1, 25):
        
        factor_reduced_hours = i/25
            
        electricity_price_HB_temp = electricity_price * calculate_factor_reduced_electricity_price_electrolyzer(factor_reduced_hours, parameters["electrolyser_linear_func_a"], parameters["electrolyser_linear_func_b"])
  
        hydrogen_price_temp,_,_,_ = calculate_hydrogen_price(
            LHV_H2=parameters["LHV_H2"],
            electricity_price=electricity_price_HB_temp, 
            electrolyser_efficiency=parameters["electrolyser_efficiency"],
            specific_investment_electrolyzer = parameters["specific_investment_electrolyzer"],
            OandM_electrolyzer= parameters["OandM_electrolyzer"],
            interest=interest_rate,
            inflation=inflation_rate,
            lifetime=parameters["lifetime_electrolyzer"],
            operating_hours = factor_reduced_hours*parameters["operating_hours_electrolyzer"]) 
    
        # Calculate LCOH for Hydrogen boiler 
        LCOH_HB_temp, LCO_investment, LCO_grid, LCO_fuel, LCO_heatsource, LCO_maintenance, LCO_CO2 = calculate_LCOH_for_technology(
            tech=HB1,
            yearly_heat_production=yearly_heat_production,
            fuel_price=hydrogen_price_temp,
            heat_source_price=0,
            specific_CO2_tax=CO2_tax,
            CO2_emission_factor=0,
            interest=interest_rate,
            inflation=inflation_rate,
            )
        
        
        # Append results to lists for tracking
        electrolyzer_utilization_list.append(factor_reduced_hours)
        LCOH_HB_temp_list.append(LCOH_HB_temp)
        LCOH_components_list.append({
            "LCO_investment": LCO_investment,
            "LCO_grid": LCO_grid,
            "LCO_fuel": LCO_fuel,
            "LCO_heatsource": LCO_heatsource,
            "LCO_maintenance": LCO_maintenance,
            "LCO_CO2": LCO_CO2
        })
    
    # Find the minimum LCOH and corresponding daily operating hours
    min_LCOH_HB_temp = min(LCOH_HB_temp_list)
    optimal_index = LCOH_HB_temp_list.index(min_LCOH_HB_temp)
    optimal_factor_reduced_hours_HB2 = electrolyzer_utilization_list[optimal_index]
    
    # Extract LCOH components for the optimal solution
    optimal_LCOH_components = LCOH_components_list[optimal_index]
    
    # Store all results in the results dictionary
    results["HB2"] = {
        "LCOH": min_LCOH_HB_temp,
        "optimal_utilization factor": optimal_factor_reduced_hours_HB2 ,
        "LCO_investment": optimal_LCOH_components["LCO_investment"],
        "LCO_grid": optimal_LCOH_components["LCO_grid"],
        "LCO_fuel": optimal_LCOH_components["LCO_fuel"],
        "LCO_heatsource": optimal_LCOH_components["LCO_heatsource"],
        "LCO_maintenance": optimal_LCOH_components["LCO_maintenance"],
        "LCO_CO2": optimal_LCOH_components["LCO_CO2"]
        }




    # Gas Boiler
    NGB = Boiler(
        name="Gas Boiler",
        specific_investment_tech=parameters["SpecificInvestment_NGBoiler"],
        efficiency=parameters["Eta_NGboiler"],
        o_and_m_fixed=parameters["O_and_M_fixed_NGBoiler"],
        o_and_m_var=parameters["O_and_M_var_NGBoiler"],
        capacity=yearly_heat_production / demand_hours,
        grid_cost_factor=parameters["grid_connection_factor_fuelBoiler"],
        lifetime=parameters["lifetime"]
    )
    
    # Calculate LCOH for natural gas boiler
    LCOH_NaturalGas, LCO_investment, LCO_grid, LCO_fuel, _, LCO_maintenance, LCO_CO2 = calculate_LCOH_for_technology(
        tech=NGB,
        yearly_heat_production=yearly_heat_production,
        fuel_price=parameters["Gas_price"],
        heat_source_price=0,  # No heat source for boilers
        specific_CO2_tax=CO2_tax,
        CO2_emission_factor=parameters["CO2_emission_factor"],
        interest=interest_rate,
        inflation=inflation_rate,
    )
    results["NGB"] = {
        "LCOH": LCOH_NaturalGas,
        "LCO_investment": LCO_investment,
        "LCO_grid": LCO_grid,
        "LCO_fuel": LCO_fuel,
        "LCO_heatsource": 0,  # No heat source cost for boilers
        "LCO_maintenance": LCO_maintenance,
        "LCO_CO2": LCO_CO2,
    }

    # Return all results
    return results

    
    
def determine_capacities_storage(tech_daily_operation_hours, yearly_heat_production, demand_hours,days_per_year):
    """Determines the capacities of the daily storage and heating technology when heating technology is operating used 24/7."""
    
    demand_hours_per_day = demand_hours/days_per_year
    daily_heat_demand = yearly_heat_production/days_per_year
    capacity_demand = yearly_heat_production/demand_hours
    
    capacity_tech = daily_heat_demand/ tech_daily_operation_hours
    capacity_min = min([capacity_tech, capacity_demand])
    
    if (demand_hours_per_day + tech_daily_operation_hours - 24) < 0:
        thermal_storage_hours = tech_daily_operation_hours# [h] storage capacity for the load shifting case.
        thermal_storage_capacity = capacity_tech*thermal_storage_hours
    else:
        thermal_storage_hours = (daily_heat_demand-(demand_hours_per_day + tech_daily_operation_hours - 24)*capacity_min)/capacity_min
        thermal_storage_capacity = capacity_min*thermal_storage_hours
    
    operating_hours = tech_daily_operation_hours*days_per_year
    
    return capacity_tech, thermal_storage_capacity, operating_hours
    

def calculate_factor_reduced_electricity_price(operating_hours_tech, linear_func_a, linear_func_b):        
    """Determines the relative reduction factor of the electricity price, based on assumption that electricity price is a linear function of ."""     
    factor_reduced_electricity_price_thermal_storage = linear_func_a*(operating_hours_tech/8760) + linear_func_b 
      
    return  factor_reduced_electricity_price_thermal_storage


def calculate_factor_reduced_electricity_price_electrolyzer(factor_reduced_hours, electrolyser_linear_func_a, electrolyser_linear_func_b):        
    """Determines the relative reduction factor of the electricity price, based on assumption that electricity price is a linear function of ."""     
    factor_reduced_electricity_price_electrolyzer = electrolyser_linear_func_a*factor_reduced_hours*100 + electrolyser_linear_func_b 
      
    return  factor_reduced_electricity_price_electrolyzer


###########################################################################################################################################################################



# Define input parameters for scenarios
Electricity_price = 130  # Example electricity price in €/MWh
DemandHours = 4000  # Hours per year
Heat_production_yearly = 36000  # MWh/year

# Call the LCOH_for_technologies function
results = LCOH_for_technologies(input_params, Electricity_price, DemandHours, Heat_production_yearly)

# Define the technologies to plot
technologies = [
    "HP1_free_source",  # HTHP 150°C with free heat source
    "HP2_free_source",  # HTHP 150°C + thermal storage, free heat source 
    "HP1",  # HTHP 150°C incl heat source cost
    "HP2",  # HTHP 150°C + thermal storage incl heat source cost
    "EB1",  # Electric boiler
    "EB2",  # Electric boiler + thermal storage
    "HB1",  # Hydrogen boiler
    "HB2",  # Hydrogen boiler + thermal storage
    "Biomass Boiler",  # Biomass boiler
]

# Extract LCOH components for the selected technologies
tp = {
    "Investment": [],
    "Electrical grid": [],
    "Fuel": [],
    "Heat source": [],
    "O&M": [],
    "CO2 tax": [],  # CO2 tax will not be included in the right-side legend
}
total_values = []  # Total LCOH for each technology

for tech in technologies:
    tech_results = results[tech]  # Get the results for the specific technology
    tp["Investment"].append(tech_results["LCO_investment"])
    tp["Electrical grid"].append(tech_results["LCO_grid"])
    tp["Fuel"].append(tech_results["LCO_fuel"])
    tp["Heat source"].append(tech_results["LCO_heatsource"])
    tp["O&M"].append(tech_results["LCO_maintenance"])
    tp["CO2 tax"].append(tech_results["LCO_CO2"])
    total_values.append(tech_results["LCOH"])



# Convert lists to numpy arrays for plotting
tp = {key: np.array(values) for key, values in tp.items()}
total_values = np.array(total_values)

# Get natural gas boiler results for background
LCOH_NG_w_tax = results["NGB"]["LCOH"]  # Natural gas with CO2 tax
LCOH_NG_wo_tax = LCOH_NG_w_tax - results["NGB"]["LCO_CO2"]  # Natural gas without CO2 tax

# Plot: Filtered technologies
width = 0.6  # Bar width
x = np.arange(len(technologies))  # X positions for technologies

fig, ax = plt.subplots(figsize=(10, 8))  

# Highlight natural gas boiler ranges with `axhspan`
ng_wo_tax = ax.axhspan(0, LCOH_NG_wo_tax, facecolor="grey", alpha=0.5, label="Natural Gas, w/o CO2 tax")
ng_w_tax = ax.axhspan(LCOH_NG_wo_tax, LCOH_NG_w_tax, facecolor="lightgrey", alpha=0.5, label="Natural Gas, w/ CO2 tax")

# Define colors for the LCOH components
cmap = mpl.colormaps['viridis']
colors = cmap(np.linspace(0.05, 0.9, 5))

# Plot stacked bars for selected technologies
bottom_primary = np.zeros(len(technologies))
bars = []
for (part, values), color in zip(tp.items(), colors):
    bar = ax.bar(x, values, width, label=part, bottom=bottom_primary, color=color, edgecolor="darkgrey")
    bars.append(bar)
    bottom_primary += values

# Add total LCOH labels above each bar
for i, total in enumerate(total_values):
    ax.text(i, total + 2, round(total), ha="center", color="black", fontsize=16)

# Configure plot
ax.grid(True, linestyle="--", linewidth=0.3, zorder=0)
ax.set_ylim(0, max(total_values) + 50)
ax.set_xticks(x)
ax.set_xticklabels(
    ["HTHP 150°C, free HS", "HTHP 150°C + termal storage, free HS","HTHP 150°C", "HTHP 150°C + thermal storage", "Electric boiler",  "Electric boiler + thermal storage", "Hydrogen boiler","Hydrogen boiler + inf. and free H2-storage", "Biomass boiler"],
    rotation=45,
    ha="right",
    fontsize=16,
)
plt.ylabel("LCOH [€/MWh]", fontsize=16)
plt.yticks(fontsize=14)
plt.title(f"Electricity: {Electricity_price} €/MWh, {DemandHours} h of heating demand", fontsize=16)

# Add separate legend for natural gas ranges
legend1 = ax.legend([ng_wo_tax, ng_w_tax], ["Natural Gas, w/o CO2 tax", "Natural Gas, w/ CO2 tax"], loc="upper left", fontsize=14)
ax.add_artist(legend1)  # Add the natural gas legend to the plot

# Add separate legend for LCOH components (excluding CO2 tax)
component_bars = bars[:]  # Exclude "CO2 tax" from the legend
component_labels = list(tp.keys())[:]  # Exclude "CO2 tax" from the labels
ax.legend(component_bars, component_labels, bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=14)

# Final layout adjustments
plt.tight_layout()
plt.show()



























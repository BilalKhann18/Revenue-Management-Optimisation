
# -*- coding: utf-8 -*-
"""

@author: Pricing Group 14
"""

import pandas as pd
from pyomo.environ import *

# Load the dataset
df = pd.read_excel('BuildMax_Rentals_Dataset_Updated.xlsx')
# Display the first few rows of the data
df.head()

# The total number of columns and rows
num_rows, num_columns = df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")

# extract values from dataset
df1 = df.iloc[:num_rows-3, 1:num_columns-3]
# Display the first few rows of the data
df1.head()

# Use sets to store unique equipment types and unique rental periods
equipment_types = set()  
rental_periods = set()
weeks=set(range(1,num_rows-2))

for column in df1.columns:
    # Split the column name by '-' and take the first part
    potential_equipment = column.split(' - ')[0] 
    
    # Add it to the set if it's not already there
    equipment_types.add(potential_equipment)

for column in df1.columns:
    parts = column.split(' - ')
    if len(parts) >= 2:
        potential_period = parts[1].split(' ')[0]
        
        # Check if the potential period is "Week" and skip if it is
        if potential_period.lower() == "week":  
            continue

        # Otherwise, check if it contains "week" and add to the set
        if 'week' in potential_period.lower():  
            rental_periods.add(potential_period)

# Remove irrelevant column names like 'Date', 'Week ending with', etc. 
# You might need to adjust this list based on your specific column names
irrelevant_columns = ['Date', 'Week ending with']  
equipment_types = equipment_types - set(irrelevant_columns)
irrelevant_periods = []  
rental_periods = rental_periods - set(irrelevant_periods)
rental_periods = sorted(list(rental_periods), key=lambda x: int(x.split('-')[0])) 

# Print the equipment types and the rental periods
print("Equipment Types:", equipment_types)
print("Rental Periods (weeks):", rental_periods)
print(f"weeks: {weeks}")

# initialise the model
model = ConcreteModel()

# Define decision variables
# Define three index sets, representing different equipment types, rental periods, weeks
model.x = Var(list(equipment_types), list(rental_periods), list(weeks), domain=NonNegativeIntegers) 

len(model.x)

# Define objective Function
def obj_rule(model):
    return sum(model.x[equipment, time_period, week] * df1[f'{equipment} - {time_period} Price per day (£)'].values[week-1] * int(time_period.split('-')[0]) * 7
    for equipment in list(equipment_types) 
    for time_period in list(rental_periods) 
    for week in list(weeks) 
)
model.obj = Objective(rule=obj_rule(model), sense=maximize)
model.obj.pprint()

# Define constraints
# Constraint: Demand constraint (rental acceptances cannot exceed demand)
def demand_rule(model, equipment, time_period, week):
    column_name = f'{equipment} - {time_period} Demand (units)'  
    return model.x[equipment, time_period, week] <= df1[column_name].values[week-1]     

model.demand_constraint = Constraint(list(equipment_types), list(rental_periods), list(weeks), rule=demand_rule)

# Print only the first few constraints
for index in list(model.demand_constraint.index_set())[:5]:  # Limit to first 5
    model.demand_constraint[index].pprint()

def inventory_rule(model, equipment, week):  
    # store the inventory of each equipment type in each week
    inventory = {}

    # Initialize the inventory for the first week
    for eq in list(equipment_types):
        inventory[(eq, 1)] = df1[f'{eq} - Start of Week Inventory'].values[0]

    # Calculate the inventory from week 2 to the current week
    for wk in range(2, week + 1):  # Iterate to the current week
        for eq in list(equipment_types):
            # Calculate the number of rentals from the previous week
            rented_out_previous_week = sum(model.x[eq, tp, wk - 1] for tp in list(rental_periods))  

            # Calculate the number of returns from the previous week
            returned_previous_week = 0
            for tp in list(rental_periods):
                rental_duration = int(tp.split('-')[0])
                previous_rental_week = wk - rental_duration  # The rental being returned started the previous week
                if previous_rental_week >= 1:
                    returned_previous_week += model.x[eq, tp, previous_rental_week]  

            # Calculate the inventory for the current week
            inventory[(eq, wk)] = inventory.get((eq, wk - 1), 0) - rented_out_previous_week + returned_previous_week

    # Calculate the total number of rentals for all rental cycles of the device in the current week
    total_rented_out_this_week = sum(model.x[equipment, tp, week] for tp in list(rental_periods))

    return total_rented_out_this_week <= inventory.get((equipment, week), 0)


model.inventory_constraint = Constraint(list(equipment_types), list(weeks), rule=inventory_rule)
# Print only the first few constraints
for index in list(model.inventory_constraint.index_set())[:5]:  # Limit to first 5
    model.inventory_constraint[index].pprint()
    
from pyomo.opt import SolverFactory

solver=SolverFactory('glpk')
results=solver.solve(model,tee=False)
if results.solver.termination_condition==TerminationCondition.optimal:
    model.solutions.load_from(results)
else:
    print('Solve failed.')

print('optimal revenue is:',model.obj())

    
# actual revenue from original dataset
actual_values = df.iloc[:9,66:]
print(actual_values)


actual_revenue = actual_values.iloc[0,1]

# total revenue improvement calculation
optimized_revenue = model.obj()
revenue_improvement = ((optimized_revenue - actual_revenue) / actual_revenue) * 100


# actual revenue by equipment type caluclation
equipment_revenue = {}

for equipment in list(equipment_types):
# Initialize the equipment's income
    total_revenue = 0

# Traverse all rental periods and weeks and calculate income
    for time_period in list(rental_periods):
        for week in list(weeks):
# Get the decision variable values ​​for the equipment, rental period, and number of weeks
            rental_count = model.x[equipment, time_period, week].value

# Get the decision variable values ​​for the equipment, rental period, and number of weeks
            if rental_count > 0:
                price_per_day = df1[f"{equipment} - {time_period} Price per day (£)"].values[week - 1]
                
                revenue = rental_count * price_per_day * int(time_period.split('-')[0]) * 7
                
                total_revenue += revenue

    equipment_revenue[equipment] = total_revenue
    

# revenue improvement by equipment type calculation
actual_equipment_values = actual_values.iloc[2:5,:]
actual_equipment_values = actual_equipment_values.T
actual_equipment_values.columns = actual_equipment_values.iloc[0]  
actual_equipment_values = actual_equipment_values[1:].reset_index(drop=True)  

# caculate revenue_improvement_by_equipment
revenue_improvement_by_equipment = {}  
equipment_list = list(equipment_types)

for eq in list(equipment_types):
    # Get the corresponding actual income
    eq_1 = eq[:-1] 
    actual_equipment_revenue = actual_equipment_values[f'Total revenue from {eq_1} = '].values[0]
    
    improvement = ((equipment_revenue[eq] - actual_equipment_revenue) / actual_equipment_revenue) * 100
    
    revenue_improvement_by_equipment[eq] = improvement


# ROI caculation
# Compute total ROI before and after optimization 
purchase_price = actual_values.iloc[6:9,:]
purchase_price = purchase_price.T
purchase_price.columns = purchase_price.iloc[0]  
purchase_price = purchase_price[1:].reset_index(drop=True)  

purchase_cost = {}
for eq in list(equipment_types):
    eq_1 = eq[:-1]
    purchase_cost[eq] = purchase_price[f"{eq_1} purchase price (£) = "].values[0] * df1[f"{eq} - Start of Week Inventory"].values[0]


purchase_total = sum(purchase_cost[eq] for eq in list(equipment_types))
# total ROI before
total_ROI_before = 100*(actual_revenue - purchase_total)/purchase_total


# totaL ROI after
total_ROI_after = 100*(optimized_revenue-purchase_total)/purchase_total


# Compute ROI for different equuipment before and after optimization (Weighted by Revenue Contribution)
# Compute ROI before optimization
equipment_ROI_before = {}
for eq in list(equipment_types):
    eq_1 = eq[:-1]
    equipment_ROI_before[eq] = 100*(actual_equipment_values[f'Total revenue from {eq_1} = '].values[0] - purchase_cost[eq])/purchase_cost[eq]


# Compute ROI after optimization 
equipment_ROI_after = {}
for eq in list(equipment_types):
    equipment_ROI_after[eq] = 100*(equipment_revenue[eq] - purchase_cost[eq])/purchase_cost[eq]


# add ROI weights  
revenue_weights = {}
total_revenue = sum(equipment_revenue.values())

for eq in equipment_types:
    revenue_weights[eq] = equipment_revenue[eq] / total_revenue

equipment_weighted_ROI_before = {}
equipment_weighted_ROI_after = {}

total_weighted_ROI_before = 0
total_weighted_ROI_after = 0

for eq in equipment_types:
    equipment_weighted_ROI_before[eq] = equipment_ROI_before[eq] * revenue_weights[eq]
    equipment_weighted_ROI_after[eq] = equipment_ROI_after[eq] * revenue_weights[eq]

# calculate utilization rate
# calculate utilization rate before optimization
weekly_utilization_rate = {week: {} for week in weeks}

for week in weeks:
    total_rented_units_week = sum([sum([df1[f'{eq} - {tp} Accepted (units)'].values[week-1] for tp in rental_periods])for eq in equipment_types])    
    total_max_inventory_week = sum(df1[f'{eq} - Start of Week Inventory'].values[week-1] for eq in equipment_types)
    weekly_utilization_rate[week]['total'] = (total_rented_units_week / total_max_inventory_week) * 100 if total_max_inventory_week > 0 else 0
    
    for eq in equipment_types:
        total_rented = sum(df1[f'{eq} - {tp} Accepted (units)'].values[week-1] for tp in rental_periods)
        max_inventory = df1[f'{eq} - Start of Week Inventory'].values[week-1]
        weekly_utilization_rate[week][eq] = (total_rented / max_inventory) * 100 if max_inventory > 0 else 0

average_utilization_rate = {}

# Calculate the average utilization of each device
for eq in equipment_types:
    utilization_rates = [weekly_utilization_rate[week][eq] for week in weeks]  
    avg_utilization = sum(utilization_rates) / len(weeks) if utilization_rates else 0  
    average_utilization_rate[eq] = avg_utilization

# calculate utilization rate after optimization
# Initialize the inventory for the first week
inventory = {}  
for eq in list(equipment_types):
    inventory[(eq, 1)] = df1[f'{eq} - Start of Week Inventory'].values[0]

# Calculate the inventory from week 2 to the current week
for wk in range(2, len(weeks)+1):  # Iterate to the current week
    for eq in list(equipment_types):
        # Calculate the number of rentals from the previous week
        rented_out_previous_week = sum(
        model.x[eq, tp, wk - 1].value for tp in list(rental_periods))  

        # Calculate the number of returns from the previous week
        returned_previous_week = 0
        for tp in list(rental_periods):
            rental_duration = int(tp.split('-')[0])
            previous_rental_week = wk - rental_duration  # The rental being returned started the previous week
            if previous_rental_week >= 1:
                returned_previous_week += model.x[eq, tp, previous_rental_week].value  

        # Calculate the inventory for the current week
        inventory[(eq, wk)] = inventory.get((eq, wk - 1), 0) - rented_out_previous_week + returned_previous_week


# Initialize dictionary for weekly utilization rate after optimization
weekly_utilization_rate_after = {week: {'total': 0} for week in weeks}

for week in weeks:
    total_rented_units_week_after = sum(
        sum(model.x[eq, tp, week].value for tp in rental_periods) for eq in equipment_types
    )
    total_max_inventory_week_after = sum(
        inventory.get((eq, week), 0) for eq in equipment_types  # Using inventory dictionary
    )

    # Calculate total utilization rate after optimization
    weekly_utilization_rate_after[week]['total'] = (
        (total_rented_units_week_after / total_max_inventory_week_after) * 100
        if total_max_inventory_week_after > 0 else 0
    )

    # Ensure each equipment type has an entry in the dictionary
    for eq in equipment_types:
        total_rented = sum(model.x[eq, tp, week].value for tp in rental_periods)
        max_inventory = inventory.get((eq, week), 0)  # Using optimized inventory

        # Set default to 0 to prevent KeyError
        weekly_utilization_rate_after[week][eq] = (
            (total_rented / max_inventory) * 100
            if max_inventory > 0 else 0
        )

# Compute Average Utilization Rate Before Optimization
average_utilization_rate_before = {
    eq: sum(weekly_utilization_rate[week].get(eq, 0) for week in weeks) / len(weeks)
    for eq in equipment_types
}
average_total_utilization_before = (
    sum(weekly_utilization_rate[week].get('total', 0) for week in weeks) / len(weeks)
)

# Compute Average Utilization Rate After Optimization
average_utilization_rate_after = {
    eq: sum(weekly_utilization_rate_after[week].get(eq, 0) for week in weeks) / len(weeks)
    for eq in equipment_types
}
average_total_utilization_after = (
    sum(weekly_utilization_rate_after[week].get('total', 0) for week in weeks) / len(weeks)
)

# Revenue per Unit (RPU) 

# Compute actual total units rented (pre-optimization)
actual_units_rented = {}

for eq in list(equipment_types):
    total_rented = 0
    for tp in list(rental_periods):
        for wk in list(weeks):
            column_name = f"{eq} - {tp} Accepted (units)"
            total_rented += df1[column_name].values[wk-1]  # Summing actual rented units

    actual_units_rented[eq] = total_rented

# Compute Revenue Per Unit before optimization
rpu_before = {eq: actual_equipment_values[f'Total revenue from {eq[:-1]} = '].values[0] / actual_units_rented[eq] 
              if actual_units_rented[eq] > 0 else 0
              for eq in equipment_types}

# Compute Revenue Per Unit after optimization
revenue_unit = {}
for eq in list(equipment_types):
    revenue_unit[eq] = equipment_revenue[eq]/sum(model.x[eq,tp,wk].value for tp in rental_periods for wk in weeks)


# results of Model
separator = "=" * 50  # Define a separator line for better readability

# 1. Optimal Revenue
print(separator)
print("OPTIMAL REVENUE")
print(separator)
print(f"Actual Revenue: £{actual_revenue:,.2f}")
print(f"Optimal Revenue: £{model.obj():,.2f}")
print(f"Revenue Difference: £{(model.obj() - actual_revenue):,.2f}\n")

# 2. Revenue Improvement
print(separator)
print("REVENUE IMPROVEMENT")
print(separator)
print(f"Total Revenue Improvement: {revenue_improvement:.2f}%\n")

print(f"{'Equipment':<20}{'Actual Revenue (£)':<20}{'Optimized Revenue (£)':<25}{'Improvement (%)'}")
print("-" * 80)

for eq in sorted(equipment_types):
    actual_eq_revenue = actual_equipment_values[f'Total revenue from {eq[:-1]} = '].values[0]
    optimized_eq_revenue = equipment_revenue[eq]
    improvement = revenue_improvement_by_equipment[eq]
    
    print(f"{eq:<20}{actual_eq_revenue:,.2f}{optimized_eq_revenue:>25,.2f}{improvement:>20.2f}%")
print()

# 3. ROI Comparison
print(separator)
print("RETURN ON INVESTMENT (ROI) COMPARISON")
print(separator)
print(f"Total ROI Before Optimization: {total_ROI_before:.2f}%")
print(f"Total ROI After Optimization:  {total_ROI_after:.2f}%\n")

for eq in sorted(equipment_types):
    print(f"  {eq}:")
    print(f"    ROI Before: {equipment_ROI_before[eq]:.2f}%")
    print(f"    ROI After:  {equipment_ROI_after[eq]:.2f}%")
print()

# 4. Weighted ROI by Equipment
print(separator)
print("WEIGHTED ROI BY EQUIPMENT")
print(separator)

for eq in sorted(equipment_types):
    print(f"  {eq}:")
    print(f"    Weighted ROI Before: {equipment_weighted_ROI_before[eq]:.2f}%")
    print(f"    Weighted ROI After:  {equipment_weighted_ROI_after[eq]:.2f}%")
print()

# 5. Average Utilization Rate Before & After Optimization
print(separator)
print("EQUIPMENT UTILIZATION RATE COMPARISON")
print(separator)

# Before Optimization
print("Average Utilization Rate Before Optimization:")
print(f"{'Total Utilization':<25}: {average_total_utilization_before:.2f}%")
for eq in sorted(equipment_types):
    print(f"{eq:<25}: {average_utilization_rate_before[eq]:.2f}%")

print("\n" + separator)

# After Optimization
print("Average Utilization Rate After Optimization:")
print(f"{'Total Utilization':<25}: {average_total_utilization_after:.2f}%")
for eq in sorted(equipment_types):
    print(f"{eq:<25}: {average_utilization_rate_after[eq]:.2f}%")

print(separator)

# 6. Revenue Per Unit (RPU)
print(separator)
print("REVENUE PER UNIT (RPU)")
print(separator)

print(f"{'Equipment':<20}{'RPU Before (£)':<20}{'RPU After (£)':<20}{'Improvement (%)'}")
print("-" * 80)

for eq in sorted(equipment_types):
    rpu_before_eq = rpu_before[eq]
    rpu_after_eq = revenue_unit[eq]
    improvement = ((rpu_after_eq - rpu_before_eq) / rpu_before_eq) * 100 if rpu_before_eq > 0 else 0

    print(f"{eq:<20}{rpu_before_eq:,.2f}{rpu_after_eq:>20,.2f}{improvement:>20.2f}%")


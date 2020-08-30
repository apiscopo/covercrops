import sqlite3
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np


class StdevFunc:

    """For use as an aggregate function in sqlite"""
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 0

    def step(self, value):
        try:
            # automatically convert text to float, like the rest of SQLite
            val = float(value) # if fails, skips this iteration, which also ignores nulls
            tM = self.M
            self.k += 1
            self.M += ((val - tM) / self.k)
            self.S += ((val - tM) * (val - self.M))
        except:
            pass

    def finalize(self):
        if self.k <= 1: # avoid division by zero
            return None
        else:
            return math.sqrt(self.S / (self.k-1))

"""Establish connection to sqlite"""
conn = sqlite3.connect('database.db')
print("connected:", conn)
conn.create_aggregate("STDEV", 1, StdevFunc)  # since STDEV not available in sqlite
cursor = conn.cursor()

"""Create sql table for importing the case study csv"""
create_table = """CREATE TABLE IF NOT EXISTS crops (
                    field_name text,
                    field_acreage text,
                    state_name text,
                    county text,
                    pair_id integer,
                    dca_field text,
                    core_num integer,
                    top_depth_cm float,
                    bottom_depth_cm float,
                    carbon_pct float,
                    bulk_density float,
                    clay_pct float, 
                    cover_2017 str,
                    cover_2018 str, 
                    cover_2019 str,
                    cl_2017 str,
                    cdl_2018 str, 
                    cdl_2019 str);"""

cursor.execute(create_table)
conn.commit()

df = pd.read_csv("indigo_fieldex_case_study.csv")
df.to_sql('crops', conn, if_exists='replace', index=False)

"""Summary statistics: Number of states, fields, etc."""
query_num_spatial = "SELECT COUNT(DISTINCT state), COUNT(DISTINCT county), COUNT(DISTINCT field_name) FROM crops;"
cursor.execute(query_num_spatial)
results_spatial = cursor.fetchall()
print("\nnum states, counties, fields:")
for row in results_spatial:
    print(row)

"""Summary statistics: By state, # of counties, # of fields, and field acreage"""
query_state_acreage = "SELECT state, COUNT(DISTINCT county) AS num_counties, " \
                      "COUNT(DISTINCT dca_field_id) AS num_fields, SUM(acreage) AS tot_acres " \
                      "FROM crops " \
                      "GROUP BY 1 " \
                      "ORDER BY 1;"
cursor.execute(query_state_acreage)
results_state_acreage = cursor.fetchall()
print("\nby state: # counties, # fields, acreage")
for row in results_state_acreage:
    print(row)

"""Find list of cover crops for each year"""
str_list_cover = ['cover_2017', 'cover_2018', 'cover_2019']
for i in range(len(str_list_cover)):
    query_cover_crops = "SELECT DISTINCT " + str_list_cover[i] + " FROM crops;"
    cursor.execute(query_cover_crops)
    results_cover_crops = cursor.fetchall()
    # print("\nall " + str_list_cover[i] + ":")
    # for row in results_cover_crops:
    #     print(row)

"""Find list of cash crops for each year"""
str_list_cash = ['cdl_2017', 'cdl_2018', 'cdl_2019']
for i in range(len(str_list_cash)):
    query_cash_crops = "SELECT DISTINCT " + str_list_cash[i] + " FROM crops;"
    cursor.execute(query_cash_crops)
    results_cash_crops = cursor.fetchall()
    # print("\nall " + str_list_cash[i] + ":")
    # for row in results_cash_crops:
    #     print(row)

"""Determine most prevalent (by acreage) cover crop and cash crop by state"""
str_list = str_list_cover + str_list_cash
for i in range(len(str_list)):
    query_crop_prevalence = "WITH synop AS ( SELECT state, " + str_list[i] + " AS cover, SUM(acreage) AS acres " \
                            "FROM crops " \
                            "GROUP BY state, cover ) " \
                            "SELECT state, cover, MAX(acres) FROM synop GROUP BY state;"
    cursor.execute(query_crop_prevalence)
    results_crop_prev = cursor.fetchall()
    # print("\nmax " + str_list[i] + ":")
    # for row in results_crop_prev:
    #     print(row)

# note: in results from query above, None for AR, IN, KY, MS, and TX = not reported, 'None' for OK = no cover crops

"""Determine proportion of each crop type in 2019 - Cover crops (Pie Chart)"""
query_crop_pct = "SELECT DISTINCT cover_2019, SUM(acreage) FROM crops GROUP BY 1;"
cursor.execute(query_crop_pct)
results_cover_pct = cursor.fetchall()
print("\nacreage by cover crop:")
for row in results_cover_pct:
    print(row)
df_cover_pct = pd.DataFrame(results_cover_pct, columns=['Crop Type', ' '])
df_cover_pct = df_cover_pct.fillna("Not Reported")
df_cover_pct = df_cover_pct.set_index('Crop Type')
pie_cover = df_cover_pct.plot.pie(subplots=True, cmap="BuPu")
plt.legend(bbox_to_anchor=(0.5, 0.5))

"""Determine proportion of each crop type in 2019 - Cash crops (Pie chart)"""
query_cash_pct = "SELECT DISTINCT cdl_2019, SUM(acreage) FROM crops GROUP BY 1;"
cursor.execute(query_cash_pct)
results_cash_pct = cursor.fetchall()
print("\nacreage by cash crop:")
for row in results_cash_pct:
    print(row)
df_cash_pct = pd.DataFrame(results_cash_pct, columns=['Crop Type', ' '])
df_cash_pct = df_cash_pct.fillna("Not Reported")
df_cash_pct = df_cash_pct.set_index('Crop Type')
pie_cash = df_cash_pct.plot.pie(subplots=True, cmap="YlOrRd")
plt.legend(bbox_to_anchor=(0.5, 0.5))
plt.show()

"""Find how many samples have both carbon_pct (g carbon / g soil) and bulk density (g soil / cm3 soil)"""
query_pct_pb = "SELECT * FROM crops " \
               "WHERE carbon_pct IS NOT NULL AND bulk_density IS NOT NULL;"
cursor.execute(query_pct_pb)
results_pct_pb = cursor.fetchall()
num_pct_pb = len(results_pct_pb)
print("\nsamples w both pct and pb measured:", num_pct_pb)
for row in results_pct_pb:
    print(row)

"""Calculate carbon stock for all fields with % carbon and bulk density data"""
query_calc_stock = "WITH carbon_stock AS ( " \
                                           "SELECT * FROM crops " \
                                           "WHERE carbon_pct IS NOT NULL AND bulk_density IS NOT NULL ) " \
                   "SELECT dca_field_id, field_name, state, county, CAST(pair_id AS int), carbon_pct, bulk_density, " \
                   "AVG(CAST(carbon_pct / 100 AS float) * CAST(bulk_density AS float)) * 607.03 AS avg_carbon_stock " \
                   "FROM carbon_stock WHERE bottom_depth_cm = 30 and state = 'OH' " \
                   "GROUP BY dca_field_id ORDER BY state;"

# 607.03 in query above represents conversion factor from: X g / cm3 * 15 cm depth * (10,000 cm2 / 1 m2)
# * (4046.86 m2 / 1 acre) * (1 kg / 1000 g) * (1 ton / 907.185 kg) = 607.03

cursor.execute(query_calc_stock)
results_calc_stock = cursor.fetchall()
print("\ncarbon stock calcs at (_-_cm): field, field_name, state, county, pair_id, carbon_pct, bulk_density, "
      "avg_carbon_stock")
for row in results_calc_stock:
    print(row)
print("count:", len(results_calc_stock))
df_stock = pd.DataFrame(results_calc_stock, columns=['field', 'field_name', 'state', 'county', 'pair_id', 'carbon_pct',
                                                     'bulk_density', 'avg_carbon_stock'])
df_stock.to_csv("stock_30.csv")
# plot below to check for relationship between bulk density and % carbon (typically inversely related)
fig = plt.figure()
ax = fig.add_subplot(221)
plt.scatter(df_stock['bulk_density'], df_stock['carbon_pct'], s=50)
ax.set_xlabel('bulk density (g/cm3)', fontsize=13)
ax.set_ylabel('% carbon in soil', fontsize=13)

"""Determine mean and std deviation of % carbon for fields with crop types reported, and figure out which paired 
fields to compare"""
query_partner = "WITH fields_unique AS ( SELECT DISTINCT dca_field_id AS field, CAST(pair_id AS int) as pairs, " \
                "bottom_depth_cm, AVG(carbon_pct), STDEV(carbon_pct), COUNT(carbon_pct) AS sample_count, " \
                "cover_2017, cover_2018, cover_2019, state " \
                "FROM crops " \
                "WHERE cover_2017 OR cover_2018 OR cover_2019 OR cdl_2017 OR cdl_2018 OR cdl_2019 IS NOT NULL " \
                "GROUP BY field, bottom_depth_cm ORDER BY state, pairs ) " \
                "SELECT * FROM fields_unique WHERE bottom_depth_cm = 15 OR bottom_depth_cm = 30;"

cursor.execute(query_partner)
results_partner = cursor.fetchall()
print("\nfield, pair_id, bottom_depth, avg_pct_carbon, stdev_pct_carbon, sample_size, crop types:")
for row in results_partner:
    print(row)
print("length", len(results_partner))
df = pd.DataFrame(results_partner, columns=['field', 'pair_id', 'bottom_depth', 'mean_pct_carbon',
                                            'stdev_pct_carbon', 'sample_size', 'cover_2017', 'cover_2018',
                                            'cover_2019', 'state'])

df.pair_id = df.pair_id.replace(np.nan, 0.0)
df.pair_id = df.pair_id.astype(int)
df_15 = df[df['bottom_depth'] == 15]
df_30 = df[df['bottom_depth'] == 30]

whichpairs = []
ixs = 2  # index spacing (since indices of df_15 are by 2)
pd.set_option('mode.chained_assignment', None)

for i in np.arange(0, df_15.shape[0]-1):

    # if (below) pair ids are equal ...
    if df_15.pair_id.iloc[i] == df_15.pair_id.iloc[i+1]:

        # if (below) 'always uses cover crops' is in 1st row of the pair and 'sometimes or never uses' cover crops is
        # in the 2nd row of the pair, then append to "whichpairs" as a set of paired fields
        if (df_15.cover_2017.iloc[i] != 'None' and df_15.cover_2018.iloc[i] != 'None' and df_15.cover_2019.iloc[i] !=
            'None') and (df_15.cover_2017.iloc[i+1] == 'None' or df_15.cover_2018.iloc[i+1] == 'None' or
                         df_15.cover_2019.iloc[i+1] == 'None'):

            # omit pairs with field_id = 0 bc faulty id
            if df_15.pair_id.iloc[i] != 0:
                whichpairs.append(df_15.pair_id.iloc[i])

        # if 'sometimes or never uses cover crops' is in the 1st row of the pair and 'always' is in 2nd row then
        # append pair to "whichpairs"
        elif (df_15.cover_2017.iloc[i] == 'None' or df_15.cover_2018.iloc[i] == 'None' or df_15.cover_2019.iloc[i] ==
              'None') and (df_15.cover_2017.iloc[i+1] != 'None' and df_15.cover_2018.iloc[i+1] != 'None' and
                           df_15.cover_2018.iloc[i+1] != 'None'):

            # omit pairs with field_id = 0 bc faulty id
            if df_15.pair_id.iloc[i] != 0:
                whichpairs.append(df_15.pair_id.iloc[i])

                # switch rows to have "always cover crops" first
                toprow_15 = df_15.loc[i * ixs, :]
                toprow_15.is_copy = False
                df_15.loc[i * ixs] = df_15.loc[i * ixs + ixs]
                df_15.loc[i * ixs + ixs] = toprow_15

                toprow_30 = df_30.loc[1 + i * ixs, :]
                toprow_30.is_copy = False
                df_30.loc[1 + i * ixs] = df_30.loc[1 + i * ixs + ixs]
                df_30.loc[1 + i * ixs + ixs] = toprow_30



print("\nset of paired fields where one field in the set always has cover crops and the other field sometimes or never "
      "has cover crops:", whichpairs)

n_fields = df_15.shape[0]
x = np.arange(1, n_fields + 1)

fig = plt.figure()
ax = fig.add_subplot(111)
data_15 = plt.errorbar(x, df_15['mean_pct_carbon'].to_list(), df_15['stdev_pct_carbon'].to_list(), fmt='^k', lw=1)
data_30 = plt.errorbar(x, df_30['mean_pct_carbon'].to_list(), df_30['stdev_pct_carbon'].to_list(), fmt='^b', lw=1)
data_15.set_label('sample depth =  0-15 cm')
data_30.set_label('sample depth = 15-30 cm')
ax.set_xticks(x)
empty_string_labels = ['']*len(x)
ax.set_xticklabels(empty_string_labels)
ax.set_xlabel('individual fields with known crop types (42 fields)', fontsize=12)
ax.set_ylabel('percent carbon in soil (%)', fontsize=12)
ax.legend(fontsize=11)
plt.savefig('box_whiskers.svg', format='svg', dpi=1200)
plt.show()

"""Determine # of acres planted with either cover or cash crop for each year, and the # acres for which crop is 
not recorded """
query_planted = "WITH planted AS (SELECT state, acreage," \
                    "CASE WHEN cover_2017 IS NULL AND cdl_2017 IS NULL " \
                    "THEN acreage " \
                    "END AS unk_acres_2017, " \
                    "CASE WHEN cover_2017 IS 'None' AND cdl_2017 IS 'Fallow/Idle Cropland' THEN -acreage " \
                    "WHEN cover_2017 IS NULL AND cdl_2017 IS NULL THEN -acreage " \
                    "ELSE 0.0 " \
                    "END AS neg_acres_2017, " \
                    "CASE WHEN cover_2018 IS NULL AND cdl_2018 IS NULL " \
                    "THEN acreage " \
                    "END AS unk_acres_2018, " \
                    "CASE WHEN cover_2018 IS 'None' AND cdl_2018 IS 'Fallow/Idle Cropland' THEN -acreage " \
                    "WHEN cover_2018 IS NULL AND cdl_2018 IS NULL THEN -acreage " \
                    "ELSE 0.0 " \
                    "END AS neg_acres_2018, " \
                    "CASE WHEN cover_2019 IS NULL AND cdl_2019 IS NULL " \
                    "THEN acreage " \
                    "END AS unk_acres_2019, " \
                    "CASE WHEN cover_2019 IS 'None' AND cdl_2019 IS 'Fallow/Idle Cropland' THEN -acreage " \
                    "WHEN cover_2019 IS NULL AND cdl_2019 IS NULL THEN -acreage " \
                    "ELSE 0.0 " \
                    "END AS neg_acres_2019 " \
                    "FROM crops) " \
                "SELECT state, SUM(acreage), SUM(unk_acres_2017), SUM(acreage) + SUM(neg_acres_2017), SUM(unk_acres_" \
                "2018), SUM(acreage) + SUM(neg_acres_2018), SUM(unk_acres_2019), SUM(acreage) + SUM(neg_acres_2019) " \
                "FROM planted " \
                "GROUP BY state;"

cursor.execute(query_planted)
results_planted = cursor.fetchall()
print("\nstate, acreage, unk_acres_2017, p_acres_2017, unk_acres_2018, p_acres_2018, unk_acres_2019, p_acres_2019")
for row in results_planted:
    print(row)

df_planted = pd.DataFrame(results_planted, columns=['states', 'acreage', 'unk_acres_2017', 'p_acres_2017',
                                                    'unk_acres_2018', 'p_acres_2018', 'unk_acres_2019', 'p_acres_2019'])
fig = plt.figure()
ax = fig.add_subplot(311)
integers = np.arange(df_planted.shape[0])
y1 = df_planted['p_acres_2019']
y2 = df_planted['unk_acres_2019'].fillna(0.0)
print("total acres unknown", sum(y2))
width = 0.35
plt.bar(integers, y2/1000, width, label='Crop Type not reported (or land not planted)', color='tab:orange')
plt.bar(integers + width, y1/1000, width, label='Planted Land of known Crop Type', color='tab:blue')
plt.ylabel('Thousands of Acres')
plt.xticks(integers + width / 2, (df_planted['states']))
plt.legend(loc='best')
plt.show()

conn.close()



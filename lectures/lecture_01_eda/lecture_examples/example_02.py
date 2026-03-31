# /// script
# source-notebook = "example_02.ipynb"
# generated-by = "tools/sync_lecture_examples.py"
# ///

# %% [markdown]
# # Exploratory Data Analysis - Demos [5-6]

# %% [markdown]
# ## Connect to google drive and load the dataset

# %%
# connect to google drive
# NOTE: Colab-only import commented for local script use: from google.colab import drive
drive.mount('/content/drive')

# %%
import pandas as pd
import numpy as np
import scipy
from collections import Counter
from pathlib import Path

# %%
# paths
data_path = Path('/content/drive/MyDrive/courses/ml_basics/datasets/Ames housing/') # replace with your path to the dataset!
fields_description_path = data_path/'data_description.txt'
dataset_path = data_path/'AmesHousing.csv'

# get fields description
with open(fields_description_path, 'r') as f:
    fields_description = f.read()

# %%
print(fields_description)

# %%
# get the dataset
df = pd.read_csv(dataset_path, na_values='NA', index_col='Order')

# %%
df.head(10)

# %%
# show eleents of the first row as ditionary
df.iloc[0].to_dict()

# %% [markdown]
# # **DEMO EDA AUTOMATION**
# 
# 
# ---

# %% [markdown]
# # Automation in EDA

# %% [markdown]
# ## ydata_profiling (ex. pandas_profiling)

# %%
# NOTE: notebook magic commented for local script use: ! pip install -U ydata-profiling

# %%
import sys
# NOTE: notebook magic commented for local script use: !{sys.executable} -m pip install -U ydata-profiling[notebook]
# NOTE: notebook magic commented for local script use: !pip install jupyter-contrib-nbextensions

# %%
# NOTE: notebook magic commented for local script use: ! jupyter nbextension enable --py widgetsnbextension

# %%
from ydata_profiling import ProfileReport

# Load the AmesHousing dataset (replace with your dataset path)
df = pd.read_csv(dataset_path, na_values='NA', index_col='Order')
df_sample = df.sample(frac=0.1)  # This would sample 10% of the rows

# Generate the profile report
profile = ProfileReport(df_sample, title="AmesHousing Data Profile", explorative=True)

# To view the report in Jupyter Notebook or export to HTML
profile.to_notebook_iframe()  # To display the report inside a notebook
profile.to_file("AmesHousing_Report.html")  # To save the report as an HTML file

# %% [markdown]
# ## SweetViz

# %%
# NOTE: notebook magic commented for local script use: ! pip install sweetviz

# %%
import sweetviz as sv
import pandas as pd

# Generate a SweetViz report
df = pd.read_csv(dataset_path, na_values='NA', index_col='Order')
df_sample = df.sample(frac=0.2)  # This would sample 20% of the rows
report = sv.analyze(df_sample)

# Display the report in a notebook or export to an HTML file
report.show_html('SweetViz_Report.html')  # Saves the report as an HTML file
report.show_notebook()

# %% [markdown]
# ## D-Tale

# %%
# NOTE: notebook magic commented for local script use: !pip install -U dtale

# %%
import pandas as pd
import dtale
import dtale.app as dtale_app

df = pd.read_csv(dataset_path, na_values='NA', index_col='Order')
df_sample = df.sample(frac=0.2)  # This would sample 20% of the rows

dtale_app.USE_COLAB = True
dtale.show(df_sample, notebook=True)

# %%
pass

# %% [markdown]
# # **DEMO VISUALIZATION TOOLS**
# 
# 
# ---
# 

# %% [markdown]
# # Visualization Tools:

# %% [markdown]
# ## ggplot (Python)

# %%
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_point, facet_wrap, scale_color_gradient, labs, theme_minimal


# Strip whitespace from column names if necessary
df.columns = df.columns.str.strip()

# Remove rows where 'YearBuilt' is missing to avoid facetting errors
df_cleaned = df.dropna(subset=['Year Built'])

# Take a subset of the data for faster rendering if needed (e.g., 1000 samples)
df_subset = df_cleaned.sample(1000, random_state=42)

# Create the plot
p = (ggplot(df_subset, aes(x='Lot Area', y='SalePrice', size='Gr Liv Area'))
     + geom_point(alpha=0.6)  # Add points with transparency
     + scale_color_gradient(low="blue", high="red")  # Set color gradient based on 'OverallQual'
     + labs(title="Sale Price vs Lot Area", x="Lot Area (sq ft)", y="Sale Price ($)",
            size="Ground Living Area", color="Overall Quality")  # Add labels and title
     + theme_minimal()  # Use a minimal theme
)

# Show the plot using plot.show() instead of print(p)
p.show()

# %% [markdown]
# ## Plotly

# %%
import plotly.express as px
import pandas as pd

# Scatter plot of Sale Price vs Number of Garage Cars
fig = px.scatter(df, x="Garage Cars", y="SalePrice",
                 title="Scatter Plot: Sale Price vs Number of Garage Cars")
fig.show()

# %%
df = pd.read_csv(dataset_path, na_values='NA', index_col='Order')
df.columns

# %%
# Box plot to visualize distribution of Sale Price by number of garage cars
fig = px.box(df, x="Garage Cars", y="SalePrice",
             title="Box Plot: Sale Price Distribution by Garage Cars")
fig.show()

# %%
# Sample DataFrame
df1 = pd.DataFrame({
    "Neighborhood": ["A", "B", "C", "D"],
    "Average Sale Price": [250000, 300000, 350000, 400000],
    "Std Dev": [20000, 25000, 30000, 35000]
})

# Bar plot with hover text for standard deviation
fig = px.bar(df1, x="Neighborhood", y="Average Sale Price",
             title="Bar Plot: Average Sale Price by Neighborhood",
             hover_data=["Std Dev"])
fig.show()

# %%
import plotly.express as px
import pandas as pd

# Sample DataFrame with 3D data
df1 = pd.DataFrame({
    "Garage Cars": [1, 2, 3, 1, 2, 2, 3, 3],
    "Lot Area": [10000, 15000, 20000, 9000, 14000, 16000, 25000, 22000],
    "Sale Price": [200000, 350000, 400000, 210000, 360000, 340000, 420000, 410000]
})

# 3D scatter plot
fig = px.scatter_3d(df1, x="Garage Cars", y="Lot Area", z="Sale Price",
                    title="3D Scatter Plot: Garage Cars vs Lot Area vs Sale Price")
fig.show()

# %%
import plotly.express as px
import pandas as pd

# Sample DataFrame
df1 = pd.DataFrame({
    "Neighborhood": ["A", "B", "C", "D"],
    "Sale Price": [250000, 300000, 350000, 400000]
})

# Pie chart
fig = px.pie(df1, names="Neighborhood", values="Sale Price",
             title="Pie Chart: Sale Price Distribution by Neighborhood")
fig.show()

# %% [markdown]
# ## Bokeh

# %%
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
import pandas as pd

# Display Bokeh plots in the notebook
output_notebook()

# Sample DataFrame
df1 = pd.DataFrame({
    "Garage Cars": [1, 2, 3, 1, 2, 2, 3, 3],
    "Sale Price": [200000, 350000, 400000, 210000, 360000, 340000, 420000, 410000]
})

# Create a scatter plot
p = figure(title="Scatter Plot: Sale Price vs Number of Garage Cars",
           x_axis_label='Number of Garage Cars', y_axis_label='Sale Price ($)')
p.scatter(df1["Garage Cars"], df1["Sale Price"], size=10, color="navy", alpha=0.6)

# Show the plot
show(p)

# %%
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

# Ensure output in notebook (for Jupyter Notebooks)
output_notebook()

# Load your dataset (replace with the actual path to your dataset)
# df = pd.read_csv('housing_data.csv')  # Uncomment and use your actual dataset

# Strip whitespace from column names if necessary
df.columns = df.columns.str.strip()

# Sample a smaller dataset for faster rendering (optional)
df_subset = df.sample(200, random_state=42)  # Adjust the number as needed

# Set x, y, and sizes from the dataset
x = df_subset['Lot Area']
y = df_subset['SalePrice']
sizes = df_subset['Gr Liv Area'] / 100  # Scale down for better visualization

# Create a figure object
p = figure(title="Housing Data: Lot Area vs Sale Price",
           x_axis_label='Lot Area (sq ft)',
           y_axis_label='Sale Price ($)',
           tools="pan, wheel_zoom, box_zoom, reset, save")

# Add circle glyphs to the figure, with sizes based on 'Gr Liv Area'
p.circle(x, y, size=sizes, color="navy", alpha=0.6)

# Display the plot
show(p)

# %%
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
import pandas as pd

# Sample DataFrame
df1 = pd.DataFrame({
    "Neighborhood": ["A", "B", "C", "D"],
    "Average Sale Price": [250000, 300000, 350000, 400000],
    "Std Dev": [20000, 25000, 30000, 35000]
})

# Create a bar plot
p = figure(x_range=df1["Neighborhood"], title="Bar Plot: Average Sale Price by Neighborhood",
           y_axis_label="Average Sale Price ($)")

# Add bars
p.vbar(x=df1["Neighborhood"], top=df1["Average Sale Price"], width=0.9, color="skyblue")

# Add hover tool
hover = HoverTool()
hover.tooltips = [("Neighborhood", "@x"), ("Average Sale Price", "@top"), ("Std Dev", "@y")]
p.add_tools(hover)

# Show the plot
show(p)

# %% [markdown]
# **[!]** Bokeh does not support 3D Scatter Plot or Pie charts

# %% [markdown]
# ## Altair

# %%
import pandas as pd
import altair as alt
import warnings
# Suppress the specific FutureWarning from pandas/altair
warnings.filterwarnings('ignore', category=FutureWarning)

# Load your dataset (replace with the actual path to your dataset)
# df = pd.read_csv('data.csv')  # Uncomment and use your actual dataset

# Strip whitespace from column names if necessary
df.columns = df.columns.str.strip()

# Take a subset of the data for faster rendering (optional)
df_subset = df.sample(500, random_state=42)  # Adjust the sample size as needed

# Create an Altair chart with increased width and height
chart = alt.Chart(df_subset).mark_circle(size=60).encode(
    x=alt.X('Lot Area:Q', title='Lot Area (sq ft)'),
    y=alt.Y('SalePrice:Q', title='Sale Price ($)'),
    size=alt.Size('Gr Liv Area:Q', title='Ground Living Area'),
    color=alt.Color('Overall Qual:N', title='Overall Quality'),  # Color by Overall Quality if available
    tooltip=['Lot Area', 'SalePrice', 'Gr Liv Area', 'Overall Qual'],  # Add more fields as needed
).properties(
    width=800,  # Adjust the width as desired
    height=600  # Adjust the height as desired
).interactive()  # Enable interactivity like zooming and panning

# Display the chart
chart

# %%
import altair as alt
import pandas as pd
import warnings
# Suppress the specific FutureWarning from pandas/altair
warnings.filterwarnings('ignore', category=FutureWarning)

# Create a slider for bin size using `alt.param`
slider = alt.binding_range(min=1, max=100, step=5, name='Bin Size: ')
bin_size = 50

# Create a binning transformation with a dynamic bin size
hist = alt.Chart(df).mark_bar().encode(
    x=alt.X('binned_SalePrice:Q', title="Sale Price ($)"),
    y=alt.Y('count()', title='Count')
).transform_bin(
    'binned_SalePrice', field='SalePrice', bin=alt.Bin(maxbins=bin_size)
).properties(
    width=600,
    height=400
).interactive()

# Display the chart
hist

# %%
pass

import streamlit as st
import pandas as pd
import numpy as np
import scipy
import pgeocode
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Wedding Day Weather')

#load data
@st.cache(allow_output_mutation=True)
def load_data():
    dly1 = pd.read_csv('./data/dly1.csv')
    dly2 = pd.read_csv('./data/dly2.csv')
    dly3 = pd.read_csv('./data/dly3.csv')
    dly = dly1.append(dly2).append(dly3)
    dly = dly.iloc[:,2:]
    inventory = pd.read_csv('./data/dly_combined_inventory.csv')
    return dly, inventory

dly, inventory = load_data()

#function to get lat, long, and location info from input zip code
def get_location_info(user_input_zip):
    nomi = pgeocode.Nominatim('us')
    zip_code_info = nomi.query_postal_code(user_input_zip)
    lat = zip_code_info['latitude']
    long = zip_code_info['longitude']
    place = zip_code_info['place_name']
    return lat, long, place

#function to get nearest weather station
def get_nearest_station(inventory,lat,long):
    def haversine_np(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        All args must be of equal length.    
        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km
    inventory['nearest_station'] = inventory.apply(lambda x: haversine_np(x['long'], x['lat'],long,lat), axis=1)
    return inventory.loc[inventory['nearest_station'].idxmin(),'station']

### function to add score to df for chosen location and temp range
def get_weather_score(dly, nearest_station,low_temp,high_temp):
    local_dly = dly[dly['GHCN_ID']==nearest_station]
    local_dly['temp_prob_score'] = norm(loc=local_dly['DLY_TMAX_NORMAL'], scale=local_dly['DLY-TMAX-STDDEV']).cdf(high_temp) - norm(loc=local_dly['DLY_TMAX_NORMAL'], scale=local_dly['DLY-TMAX-STDDEV']).cdf(low_temp) 
    local_dly['prob_score'] = local_dly['temp_prob_score']*(1-(local_dly['DLY-PRCP-PCTALL-GE005MM']/100))
    best_day_index = local_dly['prob_score'].idxmax()
    return local_dly, best_day_index

## display score details for selected location
def show_score(local_dly, place, best_day_index,low_temp,high_temp):
    #new format for displaying score 7-23-22
    months = ['January','February','March', 'April', 'May', 'June','July','August','September','October','November','December']
    best_month=local_dly.loc[best_day_index,'month']
    best_date = local_dly.loc[best_day_index,'day']
    st.metric('Your Perfect Wedding Weather Date in %s is...' % (place),  "%s %s" % (months[best_month-1], best_date))
    col1, col2, col3 = st.columns(3)
    col1.metric("Weather Score", "%.0f" % (local_dly.loc[best_day_index,'prob_score']*100), delta='Good')#, help='The weather score is the percent probablity of a high temp in your desired range and no precipitation on this date')
    col2.metric("Expected High Temperature", "%.0f °F" % (local_dly.loc[best_day_index,'DLY_TMAX_NORMAL']), delta='+/- %.0f °F' % (local_dly.loc[best_day_index,'DLY-TMAX-STDDEV']))
    col3.metric("Chance of Precipitation", "%.0f%%" % (local_dly.loc[best_day_index,'DLY-PRCP-PCTALL-GE005MM']))

    #old format
    #st.write('The best date for a wedding in %s is %s' % (place, local_dly.loc[best_day_index,'mm-dd']))
    #st.write('There is a %.1f percent probablity of a high temp in your desired range and no precipitation on this date' % (local_dly.loc[best_day_index,'prob_score']*100))
    #st.write('There is a %.1f percent probablity of a high temperate in your desired range on this date' % (local_dly.loc[best_day_index,'temp_prob_score']*100))
    #st.write('There is a %.1f percent probablity of rain on this date' % (local_dly.loc[best_day_index,'DLY-PRCP-PCTALL-GE005MM']))

    #display line chart
def weather_score_chart(local_dly):
    st.subheader('Weather Score')
    st.write('The weather score is the probability that the high temp will be in your desired range and that there will be no precipitation on this date')
    local_plot = local_dly[local_dly['day']==1]
    months = ['January','February','March', 'April', 'May', 'June','July','August','September','October','November','December']
    #fig = plt.figure()
    #sns.lineplot(x=local_plot['mm-dd'],y=local_plot['prob_score']*100)
    #sns.despine()
    #plt.legend()
    #st.write(fig)

    #simple plotly line plot
    fig = px.line(local_plot, x=months, y="prob_score")
    fig.update_layout(
        margin=dict(l=1,r=1,b=1,t=1),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(title_text='Month')
    fig.update_yaxes(title_text='Weather Score')
    st.write(fig)


def temp_chart(local_dly):
    st.subheader('Expected Temperature')
    local_plot = local_dly[local_dly['day']==1]
    #fig = plt.figure()
    #sns.lineplot(x=local_plot['mm-dd'],y=local_plot['DLY_TMAX_NORMAL'])
    #sns.lineplot(x=local_plot['mm-dd'],y=local_plot['DLY_TMIN_NORMAL'])
    #sns.despine()
    #plt.legend()
    #st.write(fig)

    #fancy plotly line plot
    months = ['January','February','March', 'April', 'May', 'June','July','August','September','October','November','December']

    fig_dly_temp=go.Figure([
        go.Scatter(x=months, 
                y=local_plot['DLY_TMAX_NORMAL'], 
                mode='lines',
                visible=True,
                line_color='red',
                name='High Temp'
                ),
        go.Scatter(x=months,
                y=local_plot['DLY_TMAX_NORMAL']+local_plot['DLY-TMAX-STDDEV'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
                name='Upper Bound'
                ),
        go.Scatter(x=months,
                y=local_plot['DLY_TMAX_NORMAL']-local_plot['DLY-TMAX-STDDEV'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(70, 70, 70, 0.15)',
                fill='tonexty',
                showlegend=False,
                name='Lower Bound'
                ),
        go.Scatter(x=months, 
                y=local_plot['DLY_TMIN_NORMAL'], 
                mode='lines',
                visible=True, 
                line_color='blue',
                name='Low Temp'
                ),
        go.Scatter(x=months,
                y=local_plot['DLY_TMIN_NORMAL']+local_plot['DLY-TMIN-STDDEV'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
                name='Upper Bound'
                ),
        go.Scatter(x=months,
                y=local_plot['DLY_TMIN_NORMAL']-local_plot['DLY-TMIN-STDDEV'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.15)',
                fill='tonexty',
                showlegend=False,
                name='Lower Bound'
                )  
    ])

    fig_dly_temp.update_layout(
        margin=dict(l=1,r=1,b=1,t=1),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig_dly_temp.update_xaxes(title_text='Month')
    fig_dly_temp.update_yaxes(title_text='Temp (F)')

    st.write(fig_dly_temp)


def precip_chart(local_dly):
    st.subheader('Expected Precipitation')
    local_plot = local_dly[local_dly['day']==1]
    fig = plt.figure()
    #sns.lineplot(x=local_plot['mm-dd'],y=local_plot['DLY-PRCP-PCTALL-GE005MM'])
    #sns.lineplot(x=local_plot['mm-dd'],y=local_plot['DLY_PRCP_PCTALL_GE050MM'])
    #sns.despine()
    #plt.legend()
    #st.write(fig)

    #fancy plotly line plot
    months = ['January','February','March', 'April', 'May', 'June','July','August','September','October','November','December']

    fig_precip=go.Figure([
        go.Scatter(x=months,
                y=local_plot['DLY_PRCP_PCTALL_GE001MM'],
                mode='lines',
                visible=True,
                line_color='green',
                name='Light Rain'
                ),
        go.Scatter(x=months, 
                y=local_plot['DLY-PRCP-PCTALL-GE005MM'], 
                mode='lines',
                visible=True,
                line_color='blue',
                name='More Rain'
                ),
        go.Scatter(x=months,
                y=local_plot['DLY_PRCP_PCTALL_GE050MM'],
                mode='lines',
                visible=True,
                line_color='red',
                name='Heavy Rain'
                ),
    ])

    fig_precip.update_layout(
        margin=dict(l=1,r=1,b=1,t=1),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig_precip.update_xaxes(title_text='Month')
    fig_precip.update_yaxes(title_text='% Chance of Rain')

    st.write(fig_precip)


## input form
with st.form("my_form"):
    st.write("Input criteria for weather query")
    user_input_zip = st.text_input('Enter zip code', '78701')
    low_temp, high_temp = st.select_slider(
        'Select desired high temp range',
        options=[i for i in np.arange(30,105,5)],
        value=(65,85))    
     
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    
if submitted:
    lat, long, place = get_location_info(user_input_zip)
    nearest_station = get_nearest_station(inventory,lat,long)
    local_dly, best_day_index = get_weather_score(dly, nearest_station,low_temp,high_temp)
    show_score(local_dly, place, best_day_index,low_temp,high_temp)
    weather_score_chart(local_dly)
    temp_chart(local_dly)
    precip_chart(local_dly)




# 1 - user enters zip code
#user_input_zip = st.text_input('Enter zip code', '78701')


# 2 - retreive lat/long for zip code using pgeocode

# 3 - save lat and long for zip code 
##st.write(f'...finding weather normals for {place} (lat: {lat}, long: {long})')

# 4 - get nearest station
#nearest_station = get_nearest_station(inventory,lat,long)
#st.write(f'...the nearest station is: {nearest_station}')

#5 - get score for weather in this location


# 5a print score details

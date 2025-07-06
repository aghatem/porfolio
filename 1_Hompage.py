import streamlit as st

st.set_page_config(page_title='Portfolio Homepage', layout="wide", initial_sidebar_state="expanded",page_icon='🏠',)

st.markdown("""<style>.main {background-color: #f4f4f4;}
</style>""", unsafe_allow_html=True)
st.sidebar.success('Select a page aboe.')

from PIL import Image

st.write('# Welcome to My Data Analytics Portfolio!')
st.write('In this page, I showcase my data analytics projects where I leverage the power of data to gain insights, make informed decisions, and solve real-world problems.')

# Add a title and image to the page

image = Image.open(r'data_analytics_image.png')
st.image(image,  width=950)

st.write('All datasets used in these projects are publicly available data and I try to keep in each project the data sources for reference,')
# Description of projects
st.write('## Projects')
st.write('Here are some of the data analytics projects I have worked on:')

# Project 1
st.write('### Project 1 - UK schools APP ')
st.write('- Description: This project involved analyzing the data available for the 48k+ schools in United Kingdom. It allows parents to learn about the schools in their district and evaluate them based on multiple criteria using interactive maps and statistical charts about the schools performance, type, distance from home and other factors.')
st.write('- Technologies Used: Python, Pandas, geospatial data visualisation using folium, Matplotlib, Google Geo-map APIs.')
st.write('- Results: The project aims to facilitate the school searchfor parents in United Kingdom and provide them the data needed to decide about the best school for their childs.')

# Project 2
st.write('### Project 2 - Stock analysis & prediction APP')
st.write('- Description: This project focused on analysing the stock market historical data, calculates the key indicators needed for technical analysis and uses predictive modeling to forecast the enxt day closing value.')
st.write('- Technologies Used: Python, yfinance API, HTML parsing, Scikit-learn, and Time Series Analysis.')
st.write('- Results: The app is accessing a wide range of financial data, market trends, historical prices, and fundamental indicators. It empowers users to make informed investment decisions based on data rather than relying solely on speculation or emotions.') 
st.write('The app enable users to monitor the performance of stocks and track the top gainers and loosers stocks.')

# Project 3
# st.write('### Project 3')
# st.write('- Description: TBD.')
# st.write('- Technologies Used: TBD.')
# st.write('- Results: TBD.')

st.write('## Disclaimer')
st.write('Please note that the results of the code and analyses presented in this portfolio are provided for informational purposes only. They should not be used for commercial purposes as the accuracy of the results is not guaranteed. The code and methodologies are subject to limitations, assumptions, and potential errors. It is always recommended to validate and verify the results through rigorous testing and independent evaluation.')

# Conclusion
st.write('Thank you for visiting my data analytics portfolio. Feel free to explore the projects and reach out to me for any inquiries or collaborations.')

# Feedback section
st.write('## Leave Feedback')
feedback = st.text_input('Enter your feedback here')

# Save feedback
if st.button('Submit'):
    # Save the feedback to a file or database
    # You can customize this part based on your needs
    with open('feedback.txt', 'a') as f:
        f.write(feedback + '\n')
    st.success('Thank you for your feedback!')

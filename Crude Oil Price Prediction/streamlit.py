from prophet import Prophet
import pandas as pd
import streamlit as st
import pickle

pickle_in = open("p_model", 'rb')
loadmodel = pickle.load(pickle_in)

def prediction(input_da):
    prediction = loadmodel.predict(input_da)
    return prediction
    
def main():
    # Set the app title
    
    st.title(' Oil Price Forecasting')
    #st.write('''This data app uses Facebook's open-source Prophet library to automatically generate future forecast values from an imported dataset.
                #You'll be able to import your data from a CSV file, visualize trends and features, analyze forecast performance, and finally download the created forecast
                #''')
    st.subheader("Select Date")
    # Add a date input widget
    date_input = st.date_input("Enter a date:", value=None, min_value=None, max_value=None)

    
    if st.button("Predict"):
        # Prepare the input data for prediction
        input_date = pd.to_datetime(date_input)
        input_data = pd.DataFrame({'ds': [input_date]})

        predictions = prediction(input_data)
        # Extract price, upper range, and lower range values
        price = predictions['yhat'].values[0]
        upper_range = predictions['yhat_upper'].values[0]
        lower_range = predictions['yhat_lower'].values[0]

        # Display the predictions
        st.subheader('Predicted Price:')
        st.write("Forecasted Oil Price(USD/BBL):", price)
        st.write("Upper Range(USD/BBL):", upper_range)
        st.write("Lower Range(USD/BBL):", lower_range)
        # Generate future dates for prediction
        
    st.subheader("Select Date Range")    
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

# Check if the start date is before the end date
    if start_date <= end_date:
    # Generate a range of dates between the start and end date
        date_range = pd.date_range(start=start_date, end=end_date)

    # Create a dataframe with the dates
        df = pd.DataFrame(date_range, columns=["ds"])

    else:
       st.error("Error: The start date must be before or equal to the end date.")    
       
    predictions = prediction(df)
    # Visualize the predicted graph
    st.subheader("Predicted Graph")
      
    chart_data = predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    chart_data.set_index('ds', inplace=True)
    st.line_chart(chart_data)
    st.subheader("Table")
    st.write(predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

if __name__ == "__main__":
    main()

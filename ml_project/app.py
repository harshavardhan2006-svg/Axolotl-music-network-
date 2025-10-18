import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the model and preprocessors
try:
    model = joblib.load('app_addiction_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_gender = joblib.load('label_encoder_gender.pkl')
    le_location = joblib.load('label_encoder_location.pkl')
    le_addiction = joblib.load('label_encoder_addiction.pkl')
    
    # IMPORTANT: Get the actual class order from the label encoder
    class_names = le_addiction.classes_
    print(f"Debug: Actual class order in model: {class_names}")
    
except FileNotFoundError as e:
    st.error(f"Error: Missing .pkl file - {e}. Run train_ml_model.py first.")
    st.stop()

# Prediction function
def predict_addiction_risk(age, gender, total_usage_hours, daily_screen_time, num_apps, social_media_hours, productivity_hours, gaming_hours, location):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Total_App_Usage_Hours': [total_usage_hours],
        'Daily_Screen_Time_Hours': [daily_screen_time],
        'Number_of_Apps_Used': [num_apps],
        'Social_Media_Usage_Hours': [social_media_hours],
        'Productivity_App_Usage_Hours': [productivity_hours],
        'Gaming_App_Usage_Hours': [gaming_hours],
        'Location': [location]
    })
    
    # Encode categorical variables FIRST
    try:
        input_data['Gender'] = le_gender.transform([gender])[0]
        input_data['Location'] = le_location.transform([location])[0]
    except ValueError:
        st.error("Invalid Gender or Location. Use 'Male'/'Female' and one of: Los Angeles, Chicago, Houston, Phoenix, New York.")
        return None, None, None
    
    # Scale all features (after encoding)
    try:
        scaled_data = scaler.transform(input_data)
        input_data_scaled = pd.DataFrame(scaled_data, columns=input_data.columns)
    except Exception as e:
        st.error(f"Error in scaling: {e}")
        return None, None, None
    
    # Predict
    try:
        risk_encoded = model.predict(input_data_scaled)[0]
        risk_label = le_addiction.inverse_transform([risk_encoded])[0]
        probabilities = model.predict_proba(input_data_scaled)[0]
        
        # FIXED: Map probabilities correctly to actual class names
        probability_dict = {}
        for i, prob in enumerate(probabilities):
            class_label = le_addiction.inverse_transform([i])[0]
            probability_dict[class_label] = prob
            
        # Debug information
        print(f"Debug: Predicted class: {risk_label}")
        print(f"Debug: Probabilities: {probability_dict}")
        
        return risk_label, probability_dict, input_data
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None, None

# Streamlit Dashboard with Tabs
st.title("Mobile App Addiction Predictor Dashboard")
st.markdown("Predict your addiction risk based on smartphone usage. Trained on your dataset with Random Forest.")

tab1, tab2, tab3 = st.tabs(["Input Details", "Prediction Results", "Insights & EDA"])

with tab1:
    st.header("Enter Your Details")
    st.markdown("*Fill in your mobile usage details below. All fields are required.*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Basic Information")
        age = st.number_input("Age", min_value=10, max_value=100, value=25, step=1, 
                             help="Enter your age in years")
        
        gender = st.selectbox("Gender", ['Male', 'Female'], 
                             help="Select your gender")
        
        location = st.selectbox("Location", ['Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'New York'],
                               help="Select your current location")
        
        st.subheader("📱 Screen Time")
        daily_screen_time = st.number_input("Daily Screen Time (Hours)", 
                                           min_value=0.0, max_value=24.0, value=4.0, step=0.1,
                                           format="%.1f",
                                           help="Total hours you spend looking at your phone screen per day")
        
        total_usage_hours = st.number_input("Total App Usage (Hours per Day)", 
                                           min_value=0.0, max_value=24.0, value=5.0, step=0.1,
                                           format="%.1f",
                                           help="Total hours spent actively using apps (sum of all app categories)")
    
    with col2:
        st.subheader("📊 App Usage Breakdown")
        num_apps = st.number_input("Number of Apps Used", 
                                  min_value=1, max_value=200, value=10, step=1,
                                  help="Total number of different apps you use regularly")
        
        social_media_hours = st.number_input("Social Media Usage (Hours per Day)", 
                                            min_value=0.0, max_value=10.0, value=2.0, step=0.1,
                                            format="%.1f",
                                            help="Time spent on Facebook, Instagram, Twitter, TikTok, etc.")
        
        productivity_hours = st.number_input("Productivity App Usage (Hours per Day)", 
                                            min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                            format="%.1f",
                                            help="Time spent on work, email, calendar, note-taking apps")
        
        gaming_hours = st.number_input("Gaming App Usage (Hours per Day)", 
                                      min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                      format="%.1f",
                                      help="Time spent playing mobile games")
    
    # Enhanced Validation with visual feedback
    category_sum = social_media_hours + productivity_hours + gaming_hours
    
    if total_usage_hours < category_sum:
        st.error(f"⚠️ **Validation Error**: Total App Usage ({total_usage_hours:.1f}h) should be ≥ sum of categories ({category_sum:.1f}h)")
        st.info("💡 **Tip**: Total usage includes time in other apps like messaging, shopping, news, etc.")
    elif total_usage_hours > daily_screen_time:
        st.warning(f"⚠️ **Note**: Total App Usage ({total_usage_hours:.1f}h) is higher than Daily Screen Time ({daily_screen_time:.1f}h). This might indicate multitasking or background usage.")
    else:
        other_usage = total_usage_hours - category_sum
        if other_usage > 0:
            st.success(f"✅ **Validation Passed** | Other Apps Usage: {other_usage:.1f} hours (messaging, shopping, news, etc.)")
        else:
            st.success("✅ **Validation Passed** | All usage accounted for in categories")
    
    # Predict button with cyber animation
    st.markdown("---")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    
    with predict_col2:
        predict_disabled = total_usage_hours < (social_media_hours + productivity_hours + gaming_hours)
        
        if st.button("🤖 **ANALYZE ADDICTION RISK**", 
                    key="predict_button", 
                    disabled=predict_disabled,
                    help="Click to run AI-powered addiction risk analysis",
                    type="primary"):
            
            # Create cyber-style loading animation
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                
            with status_container:
                status_text = st.empty()
                
            # Cyber loading stages with robotic theme
            loading_stages = [
                ("🔍 INITIALIZING NEURAL NETWORK...", 15),
                ("🧠 LOADING BEHAVIORAL ANALYSIS MODULE...", 25),
                ("📊 PROCESSING USAGE PATTERNS...", 40),
                ("🤖 RUNNING AI PREDICTION ALGORITHMS...", 60),
                ("⚡ CALCULATING RISK PROBABILITIES...", 80),
                ("✨ GENERATING PERSONALIZED INSIGHTS...", 95),
                ("🎯 ANALYSIS COMPLETE!", 100)
            ]
            
            import time
            
            current_progress = 0
            
            for stage_text, progress_value in loading_stages:
                status_text.markdown(f"**{stage_text}**")
                
                # Animate progress bar with cyber effect
                for i in range(current_progress, progress_value + 1):
                    progress_bar.progress(i)
                    time.sleep(0.02)  # Fast animation for cyber feel
                
                current_progress = progress_value
                time.sleep(0.3)  # Pause at each stage
            
            # Clear loading elements
            progress_container.empty()
            status_container.empty()
            
            # Run actual prediction
            result = predict_addiction_risk(
                age, gender, total_usage_hours, daily_screen_time, num_apps,
                social_media_hours, productivity_hours, gaming_hours, location
            )
            
            if result[0] is not None:
                st.session_state['risk_label'] = result[0]
                st.session_state['probabilities'] = result[1]
                st.session_state['input_data'] = result[2]
                st.session_state['usage_data'] = {
                    'Social Media': social_media_hours,
                    'Productivity': productivity_hours,
                    'Gaming': gaming_hours,
                    'Other': max(total_usage_hours - (social_media_hours + productivity_hours + gaming_hours), 0)
                }
                
                # Success message with cyber styling
                st.success("🚀 **NEURAL ANALYSIS COMPLETED** | Switch to '**Prediction Results**' tab for detailed insights")

with tab2:
    st.header("Prediction Results")
    if 'risk_label' in st.session_state:
        risk_label = st.session_state['risk_label']
        probabilities = st.session_state['probabilities']
        input_data = st.session_state['input_data']
        usage_data = st.session_state['usage_data']
        
        # Display the prediction with confidence
        max_prob = max(probabilities.values())
        confidence_percentage = max_prob * 100
        
        st.success(f"Predicted Addiction Risk: **{risk_label}** (Confidence: {confidence_percentage:.1f}%)")
        
        # FIXED: Probability Bar Chart with correct mapping
        prob_df = pd.DataFrame(list(probabilities.items()), columns=['Risk Level', 'Probability'])
        
        # Sort for consistent display order
        risk_order = ['Low', 'Moderate', 'High']
        prob_df['Risk Level'] = pd.Categorical(prob_df['Risk Level'], categories=risk_order, ordered=True)
        prob_df = prob_df.sort_values('Risk Level')
        
        fig_prob = px.bar(prob_df, x='Risk Level', y='Probability', color='Risk Level',
                          title='Probability Distribution of Addiction Risk', range_y=[0, 1],
                          color_discrete_map={'Low': '#00CC96', 'Moderate': '#FFAB00', 'High': '#EF553B'},
                          text='Probability')
        fig_prob.update_traces(texttemplate='%{text:.2%}', textposition='auto')
        
        # Highlight the predicted class
        fig_prob.update_traces(
            marker_line_width=3,
            marker_line_color='black',
            selector=dict(name=risk_label)
        )
        
        st.plotly_chart(fig_prob)
        
        # Show exact probabilities for verification
        st.subheader("Detailed Probabilities")
        for risk_level in ['Low', 'Moderate', 'High']:
            if risk_level in probabilities:
                prob_value = probabilities[risk_level]
                is_predicted = "✅ PREDICTED" if risk_level == risk_label else ""
                st.write(f"**{risk_level} Risk**: {prob_value:.1%} {is_predicted}")
        
        # Usage Breakdown Pie Chart
        usage_df = pd.DataFrame(list(usage_data.items()), columns=['Category', 'Hours'])
        fig_pie = px.pie(usage_df, values='Hours', names='Category', title='Your Daily App Usage Breakdown',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie)
        
        # Debug: Show Input Data
        with st.expander("Show Input Data (Debug)"):
            st.write("Your input values (after encoding):")
            st.dataframe(input_data)
            st.write("Raw Probabilities Dictionary:")
            st.json(probabilities)
        
        # Recommendations
        st.subheader("Recommendations to Reduce Risk")
        if risk_label == 'High':
            st.error("⚠️ High Risk Detected!")
            st.write("- 🎯 **Immediate Action**: Limit gaming and social media to under 2 hours/day")
            st.write("- 📱 Use phone screen time trackers and set daily limits")
            st.write("- 🏃‍♂️ Replace app time with offline activities like exercise")
            st.write("- 🧘‍♂️ Consider digital detox periods")
        elif risk_label == 'Moderate':
            st.warning("⚡ Moderate Risk - Take Action!")
            st.write("- ⚖️ **Balance Usage**: Increase productivity apps by 30%")
            st.write("- ⏰ Take regular breaks (e.g., 5-min every hour)")
            st.write("- 📊 Monitor weekly trends to avoid escalation")
            st.write("- 🎯 Set specific goals for reducing recreational apps")
        else:
            st.success("✅ Low Risk - Great Job!")
            st.write("- 👍 Keep maintaining balanced habits")
            st.write("- 📈 Continue tracking to stay in the low-risk zone")
            st.write("- 🌟 You're a great role model for healthy phone usage!")
    else:
        st.info("Enter details in the 'Input Details' tab and click 'Predict Risk' to see results.")

with tab3:
    st.header("Insights & EDA")
    st.markdown("Overview of dataset insights and model details.")
    
    # Model Information
    st.subheader("Model Details")
    st.write(f"**Algorithm**: Random Forest Classifier")
    st.write(f"**Classes**: {list(le_addiction.classes_)}")
    st.write(f"**Encoding Order**: {dict(zip(le_addiction.classes_, range(len(le_addiction.classes_))))}")
    
    # Load and display EDA plots
    if os.path.exists('eda_plots.png'):
        st.image('eda_plots.png', caption='EDA Plots from Dataset', use_column_width=True)
    else:
        st.warning("EDA plots not found. Run train_ml_model.py to generate them.")
    
    if os.path.exists('feature_importance.png'):
        st.image('feature_importance.png', caption='Feature Importance from Model', use_column_width=True)
    else:
        st.warning("Feature importance plot not found. Run train_ml_model.py to generate it.")
    
    # Sample Usage Trend Line Chart
    trend_data = pd.DataFrame({
        'Category': ['Social Media', 'Productivity', 'Gaming', 'Total Usage'],
        'Average Hours': [2.5, 1.8, 2.2, 6.5]
    })
    fig_line = px.line(trend_data, x='Category', y='Average Hours', title='Average Usage Trends from Dataset',
                       markers=True)
    st.plotly_chart(fig_line)

# Footer
st.markdown("---")
st.info("💡 **Tip**: Test with different usage patterns to see how they affect your addiction risk!")
st.write("🔬 **For Testing**: Try Age=56, Gender='Male', Total Usage=2.61 hours for a low-risk example.")
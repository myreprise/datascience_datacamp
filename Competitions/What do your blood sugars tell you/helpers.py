import plotly.graph_objs as go
import plotly.offline as pyo

def assess_risk(value):
    if value < 0.20:
        return "LOW"
    elif value < 0.50:
        return "MODERATE"
    elif value < 0.80:
        return "ELEVATED"
    else:
        return "HIGH"


# Function to create the 3D scatter plot
def create_3d_scatter_plot(user_data=None, df=None):
    # Separate data based on the outcome
    non_diabetic = df[df['Outcome'] == 0]
    diabetic = df[df['Outcome'] == 1]

    # Trace for non-diabetic observations (blue)
    trace_non_diabetic = go.Scatter3d(
        x=non_diabetic['Glucose'],
        y=non_diabetic['BMI'],
        z=non_diabetic['Age'],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',  # Color for non-diabetic
            opacity=0.6
        ),
        name='Non-Diabetic'
    )

    # Trace for diabetic observations (red)
    trace_diabetic = go.Scatter3d(
        x=diabetic['Glucose'],
        y=diabetic['BMI'],
        z=diabetic['Age'],
        mode='markers',
        marker=dict(
            size=5,
            color='red',  # Color for diabetic
            opacity=0.6
        ),
        name='Diabetic'
    )

    # User input data point
    if user_data:
        trace_user_input = go.Scatter3d(
            x=[user_data['Glucose']],
            y=[user_data['BMI']],
            z=[user_data['Age']],
            mode='markers',
            marker=dict(
                size=8,
                color='yellow',  # Distinct color for the user input
                opacity=1.0
            ),
            name='User Input'
        )
        data = [trace_non_diabetic, trace_diabetic, trace_user_input]
    else:
        data = [trace_non_diabetic, trace_diabetic]

    # Define layout
    layout = go.Layout(
        title='3D Scatter Plot of Glucose, BMI, and Age',
        scene=dict(
            xaxis=dict(title='Glucose'),
            yaxis=dict(title='BMI'),
            zaxis=dict(title='Age')
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        width=800,  # Increase plot width
        height=600,  # Increase plot height
        showlegend=True  # Show the legend
    )

    # Create the figure
    fig = go.Figure(data=data, layout=layout)

    # Display the plot
    pyo.iplot(fig)
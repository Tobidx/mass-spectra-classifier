import plotly.graph_objects as go

def create_comparison_plot(mz1, int1, mz2, int2, title1="Spectrum 1", title2="Spectrum 2"):
    """Create an interactive plot comparing two mass spectra"""
    fig = go.Figure()
    
    # Add first spectrum
    fig.add_trace(
        go.Scatter(
            x=mz1,
            y=int1,
            name=title1,
            line=dict(color='rgb(31, 119, 180)', width=1),
            mode='lines+markers',
            marker=dict(size=4),
            hovertemplate="m/z: %{x:.2f}<br>Intensity: %{y:.4f}<extra></extra>"
        )
    )
    
    # Add second spectrum
    fig.add_trace(
        go.Scatter(
            x=mz2,
            y=int2,
            name=title2,
            line=dict(color='rgb(255, 127, 14)', width=1),
            mode='lines+markers',
            marker=dict(size=4),
            hovertemplate="m/z: %{x:.2f}<br>Intensity: %{y:.4f}<extra></extra>"
        )
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Mass Spectra Comparison",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis=dict(
            title="m/z",
            tickfont=dict(size=12),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1
        ),
        yaxis=dict(
            title="Relative Intensity",
            tickfont=dict(size=12),
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='lightgray',
            zerolinewidth=1,
            rangemode='tozero'
        ),
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            x=1.0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        hovermode='x unified',
        height=600,
        margin=dict(l=80, r=20, t=100, b=80)
    )
    
    return fig.to_json()

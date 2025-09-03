#!/usr/bin/env python3
"""
Utility functions for data analysis, visualization, and business insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

def setup_plotting_style():
    """Setup consistent plotting style."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set figure size and DPI
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    
    # Set font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

def analyze_data_distribution(df, save_path=None):
    """Analyze and visualize data distribution."""
    print("📊 Analyzing data distribution...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Distribution Analysis', fontsize=20, fontweight='bold')
    
    # Age distribution
    axes[0, 0].hist(df['age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # BMI distribution
    axes[0, 1].hist(df['bmi'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('BMI Distribution')
    axes[0, 1].set_xlabel('BMI')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Children distribution
    axes[0, 2].hist(df['children'], bins=7, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('Number of Children Distribution')
    axes[0, 2].set_xlabel('Number of Children')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Sex distribution
    sex_counts = df['sex'].value_counts()
    axes[1, 0].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', 
                    colors=['lightblue', 'lightpink'])
    axes[1, 0].set_title('Sex Distribution')
    
    # Smoker distribution
    smoker_counts = df['smoker'].value_counts()
    axes[1, 1].pie(smoker_counts.values, labels=smoker_counts.index, autopct='%1.1f%%',
                    colors=['lightgreen', 'lightcoral'])
    axes[1, 1].set_title('Smoker Distribution')
    
    # Region distribution
    region_counts = df['region'].value_counts()
    axes[1, 2].bar(region_counts.index, region_counts.values, color='gold', alpha=0.7)
    axes[1, 2].set_title('Region Distribution')
    axes[1, 2].set_xlabel('Region')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Distribution analysis saved to: {save_path}")
    
    plt.show()
    
    return fig

def analyze_target_variable(df, save_path=None):
    """Analyze the target variable (charges)."""
    print("🎯 Analyzing target variable (charges)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Medical Charges Analysis', fontsize=20, fontweight='bold')
    
    # Charges distribution
    axes[0, 0].hist(df['charges'], bins=30, color='purple', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Medical Charges Distribution')
    axes[0, 0].set_xlabel('Charges ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Charges by smoking status
    smoker_charges = df.groupby('smoker')['charges'].mean()
    axes[0, 1].bar(smoker_charges.index, smoker_charges.values, 
                    color=['lightgreen', 'lightcoral'], alpha=0.7)
    axes[0, 1].set_title('Average Charges by Smoking Status')
    axes[0, 1].set_xlabel('Smoker')
    axes[0, 1].set_ylabel('Average Charges ($)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Charges by sex
    sex_charges = df.groupby('sex')['charges'].mean()
    axes[1, 0].bar(sex_charges.index, sex_charges.values, 
                    color=['lightblue', 'lightpink'], alpha=0.7)
    axes[1, 0].set_title('Average Charges by Sex')
    axes[1, 0].set_xlabel('Sex')
    axes[1, 0].set_ylabel('Average Charges ($)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Charges by region
    region_charges = df.groupby('region')['charges'].mean().sort_values(ascending=False)
    axes[1, 1].bar(region_charges.index, region_charges.values, 
                    color='gold', alpha=0.7)
    axes[1, 1].set_title('Average Charges by Region')
    axes[1, 1].set_xlabel('Region')
    axes[1, 1].set_ylabel('Average Charges ($)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🎯 Target variable analysis saved to: {save_path}")
    
    plt.show()
    
    return fig

def analyze_feature_relationships(df, save_path=None):
    """Analyze relationships between features and target."""
    print("🔗 Analyzing feature relationships...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Feature Relationships with Medical Charges', fontsize=20, fontweight='bold')
    
    # Age vs Charges
    axes[0, 0].scatter(df['age'], df['charges'], alpha=0.6, color='blue')
    axes[0, 0].set_title('Age vs Medical Charges')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Charges ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # BMI vs Charges
    axes[0, 1].scatter(df['bmi'], df['charges'], alpha=0.6, color='green')
    axes[0, 1].set_title('BMI vs Medical Charges')
    axes[0, 1].set_xlabel('BMI')
    axes[0, 1].set_ylabel('Charges ($)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Children vs Charges
    children_charges = df.groupby('children')['charges'].mean()
    axes[0, 2].bar(children_charges.index, children_charges.values, 
                    color='orange', alpha=0.7)
    axes[0, 2].set_title('Average Charges by Number of Children')
    axes[0, 2].set_xlabel('Number of Children')
    axes[0, 2].set_ylabel('Average Charges ($)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Age and BMI interaction
    scatter = axes[1, 0].scatter(df['age'], df['bmi'], c=df['charges'], 
                                 cmap='viridis', alpha=0.7, s=50)
    axes[1, 0].set_title('Age vs BMI (colored by Charges)')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('BMI')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='Charges ($)')
    
    # Smoking status impact
    smoker_age_charges = df.groupby(['smoker', 'age'])['charges'].mean().unstack()
    smoker_age_charges.plot(kind='line', ax=axes[1, 1], marker='o')
    axes[1, 1].set_title('Charges by Age and Smoking Status')
    axes[1, 1].set_xlabel('Age')
    axes[1, 1].set_ylabel('Average Charges ($)')
    axes[1, 1].legend(title='Smoker')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[1, 2], square=True)
    axes[1, 2].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🔗 Feature relationships analysis saved to: {save_path}")
    
    plt.show()
    
    return fig

def generate_business_insights(df, save_path=None):
    """Generate business insights from the data."""
    print("💼 Generating business insights...")
    
    insights = []
    
    # 1. Overall statistics
    total_charges = df['charges'].sum()
    avg_charges = df['charges'].mean()
    median_charges = df['charges'].median()
    
    insights.append({
        'category': 'Overall Statistics',
        'metric': 'Total Medical Charges',
        'value': f"${total_charges:,.2f}",
        'insight': 'Total liability across all policyholders'
    })
    
    insights.append({
        'category': 'Overall Statistics',
        'metric': 'Average Medical Charges',
        'value': f"${avg_charges:,.2f}",
        'insight': 'Expected cost per policyholder'
    })
    
    insights.append({
        'category': 'Overall Statistics',
        'metric': 'Median Medical Charges',
        'value': f"${median_charges:,.2f}",
        'insight': 'Typical cost (50th percentile)'
    })
    
    # 2. Risk factors analysis
    smoker_impact = df.groupby('smoker')['charges'].agg(['mean', 'count'])
    smoker_ratio = smoker_impact.loc['yes', 'count'] / len(df) * 100
    smoker_cost_multiplier = smoker_impact.loc['yes', 'mean'] / smoker_impact.loc['no', 'mean']
    
    insights.append({
        'category': 'Risk Factors',
        'metric': 'Smoker Cost Multiplier',
        'value': f"{smoker_cost_multiplier:.2f}x",
        'insight': f'Smokers cost {smoker_cost_multiplier:.1f}x more than non-smokers'
    })
    
    insights.append({
        'category': 'Risk Factors',
        'metric': 'Smoker Population',
        'value': f"{smoker_ratio:.1f}%",
        'insight': f'Only {smoker_ratio:.1f}% are smokers but they drive significant costs'
    })
    
    # 3. Age group analysis
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                             labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    age_group_charges = df.groupby('age_group')['charges'].mean()
    
    highest_age_group = age_group_charges.idxmax()
    highest_age_cost = age_group_charges.max()
    
    insights.append({
        'category': 'Age Analysis',
        'metric': 'Highest Cost Age Group',
        'value': f"{highest_age_group}",
        'insight': f'Age group {highest_age_group} has highest average costs (${highest_age_cost:,.0f})'
    })
    
    # 4. Regional analysis
    region_charges = df.groupby('region')['charges'].mean()
    highest_cost_region = region_charges.idxmax()
    lowest_cost_region = region_charges.idxmin()
    regional_variance = region_charges.max() - region_charges.min()
    
    insights.append({
        'category': 'Regional Analysis',
        'metric': 'Regional Cost Variance',
        'value': f"${regional_variance:,.0f}",
        'insight': f'Costs vary by ${regional_variance:,.0f} between {highest_cost_region} and {lowest_cost_region}'
    })
    
    # 5. BMI analysis
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], 
                                labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    bmi_charges = df.groupby('bmi_category')['charges'].mean()
    
    highest_bmi_cost = bmi_charges.max()
    highest_bmi_category = bmi_charges.idxmax()
    
    insights.append({
        'category': 'Health Factors',
        'metric': 'BMI Impact on Costs',
        'value': f"{highest_bmi_category}",
        'insight': f'{highest_bmi_category} individuals have highest costs (${highest_bmi_cost:,.0f})'
    })
    
    # Create insights DataFrame
    insights_df = pd.DataFrame(insights)
    
    # Save insights
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        insights_df.to_csv(save_path, index=False)
        print(f"💼 Business insights saved to: {save_path}")
    
    # Display insights
    print("\n" + "="*80)
    print("💼 BUSINESS INSIGHTS SUMMARY")
    print("="*80)
    
    for category in insights_df['category'].unique():
        print(f"\n📊 {category}:")
        category_insights = insights_df[insights_df['category'] == category]
        for _, row in category_insights.iterrows():
            print(f"   • {row['metric']}: {row['value']}")
            print(f"     {row['insight']}")
    
    return insights_df

def create_interactive_dashboard(df, save_path=None):
    """Create an interactive Plotly dashboard."""
    print("📊 Creating interactive dashboard...")
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Age vs Charges by Smoking Status', 'BMI vs Charges by Region',
                       'Charges Distribution by Sex', 'Regional Cost Comparison',
                       'Children Impact on Costs', 'Age Group Analysis'),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "histogram"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 1. Age vs Charges by Smoking Status
    for smoker_status in df['smoker'].unique():
        subset = df[df['smoker'] == smoker_status]
        fig.add_trace(
            go.Scatter(x=subset['age'], y=subset['charges'], mode='markers',
                      name=f'Smoker: {smoker_status}', opacity=0.7),
            row=1, col=1
        )
    
    # 2. BMI vs Charges by Region
    for region in df['region'].unique():
        subset = df[df['region'] == region]
        fig.add_trace(
            go.Scatter(x=subset['bmi'], y=subset['charges'], mode='markers',
                      name=f'Region: {region}', opacity=0.7),
            row=1, col=2
        )
    
    # 3. Charges Distribution by Sex
    for sex in df['sex'].unique():
        subset = df[df['sex'] == sex]
        fig.add_trace(
            go.Histogram(x=subset['charges'], name=f'Sex: {sex}', opacity=0.7),
            row=2, col=1
        )
    
    # 4. Regional Cost Comparison
    region_charges = df.groupby('region')['charges'].mean()
    fig.add_trace(
        go.Bar(x=region_charges.index, y=region_charges.values,
               name='Average Charges by Region', marker_color='gold'),
        row=2, col=2
    )
    
    # 5. Children Impact on Costs
    children_charges = df.groupby('children')['charges'].mean()
    fig.add_trace(
        go.Bar(x=children_charges.index, y=children_charges.values,
               name='Average Charges by Children', marker_color='lightgreen'),
        row=3, col=1
    )
    
    # 6. Age Group Analysis
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                             labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    age_group_charges = df.groupby('age_group')['charges'].mean()
    fig.add_trace(
        go.Bar(x=age_group_charges.index, y=age_group_charges.values,
               name='Average Charges by Age Group', marker_color='lightcoral'),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Medical Insurance Claims Cost Analysis Dashboard',
        height=1200,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Age", row=1, col=1)
    fig.update_yaxes(title_text="Charges ($)", row=1, col=1)
    fig.update_xaxes(title_text="BMI", row=1, col=2)
    fig.update_yaxes(title_text="Charges ($)", row=1, col=2)
    fig.update_xaxes(title_text="Charges ($)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Region", row=2, col=2)
    fig.update_yaxes(title_text="Average Charges ($)", row=2, col=2)
    fig.update_xaxes(title_text="Number of Children", row=3, col=1)
    fig.update_yaxes(title_text="Average Charges ($)", row=3, col=1)
    fig.update_xaxes(title_text="Age Group", row=3, col=2)
    fig.update_yaxes(title_text="Average Charges ($)", row=3, col=2)
    
    if save_path:
        fig.write_html(save_path)
        print(f"📊 Interactive dashboard saved to: {save_path}")
    
    fig.show()
    
    return fig

def save_all_analyses(df, base_path='results'):
    """Save all analyses and visualizations."""
    print("💾 Saving all analyses...")
    
    os.makedirs(base_path, exist_ok=True)
    
    # Setup plotting style
    setup_plotting_style()
    
    # 1. Data distribution analysis
    dist_path = os.path.join(base_path, 'data_distribution_analysis.png')
    analyze_data_distribution(df, dist_path)
    
    # 2. Target variable analysis
    target_path = os.path.join(base_path, 'target_variable_analysis.png')
    analyze_target_variable(df, target_path)
    
    # 3. Feature relationships analysis
    relationships_path = os.path.join(base_path, 'feature_relationships_analysis.png')
    analyze_feature_relationships(df, relationships_path)
    
    # 4. Business insights
    insights_path = os.path.join(base_path, 'business_insights.csv')
    insights_df = generate_business_insights(df, insights_path)
    
    # 5. Interactive dashboard
    dashboard_path = os.path.join(base_path, 'interactive_dashboard.html')
    create_interactive_dashboard(df, dashboard_path)
    
    print(f"✅ All analyses saved to {base_path}/ directory")
    
    return {
        'distribution': dist_path,
        'target': target_path,
        'relationships': relationships_path,
        'insights': insights_path,
        'dashboard': dashboard_path
    }

if __name__ == "__main__":
    print("🧪 Testing utility functions...")
    print("✅ Utility functions loaded successfully!")

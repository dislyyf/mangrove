import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('automl.pkl')  # 加载训练好的XGBoost模型

# Define the feature options


# Streamlit UI
st.title("红树林成活率模拟")  # 红树林成活率模拟

# Sidebar for input options
st.sidebar.header("请输入要预测的数据")  # 侧边栏输入样本数据

# dem input
dem = st.sidebar.number_input("红树林滩涂高程:", min_value=-2.0, max_value=2.0, value=0.8)  # 流速输入框

# blc input
blc = st.sidebar.number_input("滩涂滩面侵蚀与淤积:", min_value=-1.0, max_value=1.0, value=0.0266)  # 侵蚀与淤积输入框

# current input
current = st.sidebar.number_input("海水流速:", min_value=0.0, max_value=5.0, value=0.004)  # 流速输入框

# ssc input
ssc = st.sidebar.number_input("悬浮泥沙浓度:", min_value=0.0, max_value=1.0, value=0.0043)  # 悬浮泥沙输入框

# flt input
flt = st.sidebar.number_input("红树林水淹时间:", min_value=0.0, max_value=1.0, value=0.2736)  # 水淹输入框

# flt input
rpbss = st.sidebar.number_input("红树林滩涂切应力:", min_value=0.0, max_value=1.0, value=0.0036)  # 切应力输入框

# Process the input and make a prediction
feature_values = [dem, blc, current, ssc, flt, rpbss]  # 收集所有输入的特征
features = np.array([feature_values])  # 转换为NumPy数组

if st.button("红树林成活率预测"):  # 如果点击了预测按钮
    # Predict the class and probabilities
    predicted_class = model.predict(features)[0]  # 预测心脏病类别
    predicted_proba = model.predict_proba(features)[0]  # 预测各类别的概率

    # Display the prediction results
    st.write(f"**Predicted Class:** {predicted_class}")  # 显示预测的类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比

    if predicted_class == 1:  # 如果预测为心脏病
        advice = (
            f"根据我们建立的水动力和人工智能AI模型, 您所设置的生态修复场景参数能够使得该地区的红树林成功定植，该地区的红树林能够成活. "
            f"根据您的参数该地区的红树林成活率为 {probability:.1f}%. "
            "虽然这只是一个概率估计，但它表明您通过该情景的设置，红树林可能会成活. "
        )  # 如果预测为心脏病，给出相关建议
    else:  # 如果预测为无心脏病
        advice = (
            f"根据我们建立的水动力和人工智能AI模型, 您所设置的生态修复场景参数会使得该地区的红树林定植失败，该地区的红树林会死亡. "
            f"根据您的参数该地区的红树林成活率为 {probability:.1f}%. "
            "虽然这只是一个概率估计，但它表明您通过该情景的设置，红树林可能会死亡. "
        )  # 如果预测为无心脏病，给出相关建议

    st.write(advice)  # 显示建议

    # Visualize the prediction probabilities
    sample_prob = {
        'Class_0': predicted_proba[0],  # 类别0的概率
        'Class_1': predicted_proba[1]  # 类别1的概率
    }

    # Set figure size
    plt.figure(figsize=(10, 3))  # 设置图形大小

    # Create bar chart
    bars = plt.barh(['death', 'survive'], 
                    [sample_prob['Class_0'], sample_prob['Class_1']], 
                    color=['#FF00FF', '#238E23'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for mangrove", fontsize=20, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=14, fontweight='bold')  # 添加X轴标签，并设置字体大小和加粗
    plt.ylabel("Classes", fontsize=14, fontweight='bold')  # 添加Y轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['Class_0'], sample_prob['Class_1']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot
    st.pyplot(plt)  # 显示图表
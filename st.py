import streamlit as st
import pandas as pd
import mcdm
from pymcdm import weights as w
from pymcdm.helpers import rrankdata
import altair as alt

st.subheader('Объекты')

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.caption("Object 1")
    o1f1 = st.slider('Feature 1', 0.0, 1.0, 0.1, key='001')
    o1f2 = st.slider('Feature 2', 0.0, 1.0, 0.1, key='0012')
    o1f3 = st.slider('Feature 3', 0.0, 1.0, 0.1, key='0013')
   

with col2:
    st.caption("Object 2")
    o2f1 = st.slider('Feature 1', 0.0, 1.0, 0.2, key='002')
    o2f2 = st.slider('Feature 2', 0.0, 1.0, 0.2, key='0022')
    o2f3 = st.slider('Feature 3', 0.0, 1.0, 0.2, key='0023')
   

with col3:
    st.caption("Object 3")
    o3f1 = st.slider('Feature 1', 0.0, 1.0, 0.3, key='003')
    o3f2 = st.slider('Feature 2', 0.0, 1.0, 0.3, key='0032')
    o3f3 = st.slider('Feature 3', 0.0, 1.0, 0.3, key='0033')
   
with col4:
    st.caption("Object 4")
    o4f1 = st.slider('Feature 1', 0.0, 1.0, 0.4, key='004')
    o4f2 = st.slider('Feature 2', 0.0, 1.0, 0.4, key='0042')
    o4f3 = st.slider('Feature 3', 0.0, 1.0, 0.4, key='0043')


with col5:
    st.caption("Object 5")
    o5f1 = st.slider('Feature 1', 0.0, 1.0, 0.5, key='005')
    o5f2 = st.slider('Feature 2', 0.0, 1.0, 0.5, key='0052')
    o5f3 = st.slider('Feature 3', 0.0, 1.0, 0.5, key='0053')

#collecting the data from the sliders and putting them in a dataframe
data = {'Object 1': [o1f1, o1f2, o1f3], 'Object 2': [o2f1, o2f2, o2f3], 'Object 3': [o3f1, o3f2, o3f3], 'Object 4': [o4f1, o4f2, o4f3], 'Object 5': [o5f1, o5f2, o5f3]}
df = pd.DataFrame(data, index=['Feature 1', 'Feature 2', 'Feature 3']).T

st.subheader('Данные по объектам')
#displaying the dataframe
st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)



weighting_methods = [
    w.equal_weights,
    w.entropy_weights,
    w.standard_deviation_weights,
    w.gini_weights,
    w.critic_weights,
    w.angle_weights

]

weight_sets = []
for m in weighting_methods:
    weight_sets.append(m(df.to_numpy()))


labels=['Equal', 'Entropy', 'Std', 'Gini', 'Critic', 'Angle']

#displaying the weights in a altair Stacked Bar Chart with Sorted Segments

dtlist = []

for i in range(len(weight_sets)):
    df2 = pd.DataFrame(weight_sets[i], index=['Feature 1', 'Feature 2', 'Feature 3'])
    df2 = df2.T
    df2['method'] = labels[i]
    dtlist.append(df2)


df2 = pd.concat(dtlist)


df2 = df2.melt('method', var_name='Feature', value_name='Weight')
df2 = df2.sort_values(by=['Feature', 'Weight'])



#transforming the dataframe df2 that 'method' is the index and 'Feature' is the columns and the values are the 'Weight'
df3 = df2.pivot(index='method', columns='Feature', values='Weight')

st.subheader('Веса по методам')
st.dataframe(df3, use_container_width=True)




cc = alt.Chart(df2).mark_bar().encode(
    x='method',
    y='sum(Weight)',
    color='Feature'
)

st.subheader('Веса по методам')
st.altair_chart(cc, use_container_width=True)

alt_names = ["Object 1", "Object 2", "Object 3", "Object 4", "Object 5"]

prefs = []
ranks = []
for i in range(len(weight_sets)):

    rr = mcdm.rank(df.to_numpy(), alt_names=alt_names, w_vector=weight_sets[i], s_method="SAW")

    # transform list of tuples rr to list of 0 elements of each tuple
    rr = [x[0] for x in rr]

    prefs.append(rr)

 



df4 = pd.DataFrame(prefs, index=labels).T
#df4 = df4.sort_values(by=['SAW'])

st.subheader('Итоговая сортировка по взвешанной сумме')

st.dataframe(df4, use_container_width=True)








import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
import plotly.io as pio
pio.kaleido.scope.default_format = "pdf"
pio.renderers.default = "browser"


def plot_3d_curve(X, titlestr=""):

    fig = go.Figure(go.Scatter3d(x=X[0, :], y=X[1, :], z=X[2, :], mode='lines'))
    fig = set_generic_fig_properties(fig, title_text=titlestr)
    fig.show()


def plot_3d_curve_set(X, titlestr=""):

    # X is a N x n x T numpyarray
    fig = go.Figure()
    for ii in range(X.shape[0]):
        fig.add_trace(go.Scatter3d(x=X[ii][0, :], y=X[ii][1, :], z=X[ii][2, :], mode='lines'))

    fig = set_generic_fig_properties(fig, title_text=titlestr)
    fig.show()


def set_generic_fig_properties(fig, height=600, width=600, title_text="", showticks=False, showlegend=False):
    fig.update_layout(autosize=True, height=height, width=width, title_text=title_text,
                      yaxis=dict(scaleanchor="x"), showlegend=showlegend,
                      margin=dict(r=5, l=5, t=25, b=5), scene=dict(aspectmode="data"))
    fig.update_xaxes(showticklabels=showticks, autorange=True)
    fig.update_yaxes(showticklabels=showticks, autorange="reversed", scaleratio=1)
    # fig.update_yaxes(showticklabels=showticks, autorange=True)


    return fig


def plot_1d_function_vec(X, titlestr=""):
    # X is a 2D array of row-wise functions
    fig = go.Figure()
    for ii in range(X.shape[0]):
        pass

        # fig.add_trace(go.Scatter3d(x=X[ii, :], y=X[ii, :], mode='lines'))


    return

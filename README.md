{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "bfe7d98e",
   "metadata": {},
   "source": [
    "# Credit Card Fraud Detection"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d506cec9",
   "metadata": {},
   "source": [
    "## The Problem\n",
    "\n",
    "Our goal is to predict if a credit card transaction is fraudulent or not. Because this data is so skewed, we'll use different techniques to try to make our predictive model work even when there is a majority class."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2496c7e8",
   "metadata": {},
   "source": [
    "## Collecting Data\n",
    "\n",
    "This data set came from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). We'll need to use certain libraries to accomplish our goal moving forward. I'll import Pandas, Numpy, Matplotlib and Seaborn initially. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "30d5d29e",
   "metadata": {},
   "source": [
    "#### Importing Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "ed98bdfe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "        <script type=\"text/javascript\">\n",
       "        window.PlotlyConfig = {MathJaxConfig: 'local'};\n",
       "        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}\n",
       "        if (typeof require !== 'undefined') {\n",
       "        require.undef(\"plotly\");\n",
       "        requirejs.config({\n",
       "            paths: {\n",
       "                'plotly': ['https://cdn.plot.ly/plotly-2.9.0.min']\n",
       "            }\n",
       "        });\n",
       "        require(['plotly'], function(Plotly) {\n",
       "            window._Plotly = Plotly;\n",
       "        });\n",
       "        }\n",
       "        </script>\n",
       "        "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# scientific computing libaries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# data mining libaries\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.decomposition import PCA#, FastICA\n",
    "from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, learning_curve\n",
    "from sklearn import svm\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score\n",
    "\n",
    "from imblearn.pipeline import make_pipeline, Pipeline\n",
    "from imblearn.over_sampling import SMOTE\n",
    "\n",
    "#plot libaries\n",
    "import plotly\n",
    "import plotly.graph_objs as go\n",
    "import plotly.figure_factory as ff\n",
    "from plotly.offline import init_notebook_mode\n",
    "init_notebook_mode(connected=True) # to show plots in notebook\n",
    "from matplotlib import pyplot as plt\n",
    "%matplotlib inline\n",
    "import seaborn as sns\n",
    "\n",
    "# offline plotly\n",
    "from plotly.offline import plot, iplot\n",
    "\n",
    "# do not show any warnings\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "SEED = 17 # specify seed for reproducable results\n",
    "pd.set_option('display.max_columns', None) # prevents abbreviation (with '...') of columns in prints\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb2b9639",
   "metadata": {},
   "source": [
    "#### Loading the Data frame"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "1e1253e6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Time</th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>V10</th>\n",
       "      <th>V11</th>\n",
       "      <th>V12</th>\n",
       "      <th>V13</th>\n",
       "      <th>V14</th>\n",
       "      <th>V15</th>\n",
       "      <th>V16</th>\n",
       "      <th>V17</th>\n",
       "      <th>V18</th>\n",
       "      <th>V19</th>\n",
       "      <th>V20</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Amount</th>\n",
       "      <th>Class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>-1.359807</td>\n",
       "      <td>-0.072781</td>\n",
       "      <td>2.536347</td>\n",
       "      <td>1.378155</td>\n",
       "      <td>-0.338321</td>\n",
       "      <td>0.462388</td>\n",
       "      <td>0.239599</td>\n",
       "      <td>0.098698</td>\n",
       "      <td>0.363787</td>\n",
       "      <td>0.090794</td>\n",
       "      <td>-0.551600</td>\n",
       "      <td>-0.617801</td>\n",
       "      <td>-0.991390</td>\n",
       "      <td>-0.311169</td>\n",
       "      <td>1.468177</td>\n",
       "      <td>-0.470401</td>\n",
       "      <td>0.207971</td>\n",
       "      <td>0.025791</td>\n",
       "      <td>0.403993</td>\n",
       "      <td>0.251412</td>\n",
       "      <td>-0.018307</td>\n",
       "      <td>0.277838</td>\n",
       "      <td>-0.110474</td>\n",
       "      <td>0.066928</td>\n",
       "      <td>0.128539</td>\n",
       "      <td>-0.189115</td>\n",
       "      <td>0.133558</td>\n",
       "      <td>-0.021053</td>\n",
       "      <td>149.62</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>1.191857</td>\n",
       "      <td>0.266151</td>\n",
       "      <td>0.166480</td>\n",
       "      <td>0.448154</td>\n",
       "      <td>0.060018</td>\n",
       "      <td>-0.082361</td>\n",
       "      <td>-0.078803</td>\n",
       "      <td>0.085102</td>\n",
       "      <td>-0.255425</td>\n",
       "      <td>-0.166974</td>\n",
       "      <td>1.612727</td>\n",
       "      <td>1.065235</td>\n",
       "      <td>0.489095</td>\n",
       "      <td>-0.143772</td>\n",
       "      <td>0.635558</td>\n",
       "      <td>0.463917</td>\n",
       "      <td>-0.114805</td>\n",
       "      <td>-0.183361</td>\n",
       "      <td>-0.145783</td>\n",
       "      <td>-0.069083</td>\n",
       "      <td>-0.225775</td>\n",
       "      <td>-0.638672</td>\n",
       "      <td>0.101288</td>\n",
       "      <td>-0.339846</td>\n",
       "      <td>0.167170</td>\n",
       "      <td>0.125895</td>\n",
       "      <td>-0.008983</td>\n",
       "      <td>0.014724</td>\n",
       "      <td>2.69</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-1.358354</td>\n",
       "      <td>-1.340163</td>\n",
       "      <td>1.773209</td>\n",
       "      <td>0.379780</td>\n",
       "      <td>-0.503198</td>\n",
       "      <td>1.800499</td>\n",
       "      <td>0.791461</td>\n",
       "      <td>0.247676</td>\n",
       "      <td>-1.514654</td>\n",
       "      <td>0.207643</td>\n",
       "      <td>0.624501</td>\n",
       "      <td>0.066084</td>\n",
       "      <td>0.717293</td>\n",
       "      <td>-0.165946</td>\n",
       "      <td>2.345865</td>\n",
       "      <td>-2.890083</td>\n",
       "      <td>1.109969</td>\n",
       "      <td>-0.121359</td>\n",
       "      <td>-2.261857</td>\n",
       "      <td>0.524980</td>\n",
       "      <td>0.247998</td>\n",
       "      <td>0.771679</td>\n",
       "      <td>0.909412</td>\n",
       "      <td>-0.689281</td>\n",
       "      <td>-0.327642</td>\n",
       "      <td>-0.139097</td>\n",
       "      <td>-0.055353</td>\n",
       "      <td>-0.059752</td>\n",
       "      <td>378.66</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1.0</td>\n",
       "      <td>-0.966272</td>\n",
       "      <td>-0.185226</td>\n",
       "      <td>1.792993</td>\n",
       "      <td>-0.863291</td>\n",
       "      <td>-0.010309</td>\n",
       "      <td>1.247203</td>\n",
       "      <td>0.237609</td>\n",
       "      <td>0.377436</td>\n",
       "      <td>-1.387024</td>\n",
       "      <td>-0.054952</td>\n",
       "      <td>-0.226487</td>\n",
       "      <td>0.178228</td>\n",
       "      <td>0.507757</td>\n",
       "      <td>-0.287924</td>\n",
       "      <td>-0.631418</td>\n",
       "      <td>-1.059647</td>\n",
       "      <td>-0.684093</td>\n",
       "      <td>1.965775</td>\n",
       "      <td>-1.232622</td>\n",
       "      <td>-0.208038</td>\n",
       "      <td>-0.108300</td>\n",
       "      <td>0.005274</td>\n",
       "      <td>-0.190321</td>\n",
       "      <td>-1.175575</td>\n",
       "      <td>0.647376</td>\n",
       "      <td>-0.221929</td>\n",
       "      <td>0.062723</td>\n",
       "      <td>0.061458</td>\n",
       "      <td>123.50</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2.0</td>\n",
       "      <td>-1.158233</td>\n",
       "      <td>0.877737</td>\n",
       "      <td>1.548718</td>\n",
       "      <td>0.403034</td>\n",
       "      <td>-0.407193</td>\n",
       "      <td>0.095921</td>\n",
       "      <td>0.592941</td>\n",
       "      <td>-0.270533</td>\n",
       "      <td>0.817739</td>\n",
       "      <td>0.753074</td>\n",
       "      <td>-0.822843</td>\n",
       "      <td>0.538196</td>\n",
       "      <td>1.345852</td>\n",
       "      <td>-1.119670</td>\n",
       "      <td>0.175121</td>\n",
       "      <td>-0.451449</td>\n",
       "      <td>-0.237033</td>\n",
       "      <td>-0.038195</td>\n",
       "      <td>0.803487</td>\n",
       "      <td>0.408542</td>\n",
       "      <td>-0.009431</td>\n",
       "      <td>0.798278</td>\n",
       "      <td>-0.137458</td>\n",
       "      <td>0.141267</td>\n",
       "      <td>-0.206010</td>\n",
       "      <td>0.502292</td>\n",
       "      <td>0.219422</td>\n",
       "      <td>0.215153</td>\n",
       "      <td>69.99</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Time        V1        V2        V3        V4        V5        V6        V7  \\\n",
       "0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   \n",
       "1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   \n",
       "2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   \n",
       "3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   \n",
       "4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   \n",
       "\n",
       "         V8        V9       V10       V11       V12       V13       V14  \\\n",
       "0  0.098698  0.363787  0.090794 -0.551600 -0.617801 -0.991390 -0.311169   \n",
       "1  0.085102 -0.255425 -0.166974  1.612727  1.065235  0.489095 -0.143772   \n",
       "2  0.247676 -1.514654  0.207643  0.624501  0.066084  0.717293 -0.165946   \n",
       "3  0.377436 -1.387024 -0.054952 -0.226487  0.178228  0.507757 -0.287924   \n",
       "4 -0.270533  0.817739  0.753074 -0.822843  0.538196  1.345852 -1.119670   \n",
       "\n",
       "        V15       V16       V17       V18       V19       V20       V21  \\\n",
       "0  1.468177 -0.470401  0.207971  0.025791  0.403993  0.251412 -0.018307   \n",
       "1  0.635558  0.463917 -0.114805 -0.183361 -0.145783 -0.069083 -0.225775   \n",
       "2  2.345865 -2.890083  1.109969 -0.121359 -2.261857  0.524980  0.247998   \n",
       "3 -0.631418 -1.059647 -0.684093  1.965775 -1.232622 -0.208038 -0.108300   \n",
       "4  0.175121 -0.451449 -0.237033 -0.038195  0.803487  0.408542 -0.009431   \n",
       "\n",
       "        V22       V23       V24       V25       V26       V27       V28  \\\n",
       "0  0.277838 -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053   \n",
       "1 -0.638672  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724   \n",
       "2  0.771679  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752   \n",
       "3  0.005274 -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458   \n",
       "4  0.798278 -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   \n",
       "\n",
       "   Amount  Class  \n",
       "0  149.62      0  \n",
       "1    2.69      0  \n",
       "2  378.66      0  \n",
       "3  123.50      0  \n",
       "4   69.99      0  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"creditcard.csv\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0fc59750",
   "metadata": {},
   "source": [
    "#### Checking the target feature, class as Fraud or not fraud"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7092f877",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZYAAAFZCAYAAAC7Rgn3AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAApKUlEQVR4nO3dfVjVdZ7/8ecJCwlvcPCAEqFLEqJrsWOJ3egYUEY3mkVi43bDZqLM5i47KJJBN8OMqNnoFjnd4JUzl02O1M6FDeJaQmqRx2tbxEtcxPVK0Qzi5EEgQed4fn/087tzxDvoo9y9HtfFdcn38z7f8z5fz3W9+Hy/n+85NpfL5UFERMSQqzq7ARER6VkULCIiYpSCRUREjFKwiIiIUQoWERExSsEiIiJGKVhERMQoBYuIiBilYBEREaMULCIiYpSCRUREjFKwiIiIUQoWERExSsEiIiJGKVhERMQoBYuIiBilYBEREaMULCIiYpSCRUREjFKwiIiIUX06uwE5N/fGNZ3dglxhPglPdnYLIkZoxiIiIkYpWERExCgFi4iIGKVgERERoxQsIiJilIJFRESMUrCIiIhRChYRETFKwSIiIkYpWERExCgFi4iIGKVgERERoxQsIiJilIJFRESMUrCIiIhRChYRETFKwSIiIkYpWERExCgFi4iIGKVgERERoxQsIiJilIJFRESMUrCIiIhRChYRETFKwSIiIkYpWERExCgFi4iIGKVgERERozotWF599VXuuusurr/+em644QaSkpKorKz0qpk7dy4BAQFeP/Hx8V41ra2tzJ8/n/DwcEJCQpgxYwZHjhzxqnG5XMyePZuwsDDCwsKYPXs2LpfLq6ampoakpCRCQkIIDw9nwYIFnDx50qtmz5493HfffQwZMoSoqCiWLFmCx+Mxd1BERHqATguW7du38/TTT7Np0yYKCwvp06cPDz30EMeOHfOqmzRpElVVVdbP+vXrvcYzMzPZsGED+fn5FBUV0djYSFJSEm6326qZNWsWFRUVrF+/noKCAioqKkhJSbHG3W43SUlJNDU1UVRURH5+PoWFhSxatMiqOX78ONOmTSMoKIgtW7aQm5vLa6+9xuuvv36ZjpCISPdkc7lcXeJP7qamJsLCwli7di0JCQnADzOW7777jnXr1p3zMQ0NDYwYMYK8vDymT58OwOHDhxkzZgwFBQXExcVRVVVFTEwMxcXFjB8/HoCysjISEhLYuXMnERERbN68menTp7N7925CQ0MBWLduHfPmzaO6upoBAwaQn5/Piy++yL59+/Dz8wNg2bJlrF69msrKSmw2m9Hj4d64xuj+pOvzSXiys1sQMaLLXGNpamri9OnTBAQEeG0vKytjxIgRjB07lnnz5vHtt99aY+Xl5Zw6dYrY2FhrW2hoKJGRkezYsQMAh8NBv379iImJsWrGjx+Pv7+/V01kZKQVKgBxcXG0trZSXl5u1dx2221WqJypOXr0KAcPHjR2HEREurs+nd3AGQsXLmTMmDGMGzfO2hYfH8+DDz7IsGHDOHToEDk5OUyZMoXS0lJ8fX2pq6vDx8eHwMBAr33Z7Xbq6uoAqKurIzAw0GtGYbPZGDx4sFeN3W732kdgYCA+Pj5eNSEhIW2e58zY8OHDz/m6qqurO3A0ILxDj5LurKPvFZErLSIi4oLjXSJYnnvuOb744guKi4vx8fGxtj/yyCPWv0ePHk10dDRjxoxh06ZNTJky5bz783g8bYKkIzVnbz+75syF+wudBrvYf8D5uPd/3qHHSffV0feKSFfT6afCMjMz+eCDDygsLDzvX/1nDB06lJCQEA4cOABAUFAQbrcbp9PpVVdfX2/NJoKCgqivr/daveXxeHA6nV41Z2YmZzidTtxu9wVr6uvrAdrMdkREerNODZaMjAwKCgooLCzkxhtvvGi90+nk6NGjBAcHAxAdHc3VV19NSUmJVXPkyBHrgj3AuHHjaGpqwuFwWDUOh4Pm5mavmqqqKq9lyiUlJfj6+hIdHW3VlJWV0dLS4lUzdOhQhg0b1vGDICLSw3RasKSnp/Pee+/xzjvvEBAQQG1tLbW1tTQ1NQE/XMx//vnncTgcHDx4kG3btjFjxgzsdjsPPPAAAAMHDuTxxx8nOzub0tJSdu3aRUpKCqNHj2bSpEkAREZGEh8fT1paGjt37sThcJCWlsbkyZOtUw+xsbFERUUxZ84cdu3aRWlpKdnZ2TzxxBMMGDAAgMTERPz8/EhNTaWyspLCwkJWrFhBamqq8RVhIiLdWactNz579dcZGRkZZGZmcuLECWbOnElFRQUNDQ0EBwczYcIEFi1a5LV6q6WlhaysLAoKCmhpaWHixIksX77cq+bYsWNkZGSwceNGABISEli6dKlXDzU1NaSnp7N161b69u1LYmIiOTk5+Pr6WjV79uwhPT2dL7/8koCAAJKTk8nIyLgswaLlxr2PlhtLT9Fl7mMRbwqW3kfBIj1Fp1+8FxGRnkXBIiIiRilYRETEKAWLiIgYpWARERGjFCwiImKUgkVERIxSsIiIiFEKFhERMUrBIiIiRilYRETEKAWLiIgYpWARERGjFCwiImKUgkVERIxSsIiIiFEKFhERMUrBIiIiRilYRETEKAWLiIgYpWARERGjFCwiImKUgkVERIxSsIiIiFEKFhERMUrBIiIiRilYRETEKAWLiIgYpWARERGjFCwiImKUgkVERIxSsIiIiFEKFhERMUrBIiIiRilYRETEKAWLiIgYpWARERGjFCwiImJUpwXLq6++yl133cX111/PDTfcQFJSEpWVlV41Ho+HxYsXM3LkSIYMGcL999/P3r17vWpaW1uZP38+4eHhhISEMGPGDI4cOeJV43K5mD17NmFhYYSFhTF79mxcLpdXTU1NDUlJSYSEhBAeHs6CBQs4efKkV82ePXu47777GDJkCFFRUSxZsgSPx2PuoIiI9ACdFizbt2/n6aefZtOmTRQWFtKnTx8eeughjh07ZtWsXLmSvLw8lixZwpYtW7Db7UybNo3GxkarJjMzkw0bNpCfn09RURGNjY0kJSXhdrutmlmzZlFRUcH69espKCigoqKClJQUa9ztdpOUlERTUxNFRUXk5+dTWFjIokWLrJrjx48zbdo0goKC2LJlC7m5ubz22mu8/vrrl/lIiYh0LzaXy9Ul/uRuamoiLCyMtWvXkpCQgMfjYeTIkTzzzDOkp6cDcOLECSIiIvjVr35FcnIyDQ0NjBgxgry8PKZPnw7A4cOHGTNmDAUFBcTFxVFVVUVMTAzFxcWMHz8egLKyMhISEti5cycRERFs3ryZ6dOns3v3bkJDQwFYt24d8+bNo7q6mgEDBpCfn8+LL77Ivn378PPzA2DZsmWsXr2ayspKbDab0ePh3rjG6P6k6/NJeLKzWxAxostcY2lqauL06dMEBAQAcPDgQWpra4mNjbVq/Pz8uP3229mxYwcA5eXlnDp1yqsmNDSUyMhIq8bhcNCvXz9iYmKsmvHjx+Pv7+9VExkZaYUKQFxcHK2trZSXl1s1t912mxUqZ2qOHj3KwYMHzR4MEZFurE9nN3DGwoULGTNmDOPGjQOgtrYWALvd7lVnt9s5evQoAHV1dfj4+BAYGNimpq6uzqoJDAz0mlHYbDYGDx7sVXP28wQGBuLj4+NVExIS0uZ5zowNHz78nK+rurr60g7AWcI79Cjpzjr6XhG50iIiIi443iWC5bnnnuOLL76guLgYHx8fr7GzTzF5PJ6LnnY6u+Zc9ZdSc/b2c/VyocfCxf8Dzse9//MOPU66r46+V0S6mk4/FZaZmckHH3xAYWGh11/9wcHBANaM4Yz6+nprphAUFITb7cbpdF6wpr6+3mv1lsfjwel0etWc/TxOpxO3233Bmvr6eqDtrEpEpDfr1GDJyMigoKCAwsJCbrzxRq+xYcOGERwcTElJibWtpaWFsrIy63pJdHQ0V199tVfNkSNHrAv2AOPGjaOpqQmHw2HVOBwOmpubvWqqqqq8limXlJTg6+tLdHS0VVNWVkZLS4tXzdChQxk2bJihIyIi0v11WrCkp6fz3nvv8c477xAQEEBtbS21tbU0NTUBP5xemjt3LitWrKCwsJDKykpSU1Px9/cnMTERgIEDB/L444+TnZ1NaWkpu3btIiUlhdGjRzNp0iQAIiMjiY+PJy0tjZ07d+JwOEhLS2Py5MnWqYfY2FiioqKYM2cOu3btorS0lOzsbJ544gkGDBgAQGJiIn5+fqSmplJZWUlhYSErVqwgNTXV+IowEZHurNOWG59Z/XW2jIwMMjMzgR9OWeXm5vLuu+/icrkYO3Ysr7zyCqNGjbLqW1payMrKoqCggJaWFiZOnMjy5cu9VngdO3aMjIwMNm7cCEBCQgJLly716qGmpob09HS2bt1K3759SUxMJCcnB19fX6tmz549pKen8+WXXxIQEEBycjIZGRmXJVi03Lj30XJj6Sm6zH0s4k3B0vsoWKSn6PSL9yIi0rMoWERExCgFi4iIGKVgERERoxQsIiJilIJFRESMUrCIiIhRChYRETFKwSIiIkYpWERExCgFi4iIGKVgERERoxQsIiJilIJFRESMUrCIiIhRChYRETFKwSIiIkYpWERExCgFi4iIGKVgERERoxQsIiJilIJFRESMalew3HzzzRQVFZ13vLi4mJtvvvlHNyUiIt1Xu4Ll0KFDNDc3n3e8ubmZmpqaH92UiIh0X+0+FWaz2c47tn//fvr37/+jGhIRke6tz8UK3nvvPf74xz9av7/yyiusWbOmTZ3L5aKyspLJkyeb7VBERLqViwZLc3MztbW11u8NDQ2cPn3aq8Zms3Httdfy5JNPsnDhQvNdiohIt2FzuVyeSy2+6aabyM3N5b777rucPQng3th2Vig9m0/Ck53dgogRF52x/K2KiorL1YeIiPQQ7QqWMxobGzl8+DDHjh3D42k74bnjjjt+dGMiItI9tStYjh07RkZGBv/xH/+B2+1uM+7xeLDZbHz33XfGGhQRke6lXcGSlpbGRx99xDPPPMMdd9xBQEDAZWpLRES6q3YFy8cff0xKSgq//vWvL1c/IiLSzbXrBslrrrmGG2644XL1IiIiPUC7gmXq1Kls3rz5cvUiIiI9QLuC5dlnn+Wbb75hzpw57Ny5k2+++YZvv/22zY+IiPRe7bpBctCgQdhsNmv11/loVdiPpxskex/dICk9Rbsu3i9YsOCCgSIiItKuYMnMzDT65J999hmvvfYau3bt4ujRo+Tl5TFz5kxrfO7cuV4fgAlwyy238PHHH1u/t7a28vzzz/PBBx/Q0tLCxIkTWb58Odddd51V43K5WLBgAcXFxQDce++9LF261Gu5dE1NDenp6Wzbto2+ffuSmJhITk4O11xzjVWzZ88e5s+fz5dffsmgQYN46qmnFLYiImfp1G+QbG5uZtSoUeTm5uLn53fOmkmTJlFVVWX9rF+/3ms8MzOTDRs2kJ+fT1FREY2NjSQlJXndwDlr1iwqKipYv349BQUFVFRUkJKSYo273W6SkpJoamqiqKiI/Px8CgsLWbRokVVz/Phxpk2bRlBQEFu2bCE3N5fXXnuN119/3fBRERHp3to1Y1myZMlFa2w2GwsWLLik/d1zzz3cc889AKSmpp6zxtfXl+Dg4HOONTQ08Ic//IG8vDzuuusuAN58803GjBlDaWkpcXFxVFVV8fHHH1NcXExMTAwAv/3tb0lISKC6upqIiAi2bNnC3r172b17N6GhoQC89NJLzJs3j6ysLAYMGMD69es5ceIEq1atws/Pj1GjRrFv3z7eeOMN/vmf/1mzFhGR/69dwZKbm3vesb+9qH+pwXIpysrKGDFiBAMHDuSOO+4gKysLu90OQHl5OadOnSI2NtaqDw0NJTIykh07dhAXF4fD4aBfv35WqACMHz8ef39/duzYQUREBA6Hg8jISCtUAOLi4mhtbaW8vJyJEyficDi47bbbvGZWcXFx/PrXv+bgwYMMHz7c2GsWEenO2v1ZYWc7ffo0hw4d4s0332THjh0UFBQYay4+Pp4HH3yQYcOGcejQIXJycpgyZQqlpaX4+vpSV1eHj48PgYGBXo+z2+3U1dUBUFdXR2BgoNeMwmazMXjwYK+aM2F1RmBgID4+Pl41ISEhbZ7nzNj5gqW6urpDrz28Q4+S7qyj7xWRKy0iIuKC4x36dOO/ddVVVzF8+HAWL15McnIyCxcu5K233vqxuwXgkUcesf49evRooqOjGTNmDJs2bWLKlCnnfdzZy6HPdZrqUmrO3n52zZlPdr7QabCL/Qecj3v/5x16nHRfHX2viHQ1Ri/eT5gwgU2bNpncpZehQ4cSEhLCgQMHAAgKCsLtduN0Or3q6uvrrdlEUFAQ9fX1Xh/v7/F4cDqdXjVnZiZnOJ1O3G73BWvq6+sB2sx2RER6M6PBUl1dfc7vZzHF6XRy9OhR62J+dHQ0V199NSUlJVbNkSNHqKqqsq6pjBs3jqamJhwOh1XjcDhobm72qqmqquLIkSNWTUlJCb6+vkRHR1s1ZWVltLS0eNUMHTqUYcOGXbbXLCLS3bTrVNhnn312zu0NDQ1s27aNt99+m4ceeuiS99fU1GTNPk6fPs3hw4epqKhg0KBBDBo0iNzcXKZMmUJwcDCHDh3i5Zdfxm6388ADDwAwcOBAHn/8cbKzs7Hb7QwaNIhFixYxevRoJk2aBEBkZCTx8fGkpaWxcuVKPB4PaWlpTJ482Tr1EBsbS1RUFHPmzCEnJ4djx46RnZ3NE088wYABAwBITExkyZIlpKamkp6ezv79+1mxYoXuYxEROUuHPtLlbB6PBx8fHx555BGWLFlyyd/Tsm3bNh588ME22x977DFeffVVZs6cSUVFBQ0NDQQHBzNhwgQWLVrktXqrpaWFrKwsCgoKvG6Q/NuaM19QtnHjRgASEhLOe4Pk1q1bvW6Q9PX1tWr27NlDeno6X375JQEBASQnJ5ORkXFZgkUf6dL76CNdpKdoV7Bs37697Q5sNgICAggLC6N///5Gm+vNFCy9j4JFeop2nQq78847L1cfIiLSQ3RouXFjYyPbt2/n0KFDAISFhXHnnXdqxiIiIu0PljfffJOcnByam5u9VoD5+/uTlZXl9RlcIiLS+7QrWN5//30WLlzI2LFjmTt3LpGRkXg8Hvbt28fvfvc7MjMzGTRoENOnT79c/YqISBfXrov3EyZMwN/fn48++og+fbwz6a9//SsPPPAAzc3NbNu2zXijvY0u3vc+ungvPUW7bpCsrq7m4YcfbhMqAH369OHhhx9m//79xpoTEZHup13B4u/vT21t7XnHa2trufbaa390UyIi0n21K1hiY2N58803z3mqa/v27bz11lvExcUZa05ERLqfdl1jOXz4MJMnT+bo0aPcdNNN3HjjjQDs27ePiooKhg4dyn/+5396fS2wdIyusfQ+usYiPUW7ZiyhoaFs27aN1NRUvv/+ewoLCyksLOT777/nF7/4Bdu2bVOoiIj0cu2asTQ3N/Pdd99x/fXXn3O8pqaGwMBAXWcxQDOW3kczFukp2jVjee655/j5z39+3vGZM2eSlZX1o5sSEZHuq13BUlJSYn1k/bk88MADfPLJJz+6KRER6b7aFSy1tbUMGTLkvOPBwcF88803P7opERHpvtoVLIMHD2bv3r3nHd+7dy8DBw780U2JiEj31a5gufvuu1mzZg07duxoM7Zz507WrFnD3Xffbaw5ERHpftq1Kqy2tpbY2Fi++eYb4uPjGTVqFDabjT179vDxxx8THBzMJ598wtChQy9nz72CVoX1PloVJj1Fu4IFoK6ujhdeeIG//OUvNDY2AtC/f38eeOABXnjhBYKDgy9Lo72NgqX3UbBIT9Hu72MJCgpi1apVeDwe6uvr8Xg82O32y/K97yIi0v106Bsk4Yfvurfb7SZ7ERGRHqBdF+9FREQuRsEiIiJGKVhERMQoBYuIiBilYBEREaMULCIiYpSCRUREjFKwiIiIUQoWERExSsEiIiJGKVhERMQoBYuIiBilYBEREaMULCIiYpSCRUREjFKwiIiIUQoWERExSsEiIiJGdWqwfPbZZ8yYMYOoqCgCAgJYu3at17jH42Hx4sWMHDmSIUOGcP/997N3716vmtbWVubPn094eDghISHMmDGDI0eOeNW4XC5mz55NWFgYYWFhzJ49G5fL5VVTU1NDUlISISEhhIeHs2DBAk6ePOlVs2fPHu677z6GDBlCVFQUS5YswePxmDsgIiI9QKcGS3NzM6NGjSI3Nxc/P7824ytXriQvL48lS5awZcsW7HY706ZNo7Gx0arJzMxkw4YN5OfnU1RURGNjI0lJSbjdbqtm1qxZVFRUsH79egoKCqioqCAlJcUad7vdJCUl0dTURFFREfn5+RQWFrJo0SKr5vjx40ybNo2goCC2bNlCbm4ur732Gq+//vplOjoiIt2TzeVydYk/ua+77jqWLl3KzJkzgR9mKyNHjuSZZ54hPT0dgBMnThAREcGvfvUrkpOTaWhoYMSIEeTl5TF9+nQADh8+zJgxYygoKCAuLo6qqipiYmIoLi5m/PjxAJSVlZGQkMDOnTuJiIhg8+bNTJ8+nd27dxMaGgrAunXrmDdvHtXV1QwYMID8/HxefPFF9u3bZ4XgsmXLWL16NZWVldhsNqPHw71xjdH9Sdfnk/BkZ7cgYkSXvcZy8OBBamtriY2Ntbb5+flx++23s2PHDgDKy8s5deqUV01oaCiRkZFWjcPhoF+/fsTExFg148ePx9/f36smMjLSChWAuLg4WltbKS8vt2puu+02r5lVXFwcR48e5eDBg+YPgIhIN9Wnsxs4n9raWgDsdrvXdrvdztGjRwGoq6vDx8eHwMDANjV1dXVWTWBgoNeMwmazMXjwYK+as58nMDAQHx8fr5qQkJA2z3NmbPjw4ed8HdXV1Zf8mv9WeIceJd1ZR98rIldaRETEBce7bLCccfYpJo/Hc9HTTmfXnKv+UmrO3n6uXi70WLj4f8D5uPd/3qHHSffV0feKSFfTZU+FBQcHA1gzhjPq6+utmUJQUBButxun03nBmvr6eq/VWx6PB6fT6VVz9vM4nU7cbvcFa+rr64G2syoRkd6sywbLsGHDCA4OpqSkxNrW0tJCWVmZdb0kOjqaq6++2qvmyJEj1gV7gHHjxtHU1ITD4bBqHA4Hzc3NXjVVVVVey5RLSkrw9fUlOjraqikrK6OlpcWrZujQoQwbNsz8ARAR6aY6NViampqoqKigoqKC06dPc/jwYSoqKqipqcFmszF37lxWrFhBYWEhlZWVpKam4u/vT2JiIgADBw7k8ccfJzs7m9LSUnbt2kVKSgqjR49m0qRJAERGRhIfH09aWho7d+7E4XCQlpbG5MmTrVMPsbGxREVFMWfOHHbt2kVpaSnZ2dk88cQTDBgwAIDExET8/PxITU2lsrKSwsJCVqxYQWpqqvEVYSIi3VmnLjfetm0bDz74YJvtjz32GKtWrcLj8ZCbm8u7776Ly+Vi7NixvPLKK4waNcqqbWlpISsri4KCAlpaWpg4cSLLly/3WuF17NgxMjIy2LhxIwAJCQksXbqUgIAAq6ampob09HS2bt1K3759SUxMJCcnB19fX6tmz549pKen8+WXXxIQEEBycjIZGRmXJVi03Lj30XJj6Sm6zH0s4k3B0vsoWKSn6LLXWEREpHtSsIiIiFEKFhERMUrBIiIiRilYRETEKAWLiIgYpWARERGjFCwiImKUgkVERIxSsIiIiFEKFhERMUrBIiIiRilYRETEKAWLiIgYpWARERGjFCwiImKUgkVERIxSsIiIiFEKFhERMUrBIiIiRilYRETEKAWLiIgYpWARERGjFCwiImKUgkVERIxSsIiIiFEKFhERMUrBIiIiRilYRETEKAWLiIgYpWARERGjFCwiImKUgkVERIxSsIiIiFEKFhERMUrBIiIiRilYRETEqC4dLIsXLyYgIMDr58Ybb7TGPR4PixcvZuTIkQwZMoT777+fvXv3eu2jtbWV+fPnEx4eTkhICDNmzODIkSNeNS6Xi9mzZxMWFkZYWBizZ8/G5XJ51dTU1JCUlERISAjh4eEsWLCAkydPXrbXLiLSXXXpYAGIiIigqqrK+vn888+tsZUrV5KXl8eSJUvYsmULdrudadOm0djYaNVkZmayYcMG8vPzKSoqorGxkaSkJNxut1Uza9YsKioqWL9+PQUFBVRUVJCSkmKNu91ukpKSaGpqoqioiPz8fAoLC1m0aNGVOQgiIt1In85u4GL69OlDcHBwm+0ej4dVq1bxr//6r0ydOhWAVatWERERQUFBAcnJyTQ0NPCHP/yBvLw87rrrLgDefPNNxowZQ2lpKXFxcVRVVfHxxx9TXFxMTEwMAL/97W9JSEigurqaiIgItmzZwt69e9m9ezehoaEAvPTSS8ybN4+srCwGDBhwhY6GiEjX1+VnLF999RVRUVHcdNNN/NM//RNfffUVAAcPHqS2tpbY2Fir1s/Pj9tvv50dO3YAUF5ezqlTp7xqQkNDiYyMtGocDgf9+vWzQgVg/Pjx+Pv7e9VERkZaoQIQFxdHa2sr5eXll+uli4h0S116xnLLLbfwxhtvEBERQX19PcuWLeOee+7hiy++oLa2FgC73e71GLvdztGjRwGoq6vDx8eHwMDANjV1dXVWTWBgIDabzRq32WwMHjzYq+bs5wkMDMTHx8eqOZ/q6uoOvHII79CjpDvr6HtF5EqLiIi44HiXDpa7777b6/dbbrmF6Oho3nvvPW699VYAr0CAH06Rnb3tbGfXnKv+UmoutP2Mi/0HnI97/+cXL5IepaPvFZGupsufCvtb/fr1Y+TIkRw4cMC67nL2jKG+vt6aXQQFBeF2u3E6nResqa+vx+PxWOMejwen0+lVc/bzOJ1O3G53m5mMiEhv162CpaWlherqaoKDgxk2bBjBwcGUlJR4jZeVlVnXS6Kjo7n66qu9ao4cOUJVVZVVM27cOJqamnA4HFaNw+GgubnZq6aqqsprmXJJSQm+vr5ER0dfzpcsItLtdOlTYc8//zz33nsvoaGh1jWW77//nsceewybzcbcuXNZvnw5ERERjBgxgldeeQV/f38SExMBGDhwII8//jjZ2dnY7XYGDRrEokWLGD16NJMmTQIgMjKS+Ph40tLSWLlyJR6Ph7S0NCZPnmydmoiNjSUqKoo5c+aQk5PDsWPHyM7O5oknntCKMBGRs3TpYPn666+ZNWsWTqeTwYMHc8stt7B582bCwsIA+Jd/+RdOnDjB/PnzcblcjB07lg8//JD+/ftb+/jNb36Dj48PycnJtLS0MHHiRH73u9/h4+Nj1bz99ttkZGTw8MMPA5CQkMDSpUutcR8fH9atW0d6ejr33nsvffv2JTExkZycnCt0JEREug+by+XyXLxMrjT3xjWd3YJcYT4JT3Z2CyJGdKtrLCIi0vUpWERExCgFi4iIGKVgERERoxQsIiJilIJFRESMUrCIiIhRChYRETFKwSIiIkYpWERExCgFi4iIGKVgERERoxQsIiJilIJFRESMUrCIiIhRChYRETFKwSIiIkYpWERExCgFi4iIGKVgERERoxQsIiJilIJFRESMUrCIiIhRChYRETFKwSIiIkYpWERExCgFi4iIGKVgERERoxQsIiJilIJFRESMUrCIiIhRChYRETFKwSIiIkYpWERExCgFi4iIGKVgERERoxQsIiJilIJFRESMUrC00zvvvMNNN91EcHAwP/vZz/j88887uyURkS5FwdIOH374IQsXLuSXv/wlW7duZdy4cTz66KPU1NR0dmsiIl2GgqUd8vLy+PnPf86TTz5JZGQky5YtIzg4mNWrV3d2ayIiXUafzm6guzh58iTl5eU8++yzXttjY2PZsWOH8efzSXjS+D5FRK4EzVgukdPpxO12Y7fbvbbb7Xbq6uo6qSsRka5HwdJONpvN63ePx9Nmm4hIb6ZguUSBgYH4+Pi0mZ3U19e3mcWIiPRmCpZLdM011xAdHU1JSYnX9pKSEmJiYjqpKxGRrkcX79vhF7/4BSkpKYwdO5aYmBhWr17NN998Q3Jycme3JiLSZWjG0g4PP/wwixcvZtmyZUyYMIEvvviCP/3pT4SFhXV2az2Cbj6VK+Wzzz5jxowZREVFERAQwNq1azu7pR5FwdJOs2bNYvfu3dTV1fHpp59yxx13dHZLPYJuPpUrqbm5mVGjRpGbm4ufn19nt9Pj2Fwul6ezmxCJi4tj9OjR/Pu//7u17ac//SlTp07lhRde6MTOpKe77rrrWLp0KTNnzuzsVnoMzVik0525+TQ2NtZr++W6+VRELi8Fi3Q63Xwq0rMoWKTL0M2nIj2DgkU6nW4+FelZFCzS6XTzqUjPohskpUvQzadyJTU1NXHgwAEATp8+zeHDh6moqGDQoEFcf/31ndxd96flxtJlvPPOO6xcuZLa2lqioqL4zW9+o/uE5LLYtm0bDz74YJvtjz32GKtWreqEjnoWBYuIiBilaywiImKUgkVERIxSsIiIiFEKFhERMUrBIiIiRilYRETEKAWLSBc3d+5cxowZ09ltiFwy3Xkv0km+/fZb8vLyKC4u5tChQ3g8Hv7u7/6Oe+65hzlz5jBkyJDOblGkQxQsIp3gv//7v3n00UdpbGzkkUce4ZlnnuGqq65iz549rFmzhg0bNvBf//Vfnd2mSIcoWESuMJfLxcyZM7HZbJSWlhIVFeU1npWVxYoVKzqnOREDdI1F5Ap79913+frrr8nJyWkTKgADBw686Ncxr127lqlTp3LjjTcSFBTE2LFjWbFiBadPn/aqO3DgAE899RSRkZEEBwczevRonnzySb7++mur5tNPPyUhIYFhw4Zx3XXXccstt/DLX/7SzIuVXkkzFpErbOPGjfTt25dp06Z1eB9vv/02ERERxMfH4+fnR0lJCS+++CLHjx8nOzsbgFOnTvHwww/T0tLCrFmzCA4Opra2li1btvD1118TEhLC//zP/zB9+nRGjRrFwoULufbaa/nqq6/YtGmTqZcrvZA+hFLkChs+fDihoaFs3779kurnzp3L9u3b2b17t7Xt+++/59prr/Wqe/bZZ/nwww85cOAAvr6+7N69mwkTJrBmzRqmTp16zn2vWrWKzMxM/vd//5fAwMCOvyiRv6FTYSJXWGNjI/379/9R+zgTKm63G5fLhdPp5M4776S5uZnq6moA6zk++eQTmpubz7mfMzV/+ctf2pxGE+koBYvIFda/f38aGxt/1D7KyspISEhg6NChDB8+nBtuuIGUlBQAGhoagB9mRnPmzOH3v/89N9xwA1OnTuWNN97A6XRa+3nkkUeIiYlh3rx5jBgxgqeeeoo//elPnDp16kf1J72bgkXkCouMjGT//v2cPHmyQ4//6quvmDZtGg0NDSxevJh169bx5z//mZdeegnAa+aRm5tLWVkZCxYswO12k5WVxa233srevXsB8PPzY+PGjRQWFvKP//iPVFdXM3v2bOLi4jhx4sSPf7HSKylYRK6whIQEWlpa+POf/9yhxxcVFdHS0sL777/P008/zeTJk5k0aRIBAQHnrI+KiuLf/u3f+Oijj/j00085fvy417ckXnXVVUycOJGXX36Zzz77jOXLl1NRUcGGDRs61J+IgkXkCnvqqacICQnh+eefp6qqqs348ePHefnll8/7eB8fHwA8nv9bd9Pa2spbb73VZj9//etfvbZFRkbi5+eHy+UC4Lvvvmuz/5tvvhnAqhFpLy03FrnCAgICWLt2LY8++ig/+9nPSExM5Kc//al15/0HH3zAT37yE2vZ8Nni4uK45pprmDFjBk899RQnT57k/fff56qrvP9O3Lp1K/Pnz2fKlClERETg8Xj48MMPrbv9AZYuXcr27duZPHkyYWFhuFwuVq9ejb+/P/fee+9lPxbSMylYRDrBP/zDP1BWVsbrr79OcXExH3zwAR6Ph/DwcJKTk60L8ecyYsQI1q5dy8svv8wLL7xAYGAgM2bM4M477/S6N+bv//7viY+PZ/Pmzfz+97/H19eXqKgo1q5dy/333w/Afffdx+HDh/njH/9IfX09P/nJT7j11ltZsGABYWFhl/04SM+k+1hERMQoXWMRERGjFCwiImKUgkVERIxSsIiIiFEKFhERMUrBIiIiRilYRETEKAWLiIgYpWARERGjFCwiImLU/wNT888JetXEIgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# apply the Fivethirtyeight style to plots.\n",
    "plt.style.use(\"fivethirtyeight\")\n",
    "\n",
    "# display a frequency distribution for churn.\n",
    "plt.figure(figsize=(5, 5))\n",
    "ax = sns.countplot(x=df['Class'], palette='Reds', linewidth=1)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "81b0eff5",
   "metadata": {},
   "source": [
    "The plot shows a class imbalance of the data between Class (fraud) and class (not-fraud). To address this, resampling would be a suitable approach. To keep this case simple, the imbalance is kept forward and specific metrics are chosen for model evaluations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "ef9f7aeb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    0.998273\n",
       "1    0.001727\n",
       "Name: Class, dtype: float64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Class'].value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a7818b98",
   "metadata": {},
   "source": [
    "It looks like they're losing 99.8% of their customers transactions are not fraudulent. Only 0.17% are.\n",
    "\n",
    "So our original dataset is very imbalanced. Since nearly all transactions are non-fraudulent, our predictive models might make assumptions. We don't want that. We want it to actually catch fraudulent transactions."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "975bb362",
   "metadata": {},
   "source": [
    "To handle class imbalance in this classification problem, several strategies can be employed:\n",
    "\n",
    "* Collect more data (not applicable in this case)\n",
    "* Change performance metric:\n",
    "* Precision, Recall, and F1 Score using confusion matrix\n",
    "* Kappa for normalized accuracy\n",
    "* ROC curves to calculate sensitivity/specificity ratio\n",
    "* Resample the dataset:\n",
    "* Over-sampling by adding copies of under-represented class (for little data)\n",
    "* Under-sampling by removing instances from over-represented class (for lots of data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "d027bfc4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>V1</th>\n",
       "      <th>V2</th>\n",
       "      <th>V3</th>\n",
       "      <th>V4</th>\n",
       "      <th>V5</th>\n",
       "      <th>V6</th>\n",
       "      <th>V7</th>\n",
       "      <th>V8</th>\n",
       "      <th>V9</th>\n",
       "      <th>V10</th>\n",
       "      <th>V11</th>\n",
       "      <th>V12</th>\n",
       "      <th>V13</th>\n",
       "      <th>V14</th>\n",
       "      <th>V15</th>\n",
       "      <th>V16</th>\n",
       "      <th>V17</th>\n",
       "      <th>V18</th>\n",
       "      <th>V19</th>\n",
       "      <th>V20</th>\n",
       "      <th>V21</th>\n",
       "      <th>V22</th>\n",
       "      <th>V23</th>\n",
       "      <th>V24</th>\n",
       "      <th>V25</th>\n",
       "      <th>V26</th>\n",
       "      <th>V27</th>\n",
       "      <th>V28</th>\n",
       "      <th>Class</th>\n",
       "      <th>normAmount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-1.359807</td>\n",
       "      <td>-0.072781</td>\n",
       "      <td>2.536347</td>\n",
       "      <td>1.378155</td>\n",
       "      <td>-0.338321</td>\n",
       "      <td>0.462388</td>\n",
       "      <td>0.239599</td>\n",
       "      <td>0.098698</td>\n",
       "      <td>0.363787</td>\n",
       "      <td>0.090794</td>\n",
       "      <td>-0.551600</td>\n",
       "      <td>-0.617801</td>\n",
       "      <td>-0.991390</td>\n",
       "      <td>-0.311169</td>\n",
       "      <td>1.468177</td>\n",
       "      <td>-0.470401</td>\n",
       "      <td>0.207971</td>\n",
       "      <td>0.025791</td>\n",
       "      <td>0.403993</td>\n",
       "      <td>0.251412</td>\n",
       "      <td>-0.018307</td>\n",
       "      <td>0.277838</td>\n",
       "      <td>-0.110474</td>\n",
       "      <td>0.066928</td>\n",
       "      <td>0.128539</td>\n",
       "      <td>-0.189115</td>\n",
       "      <td>0.133558</td>\n",
       "      <td>-0.021053</td>\n",
       "      <td>0</td>\n",
       "      <td>0.244964</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.191857</td>\n",
       "      <td>0.266151</td>\n",
       "      <td>0.166480</td>\n",
       "      <td>0.448154</td>\n",
       "      <td>0.060018</td>\n",
       "      <td>-0.082361</td>\n",
       "      <td>-0.078803</td>\n",
       "      <td>0.085102</td>\n",
       "      <td>-0.255425</td>\n",
       "      <td>-0.166974</td>\n",
       "      <td>1.612727</td>\n",
       "      <td>1.065235</td>\n",
       "      <td>0.489095</td>\n",
       "      <td>-0.143772</td>\n",
       "      <td>0.635558</td>\n",
       "      <td>0.463917</td>\n",
       "      <td>-0.114805</td>\n",
       "      <td>-0.183361</td>\n",
       "      <td>-0.145783</td>\n",
       "      <td>-0.069083</td>\n",
       "      <td>-0.225775</td>\n",
       "      <td>-0.638672</td>\n",
       "      <td>0.101288</td>\n",
       "      <td>-0.339846</td>\n",
       "      <td>0.167170</td>\n",
       "      <td>0.125895</td>\n",
       "      <td>-0.008983</td>\n",
       "      <td>0.014724</td>\n",
       "      <td>0</td>\n",
       "      <td>-0.342475</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.358354</td>\n",
       "      <td>-1.340163</td>\n",
       "      <td>1.773209</td>\n",
       "      <td>0.379780</td>\n",
       "      <td>-0.503198</td>\n",
       "      <td>1.800499</td>\n",
       "      <td>0.791461</td>\n",
       "      <td>0.247676</td>\n",
       "      <td>-1.514654</td>\n",
       "      <td>0.207643</td>\n",
       "      <td>0.624501</td>\n",
       "      <td>0.066084</td>\n",
       "      <td>0.717293</td>\n",
       "      <td>-0.165946</td>\n",
       "      <td>2.345865</td>\n",
       "      <td>-2.890083</td>\n",
       "      <td>1.109969</td>\n",
       "      <td>-0.121359</td>\n",
       "      <td>-2.261857</td>\n",
       "      <td>0.524980</td>\n",
       "      <td>0.247998</td>\n",
       "      <td>0.771679</td>\n",
       "      <td>0.909412</td>\n",
       "      <td>-0.689281</td>\n",
       "      <td>-0.327642</td>\n",
       "      <td>-0.139097</td>\n",
       "      <td>-0.055353</td>\n",
       "      <td>-0.059752</td>\n",
       "      <td>0</td>\n",
       "      <td>1.160686</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.966272</td>\n",
       "      <td>-0.185226</td>\n",
       "      <td>1.792993</td>\n",
       "      <td>-0.863291</td>\n",
       "      <td>-0.010309</td>\n",
       "      <td>1.247203</td>\n",
       "      <td>0.237609</td>\n",
       "      <td>0.377436</td>\n",
       "      <td>-1.387024</td>\n",
       "      <td>-0.054952</td>\n",
       "      <td>-0.226487</td>\n",
       "      <td>0.178228</td>\n",
       "      <td>0.507757</td>\n",
       "      <td>-0.287924</td>\n",
       "      <td>-0.631418</td>\n",
       "      <td>-1.059647</td>\n",
       "      <td>-0.684093</td>\n",
       "      <td>1.965775</td>\n",
       "      <td>-1.232622</td>\n",
       "      <td>-0.208038</td>\n",
       "      <td>-0.108300</td>\n",
       "      <td>0.005274</td>\n",
       "      <td>-0.190321</td>\n",
       "      <td>-1.175575</td>\n",
       "      <td>0.647376</td>\n",
       "      <td>-0.221929</td>\n",
       "      <td>0.062723</td>\n",
       "      <td>0.061458</td>\n",
       "      <td>0</td>\n",
       "      <td>0.140534</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-1.158233</td>\n",
       "      <td>0.877737</td>\n",
       "      <td>1.548718</td>\n",
       "      <td>0.403034</td>\n",
       "      <td>-0.407193</td>\n",
       "      <td>0.095921</td>\n",
       "      <td>0.592941</td>\n",
       "      <td>-0.270533</td>\n",
       "      <td>0.817739</td>\n",
       "      <td>0.753074</td>\n",
       "      <td>-0.822843</td>\n",
       "      <td>0.538196</td>\n",
       "      <td>1.345852</td>\n",
       "      <td>-1.119670</td>\n",
       "      <td>0.175121</td>\n",
       "      <td>-0.451449</td>\n",
       "      <td>-0.237033</td>\n",
       "      <td>-0.038195</td>\n",
       "      <td>0.803487</td>\n",
       "      <td>0.408542</td>\n",
       "      <td>-0.009431</td>\n",
       "      <td>0.798278</td>\n",
       "      <td>-0.137458</td>\n",
       "      <td>0.141267</td>\n",
       "      <td>-0.206010</td>\n",
       "      <td>0.502292</td>\n",
       "      <td>0.219422</td>\n",
       "      <td>0.215153</td>\n",
       "      <td>0</td>\n",
       "      <td>-0.073403</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         V1        V2        V3        V4        V5        V6        V7  \\\n",
       "0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   \n",
       "1  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   \n",
       "2 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   \n",
       "3 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   \n",
       "4 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   \n",
       "\n",
       "         V8        V9       V10       V11       V12       V13       V14  \\\n",
       "0  0.098698  0.363787  0.090794 -0.551600 -0.617801 -0.991390 -0.311169   \n",
       "1  0.085102 -0.255425 -0.166974  1.612727  1.065235  0.489095 -0.143772   \n",
       "2  0.247676 -1.514654  0.207643  0.624501  0.066084  0.717293 -0.165946   \n",
       "3  0.377436 -1.387024 -0.054952 -0.226487  0.178228  0.507757 -0.287924   \n",
       "4 -0.270533  0.817739  0.753074 -0.822843  0.538196  1.345852 -1.119670   \n",
       "\n",
       "        V15       V16       V17       V18       V19       V20       V21  \\\n",
       "0  1.468177 -0.470401  0.207971  0.025791  0.403993  0.251412 -0.018307   \n",
       "1  0.635558  0.463917 -0.114805 -0.183361 -0.145783 -0.069083 -0.225775   \n",
       "2  2.345865 -2.890083  1.109969 -0.121359 -2.261857  0.524980  0.247998   \n",
       "3 -0.631418 -1.059647 -0.684093  1.965775 -1.232622 -0.208038 -0.108300   \n",
       "4  0.175121 -0.451449 -0.237033 -0.038195  0.803487  0.408542 -0.009431   \n",
       "\n",
       "        V22       V23       V24       V25       V26       V27       V28  \\\n",
       "0  0.277838 -0.110474  0.066928  0.128539 -0.189115  0.133558 -0.021053   \n",
       "1 -0.638672  0.101288 -0.339846  0.167170  0.125895 -0.008983  0.014724   \n",
       "2  0.771679  0.909412 -0.689281 -0.327642 -0.139097 -0.055353 -0.059752   \n",
       "3  0.005274 -0.190321 -1.175575  0.647376 -0.221929  0.062723  0.061458   \n",
       "4  0.798278 -0.137458  0.141267 -0.206010  0.502292  0.219422  0.215153   \n",
       "\n",
       "   Class  normAmount  \n",
       "0      0    0.244964  \n",
       "1      0   -0.342475  \n",
       "2      0    1.160686  \n",
       "3      0    0.140534  \n",
       "4      0   -0.073403  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "amount_col = df['Amount'].values.reshape(-1, 1)\n",
    "scaler = StandardScaler().fit(amount_col)\n",
    "df['normAmount'] = scaler.transform(amount_col)\n",
    "df.drop(['Time', 'Amount'], axis=1, inplace=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0339ed63",
   "metadata": {},
   "source": [
    "#### Assign X and Y without resampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "59624a73",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.loc[:, df.columns != 'Class']\n",
    "y = df.loc[:, df.columns == 'Class']"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7ad47da8",
   "metadata": {},
   "source": [
    "#### Undersample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8d090871",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Percentage of normal transactions: 50.00%\n",
      "Percentage of fraud transactions: 50.00%\n",
      "Total number of transactions in resampled data: 984\n"
     ]
    }
   ],
   "source": [
    "# Find the indices of the minority class\n",
    "fraud_indices = df[df.Class == 1].index\n",
    "\n",
    "# Find the indices of the majority class\n",
    "normal_indices = df[df.Class == 0].index\n",
    "\n",
    "# Sample an equal number of instances from the majority class as the minority class\n",
    "normal_sample = df.loc[np.random.choice(normal_indices, len(fraud_indices), replace=False), :]\n",
    "\n",
    "# Concatenate the minority class and the sampled majority class\n",
    "under_sample_data = pd.concat([df.loc[fraud_indices], normal_sample])\n",
    "\n",
    "# Split the data into X (features) and y (labels)\n",
    "X_undersample = under_sample_data.drop(\"Class\", axis=1)\n",
    "y_undersample = under_sample_data[\"Class\"]\n",
    "\n",
    "# Calculate class proportions\n",
    "normal_prop = len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data)\n",
    "fraud_prop = len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data)\n",
    "\n",
    "# Print class proportions\n",
    "print(\"Percentage of normal transactions: {:.2f}%\".format(normal_prop * 100))\n",
    "print(\"Percentage of fraud transactions: {:.2f}%\".format(fraud_prop * 100))\n",
    "print(\"Total number of transactions in resampled data: {}\".format(len(under_sample_data)))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1cb56512",
   "metadata": {},
   "source": [
    "#### Much better!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "74412b9f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABbQAAAUdCAYAAAAgscXxAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAEAAElEQVR4nOzde1iUdf7/8dcMIGIehkxABTUxCY90EqPCxIqNSMqWAHHooFuZutbamuVWWpvaakcz1jbbTQba6aCrZeQWYZhmta2HtgOVB5QNNZHJMyDD749+zXcJvMHAuQd8Pq7rvq74fD73zesekMa3bz63xeVy1QoAAAAAAAAAAB9nNTsAAAAAAAAAAABNQUEbAAAAAAAAANAqUNAGAAAAAAAAALQKFLQBAAAAAAAAAK0CBW0AAAAAAAAAQKtAQRsAAAAAAAAA0CpQ0AYAAPARNptNK1asaNY1cnNz1bNnzxZKdGqlpaVp4sSJZsc4aS31Gq9du1Y2m03l5eUtkMp3tabvSQAAAPg+CtoAAADNNHHiRKWlpZkdo02qra3V0qVLdeWVVyo8PFwRERGKj4/X008/rQMHDpgdr8kGDx6shQsX1hmLjY1VcXGxzjzzzFP6uXNzc2Wz2XT++efXm/vnP/8pm8120gXna665Rr///e+btHbMmDHatGnTSV0fAAAAOBEK2gAAAPBZt99+u6ZPn64rr7xSK1as0AcffKCZM2dq7dq1euONN37xdaurq+uNVVVVNSfqSWvXrp1CQ0NlsVhO+edq3769fvjhB33wwQd1xh0Oh8LDw0/Z562urlZQUJC6det2yj4HAAAATi8UtAEAAFrYTx3bTz31lPr3769evXpp1qxZcrvdmjt3rvr166f+/fvrqaeeqnfunj17dOONN6p79+4aNGiQnE5nnflZs2bpwgsvVFhYmAYPHqwHH3xQx44dO2GW7du3KyMjQ/3791ePHj0UHx+vt99+u86awYMHa/78+brrrrsUERGhAQMG6Jlnnqmz5sCBA/rd736nqKgohYaGatiwYVq2bJln/qOPPlJSUpK6d++u6Oho/e53v6vTQX3kyBFNnDhRPXv21DnnnKPHH3+80ddx+fLleuWVV/T8889r+vTpuuCCC9S7d28lJibqtdde0zXXXCNJcrvd+tOf/qSBAwcqJCREcXFxWrVqlec6JSUlstlseu2113TttdcqLCxMf/3rX+t8nQYMGKABAwZIkr777jvdeuut6t27t3r37q0bb7xRW7du/cWv8TXXXKNdu3bpgQcekM1mk81mk9TwliMrV65UXFycQkJCNHDgQC1YsEC1tbUn9bVqiJ+fn9LS0uRwODxj5eXlWr16tdLT0+us3b9/v8aPH68BAwYoLCxMw4cPr3PexIkTtW7dOv3lL3/x3E9JSYnnfv75z38qISFB3bp1U0FBQZ0tR2pra3XdddcpJSXFc1+HDh3S+eef3+SObwAAAJzeKGgDAACcAuvXr1dJSYnefPNNPfHEE3r66aeVmpqqqqoqvf3225oxY4ZmzZpVbyuGuXPn6uqrr9batWt1880364477tDGjRs98x06dNCzzz6rjz76SI8//riWLVumBQsWnDDHoUOHdOWVV2r58uX64IMPNHr0aNntdn399dd11j333HMaMGCA3n//fU2dOlUPPvigPv74Y0k/FiFTU1O1bt06LVq0SB999JEeffRRBQQESJI+//xzjRkzRldffbU++OAD5eTk6LPPPtPkyZM913/ggQe0Zs0aLV26VCtWrNCWLVu0fv16w9fwlVdeUb9+/TR69OgG538qDGdnZ2vhwoWaNWuW1q9fr2uuuUZ2u11btmyps3727NmaMGGCNmzY4CmGr1u3Tp9//rlee+01rVixQkeOHNG1116rwMBArVq1Su+8845CQ0OVkpKiI0eO/KLX2OFwqGfPnpo+fbqKi4tVXFzc4HU2bdqkm2++WcnJyVq/fr0eeughPfnkk3r++efrrDP6Whmx2+1auXKlDh48KEn6+9//rmHDhunss8+us+7YsWMaOnSo/v73v2vDhg264447dPfdd+v999+XJM2bN0/Dhg1TZmam537+t8t71qxZ+sMf/qBPPvlEF154YZ1rWywWZWdn67PPPvNswXLvvfeqXbt2evjhhxu9BwAAAMDf7AAAAABtUefOnbVgwQL5+fmpf//+evbZZ1VWVqbXX39dktSvXz89+eSTWrt2rWJiYjznXXvttbrlllskSffcc4/Wrl2r7OxsT1Fz+vTpnrW9e/fW7373Oy1cuFB/+MMfGswxePBgDR482PPxPffco7ffflsrVqyo0xGbkJCg2267TdKP23wsXrxY77//voYNG6Y1a9bo448/1oYNGxQVFSVJ6tOnj+fcZ555Rtdf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TW+88YYuvvhizZ49W3fddZfnAZAtpX379jrzzDNPOP/73/9e559/vlJTU5WUlKQOHTp4Hrj5kylTpqhdu3YaPnx4k35T4Cf79u3TnXfeqXvuuUcXXnihJOnuu+/Wueeeq0mTJjV56xQAAACcviwul4t3jQAAAAAAAAAAn0eHNgAAAAAAAACgVaCgDQAAAAAAAABoFShoAwAAAAAAAABaBQraAAAAAAAAAIBWgYI2AAAAAAAAAKBVoKANAADQisydO1eTJk0yXJOWlqaJEyd6KZFv+vrrr3XllVcqNDRUgwcPNjuOx4YNGxQXF6du3brpmmuuMTtOs7399tu69NJL5Xa7zY4CAACA0wQFbQAAgFbi+++/13PPPaff//73Xv/cubm5dQqwEydOVFpamtdzNNUf//hHBQUF6eOPP1ZhYaHZcTxmzJihQYMGaePGjXI4HGbHabZf/epX8vPz0yuvvGJ2FAAAAJwmKGgDAAC0EkuXLtX555+vPn36mB3F523btk3Dhw9X7969ddZZZ/2ia1RVVTVpndvtVk1NTZNzxcfHKzw8XMHBwac0l7dkZmZq8eLFZscAAADAaYKCNgAAQCvx2muv6eqrr64zduTIEU2cOFE9e/bUOeeco8cff7zeeS6XS3fccYd69+6tsLAwpaSk6Msvv/TM5+bmqmfPnnr//fd18cUXq0ePHkpOTtaOHTsazDF37ly9/PLLWr16tWw2m2w2m9auXdvg2oY6uefOnauLL77Y8/Hnn3+u0aNHKyIiQuHh4brkkktUVFTkmf/qq6904403Kjw8XP369dP48eO1Z8+eE75ONptN//nPf/SnP/1JNptNc+fO9XyelJQUhYWFqU+fPpo4caJ++OGHelmfeuopDRgwQAMGDGjw+j+9Xv/85z918cUXq1u3biouLlZVVZUeeughDRgwQD169NDIkSNVUFAgSSopKZHNZtOBAwc0efJk2Ww25ebmNun+TpTru+++06233qrevXurd+/euvHGG7V169Z6r/Prr7+umJgYhYeHa+zYsSovL69zP3l5eYqLi1NISIjOOeecOtvV/PDDD5o6dar69eun8PBwJSUlaePGjXXOv/rqq7Vx40Zt27bthF8TAAAAoKVQ0AYAAGgFKioq9NVXX+m8886rM/7AAw9ozZo1Wrp0qVasWKEtW7Zo/fr1ddZMnDhRn376qfLy8lRQUKCgoCD9+te/1tGjRz1rKisr9cQTT+jZZ5/VP//5T/3www/63e9+12CWKVOm6Prrr9fll1+u4uJiFRcXKzY29hff229+8xuFhYWpoKBARUVFmjFjhtq3by9J2r17t5KSkhQdHa2CggL94x//0KFDh5SRkXHCfZuLi4t1zjnnaPLkySouLtaUKVN05MgR/frXv9YZZ5yhgoICORwOffzxx5o8eXKdc9etW6fPP/9cr732mlasWHHCzMeOHdOCBQv05JNP6qOPPlJERIQmTZqkdevW6S9/+YvWr1+vjIwMpaen67PPPlN4eLiKi4vVoUMHzZ07V8XFxRozZkyT7+/nuY4cOaJrr71WgYGBWrVqld555x2FhoYqJSVFR44c8Zy3c+dOLVu2TA6HQ8uWLdOWLVv0yCOPeOb/+te/6u6779bYsWO1bt06vfrqq4qOjpYk1dbWKi0tTWVlZXI6nSoqKlJcXJxGjx6t3bt3e64RERGhkJAQffDBByfxVQcAAAB+GX+zAwAAAKBxu3btUm1trUJDQz1jhw4dUk5Ojp599lmNGjVKkrRo0aI6ncVbt25Vfn6+Vq1apUsuuUSStHjxYg0ePFivvvqqsrKyJEnHjx/XggULdM4550j6sWg9adIkud1uWa1WZWZmKjMzU5LUsWNHtW/fXoGBgXXyNOfeJk+erP79+0uS+vbt65lbsmSJBg0apNmzZ3vGFi9erD59+mjjxo264IIL6l0vNDRU/v7+OuOMMzz5XnrpJR0+fFiLFy9Wp06dJElPPfWUrr32Wm3bts3zOQMDA/Xss88qMDDQMHNNTY3+9Kc/KSYmRpK0fft2vfbaa9qyZYsiIiIkSbfddpvWrFmjv/3tb3r88ccVGhoqi8Wizp07e3I98cQTTbq/n+fKyclRbW2tnnvuOVksFs/99OvXT6tXr9b1118v6cev63PPPacuXbpIkm6++WZPZ7gkzZ8/XxMnTqxT2P/pnoqKivTZZ5/p22+/VVBQkCTpD3/4g95++205nU5NnTrVc05YWJh27txp+JoBAAAALYGCNgAAQCtw7NgxSfJ0Lks/FlGrqqo0bNgwz1jHjh01cOBAz8fFxcWyWq111nTp0kUDBgzQV1995RkLDAz0FLOlHwuU1dXV+uGHH37xXs9Ndeedd+q3v/2tXn75ZY0YMUKjR4/2FLc3b96s9evXq2fPnvXO2759e4MF7YYUFxdr4MCBnmK2JMXGxspqteqrr77yFLSjo6MbLWZLkr+/vwYPHuz5ePPmzaqtrdXw4cPrrKusrFR8fPwJr9PU+/t5rs2bN6ukpETh4eF1zjly5Ii2b9/u+TgiIsJTzJZ+/Lru27dP0o8PGf3uu+80YsSIE2Y7cuSI+vXrV2f82LFjdT6HJAUFBXm+RwEAAIBTiYI2AABAK9C1a1dJP+6HHRYWJunHLSEaY7Tmp85e6ccCbUNzJ9rWo6msVmu9DMePH6/z8X333acbb7xR77zzjt577z099thjeuKJJ2S32+V2u3XVVVfpj3/8Y71rd+vWrck5mvo6nHHGGU26XmBgoPz8/Dwfu91uWSwWvffeewoICKiz9n//EeLnmnp/P8/ldrs1ePBgvfjii/XO+99/gPh5FovF4vmaNvb943a7FRISovz8/Hpz//sPA9KPW+L80odvAgAAACeDgjYAAEArcPbZZ6tz584qLi7WueeeK+nHrTkCAgL0ySefqE+fPpKkw4cP64svvvB8fO6558rtduvjjz/2bDly4MABffHFFxo7duwvztOuXTvV1NQ0uu6ss87SZ599Vmfs5x9LUmRkpCIjI3XHHXfod7/7nXJycmS32zV06FAtX75cERER9YqzJ+Pcc89Vbm6uDh486CnGfvTRR3K73YqKivrF1/3JkCFDVFtbqz179hh2ZP/cL72/oUOH6rXXXtOZZ54pm832CxJLISEh6tGjh95//32NHDmywc+xd+9eWa1Wz/dTQ37q2B46dOgvygEAAACcDB4KCQAA0ApYrVaNGDFCH374oWesY8eOstvtmjVrlgoLC/Xll19q8uTJdbqqIyMjlZSUpLvvvlvr16/X559/rttuu02dOnVSamrqL87Tq1cvffnll/rmm29UXl6u6urqBtfFx8dry5YtysnJ0bZt2/T0009rw4YNnvmjR4/qnnvu0dq1a1VSUqJ//etf2rBhg6fIPGHCBB04cEC33HKL/vWvf2nHjh1as2aNpk6dqoMHDzY5b2pqqjp06KA77rhDn3/+udatW6e7775b1157bZ09u3+pfv366cYbb9Sdd96pFStWaMeOHdq4caMWLlyolStXnvC8X3p/qampCgkJ0dixY/XBBx9ox44dWrdunWbOnKmtW7c2Ofe0adOUnZ2tRYsW6dtvv9WWLVu0cOFCSdLll1+u4cOHa+zYsXrnnXe0Y8cOffzxx5ozZ06dB49+8sknCgwMbNaDQQEAAICmoqANAADQStx8881avnx5nc7oRx55RJdeeqnGjRuna6+9VtHR0YqLi6tz3nPPPafzzz9fGRkZGjVqlI4eParXXnvN86C/X+Kmm25S//79NXLkSEVGRtYpUv+vUaNG6d5779Uf//hHXX755dq5c6cmTJjgmffz85PL5dLEiRN10UUXady4cbrooov06KOPSpK6d++u1atXy2q16oYbbtDw4cN1zz33qF27dk3a6/onHTp00Ouvv66DBw9q1KhRGjt2rC666CI9++yzv/g1+LlFixYpMzNTDz74oC666CKlpaVp3bp16tWr1wnP+aX316FDB7311lvq06ePbr75Zg0bNkwTJ06Uy+U6qY7t8ePHa/78+Vq6dKkuvvhi/frXv/bsrW6xWPTKK6/osssu09SpU3XRRRfplltu0bfffqvu3bt7rvH66697/sEAAAAAONUsLper8c0XAQAA4BOuvPJKjR8/Xunp6WZHAbRv3z5ddNFFKiwsNNyWBAAAAGgpdGgDAAC0Ik8++WSzH9QItJSSkhI9/vjjFLMBAABakXXr1ik9PV3R0dGy2WzKzc1t9JzPP/9cSUlJCgsLU3R0tB577LEmPaT+VOChkAAAAK3IoEGDNGjQILNjAJKkCy64QBdccIHZMQAAAHASDh8+rAEDBigjI0N33HFHo+sPHDig66+/XnFxcXrvvff0zTffaNKkSerQoYOmTJnihcR1UdAGAAAAAAAAgNPEVVddpauuukqSdOeddza6/tVXX9XRo0eVnZ2toKAgDRgwQF9//bWee+45TZ48WRaL5VRHroMtRwAAAAAAAAAADfr444918cUX13mo/KhRo1RWVqaSkhKv56FD20ddM26T2REM5fV82uwIjQoZEWt2BENHBww3O4KhoG//bXYEQ+6wXmZHMGSprjI7giHLD+VmRzB27IjZCQxV7/3e7AiG/AfFmB3BkGX/XrMjGKrpcbbZEQz5Vfj26ydJ8g8wO4GxqmNmJzDkLvftnzHW9u3NjmCoJqKf2REMHXj9NbMjGOqScp3ZEQxZK337PYJ75zazIxg7hy2rmq3Wt59jYS3fY3YEY9WVZicwVPmr282OYCjg9SfMjmCo+obfmR2h1fBm3W+V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6dNH+/fvd3LZNTf7ZWJWVpZOnDhh2P8vTZo00fDhw/XYY49JuvbDQvfu3dWlSxeNGzfO8Nl1klS/fn3NmjVLzZs3V0pKiurWravExETrrM41a9botdde0/bt2w3pq1Onjho0aKABAwZYZ6hZLBZ17NhR7733nqpWrSpJuv/++w3p8/Hx0W+//aYKFSqodevW6ty5swYMGGDdv2DBAs2ePVvfffedIX3+/v7aunWrgoKCVLduXSUkJOiuu+6y7t+3b5/atGmjgwcPGtJ3/fkrWbKkli5dqgULFuiHH35QQECAYmJiFB0dbeg7pf7+GnzlyhVVqlRJX331lRo1aiRJ2rt3r9q2bWvY81enTh3NmzdPzZo1k3RtqYcmTZpo//798vT01MSJE/XNN9/oq6++MqSP1xDHuPpriKu/fki8hjiK1xDHuPpryN/5+/vrxx9/VEBAgPr27avAwECNHDlSR48eVXh4uI4fP25on4+Pj37//fccywT98ccfatWqlY4ePWpQGXBrWEO7EClbtqzRCTe1dOlSTZs2TQ899JCka2/H6dKli55++mnrb8yNXKNw06ZNWrhwoRo3bqymTZuqevXq+uyzz1SpUiVJ0vDhwzV48GDD+urWrauAgACNGzdOnp6ekq59I9moUSMtWbJE1atXN6ztest1n332md544w117NhRkhQYGKjMzExNmDDBsAHtrKws61t0f/nlF40aNcr6sclk0rPPPqvWrVsb0nbdc889p5EjR2rz5s2aP3++hg8frtdee02PPfaYnnjiCTVu3Niwtn379mnOnDlyd3fXU089pWHDhikmJsa6/8knn9T8+fMN61uxYoU+/fRT1a1bV5I0bdo01a5dW6+88opiYmKUkZGh9957z7AB7YyMDLv/T5ctW9bQdzCcPHlS3bt3v+HsqpSUFL3//vtOrvo/R48e1d133239OCwsTCtWrFD79u2VnZ2tF154wbC26/766y/r/8MVK1aUl5eX9S3Y0rXBgD///NOoPH333Xfq37+/pk6dqtmzZ1uXWDKZTGrUqJFq165tWNt1178HOHz4cI6Bu+bNm2vEiBFGZEmSatasqR9//FFBQUEqXbq00tLSbPanp6e7xDrLXl5eio6OVnR0tPbu3auPP/5YcXFxmjhxoiIiIpSUlGRI19+fm+t/vv4afP3PRs6QvXDhgvX7PUny8/NTRkaG0tLS5Ofnpw4dOujdd981rI/XEMe4+muIq79+SLyGOIrXEMe4+mvI3wUEBGjbtm3y8fHR2rVr9fHHH0u6tmyPl5eXYV3du3eXdO3569u3r81SqmazWb/++quaNGliVB5wyxjQLkS8vLzUr18/hYaG5rr/yJEjGjVqlJOrbJ08eVK1atWyfhwYGKgVK1aoQ4cOeuqpp/Tmm28aWHdtvefr61D7+PioePHiNut5V69eXSdOnDAqT+vWrdNrr72mJ554QnPmzFG9evWs+/z9/VWlShXD2q67/g3Gn3/+af2N+XV33323ob/xrV+/vjZu3Kjq1avL399fR44csZlRdOTIEZdZv/qBBx7QAw88oLNnzyoxMVHx8fFKSEhQnTp19P333xvSVLRoUeuSMpcvX5bZbNaVK1es+y9fvmzo2p1Xr161mcleokQJXb16VZcuXVLx4sXVunVrQ9+Off/992vEiBH64IMPcqx3n5KSopEjR+qBBx4wqO7aD8v16tXTM888k+v+PXv2GDoYUa5cOR07dsw6A0y69sPh8uXL1b59e/3111+GtV1XtmxZpaamKiAgQJLUtm1bm/VOL168aOh1GMqWLatFixYpLi5OrVq10sSJE23WkHUFq1evVunSpVWsWDFdvHjRZt/ly5cNW/tUkgYOHKiRI0eqQoUKGjJkiIYNG6aJEyeqVq1a+v333zVs2DBD36qb20BInTp19Pbbb2vMmDH67LPPtGDBAgPKrgkLC9M777yjYcOGaeHChQoKCtIHH3ygWbNmSZJmz56tOnXqGNZXt25dLV682PpL9yVLlqhEiRLWQTuz2Wzov19eQxzj6q8hrv76IfEa4iheQxzj6q8hfzdgwAA9++yzKlGihAIDA61rUn///ffWiTdGuP5LUYvFIm9vb5uLkBYtWlRNmzbVE088YVQecMsY0C5EQkND5ePjo6ioqFz3G7k28HV+fn46ePCgzTeTvr6++vzzz9W+fXs999xzBtZJ5cuX18mTJ63fTD7zzDM2FzZJT09XiRIljMqTp6enJk2apJUrV6pLly4aNGiQ+vfvb1hPbubNm6cSJUqoaNGiOd62ee7cOUO/GX/llVfUp08feXh4qF+/fnr11Vd19uxZhYSE6Pfff9eECROsv7k2Qm7fSPr4+GjAgAEaMGCAtmzZYug3kuHh4Ro1apQGDx6spKQk3XXXXZo0aZLmzZsnk8mkSZMm2bx10tnuvvtu6zI3kjRz5kyVL1/e+na6CxcuGPrvd8qUKXr88cdVv359hYSEqEKFCjKZTDp16pSSk5NVu3ZtLV682LC+8PBw/fHHHzfcX7JkSetb8Y3QtGlTrVixIseFamrVqmV9DTFa3bp1tWPHDusvyv65VufOnTttfqlrlH79+qlZs2Z65plnDFs+4UYGDRpk/fPmzZsVHh5u/Xj79u2GXSxLkrp166azZ8+qZ8+eMpvNys7Oti5fIElt2rTRW2+9ZVifvZlpnp6e6tatm7p16+bEIluvv/66unTpokWLFql8+fJasWKFBg4cqJo1a8pkMun8+fNatGiRYX0jRozQ448/rpUrV6pYsWL68ccfbdZYXrt2raEXPuY1xDGu/hpSUF4/JF5DbhevIY5x9deQv+vdu7fCwsJ07NgxtWrVyvqLlGrVqunVV181rOv64H+VKlU0aNAgQ38uAvIDa2gXIlOmTNGVK1du+FaqY8eO6a233rL+R2aEQYMGyWw2a+bMmTn2nThxQo8++qgOHjxo2Pp1PXr0UIsWLW44sD537lx9/vnnhq2x/Hd//vmn9a1Cmzdv1rfffmv4W/1CQ0NtBmWfe+45mwH3WbNmadmyZYatsSxJX3zxhYYNG6bjx4/bfOPm6emp3r17a9y4cTZvX3Omv6/954r279+vxx9/XAcOHFDt2rW1dOlSDRkyxPrDjI+Pj5YsWaKwsDBD+nbt2qWOHTvK3d1dRYoUUWpqquLi4tS5c2dJ0pw5c/S///3P0BliZrNZa9eu1fbt23Xq1ClJ136p16RJE7Vu3drQmUOu7ueff9bOnTvVq1evXPfv3btXn3/+uYYNG+bksv+TmpoqNzc3m1+E/t2aNWtUrFixG66B62yXLl3Syy+/rE2bNumzzz4zfNmqm1m9erWKFCmiiIgIQzvS09O1fv16HTp0SGazWX5+fmratKmhF0OTpMTERHXu3Nm6JJkrunjxon7//XfVqFFDJUuWVEZGhhYvXqyMjAy1atXKZokFI/z8889atmyZrly5ooiICLVq1crQnn/iNeT2ufprSEF7/ZB4DbldvIbcPld/DbEnKyvL0HeyAoURA9qFyMaNG13qm5zcLFiwQJUqVdKDDz6Y6/4TJ05o3bp16tmzp5PLrlm/fr3dH162b9+uYsWK3XBZl3/bP7/GZrNZkydP1qZNmzRr1izDlxy52d/B7du3q2jRoje9cNC/5Xqf2WzWzp07bb6RDAsLM/zCixs3btR9990nDw/XfPPM9efvzJkzNut4bty4UZcvX1aTJk0MXct/48aNCgkJ0Zo1a3TlyhU1b97c8F/yAAAAALhzvP/++6pYsaL1nfMDBw7Uf//7X1WrVk3//e9/DR94P3v2rMaOHauNGzfqr7/+yjE7n4tCoqBgQLsQ8fHxUZUqVRQTE6OePXvaXNjGVbh6I32O+XtfdHR0jjUejVaQnj9X73Plr2+vXr0UHR3tcs/fzVy8eFE7d+7M8XZoV0Gf41y9kT7H0OcY+hxjNpv1559/2lz7xZXQ5xiLxaJjx465bJ/Ec+goV3/+6Mu7u+66SzNmzNB9992n7777Tt26ddP06dO1fPlyXbp0ybALa14XHR2t3bt368knn5S/v3+OZS+NmlwI3Crel1aIbN26Ve3bt9cHH3ygBg0a6PHHH9cXX3yh7Oxso9OsXL2RPsf8vS80NNSl+1z9+XP1Plf++s6ZM8cln7+bOXDggOFreNpDn+NcvZE+x9DnGPrsy8jI0AsvvKDg4GA1atRIcXFxNvtPnz5t2DvgJPocdbO+v/76y9A+iefQUQX9+aMv71JSUqzvnF69erWioqL02GOPadiwYdq+fbvBddKmTZv00UcfaejQoYqOjlbPnj1tbkBBwYB2IRISEqJx48bp119/1YcffiiTyaQnn3xSderU0ahRo/T7778bnejyjfTRRx99AAC4mokTJ2rNmjUaMWKEevXqpcmTJ6tv374ym83WY+xd1I0++hzl6o300Wf0v5HrSpUqpdTUVEnXljS9viRnkSJFdOXKFSPTJEnly5fngpAoFFhypJBLSUlRYmKiEhISdOjQIYWHh2vVqlVGZ9lw9Ub6HEOfY+hzjCv15XV9caMuikuf41y9kT7H0OcY+hwTFhamSZMm6aGHHpJ0bY3TLl26qF69epo7d65Onz6t2rVr00ffHdtIH31G/xu5rm/fvtq3b58aNGigZcuW6eeff5aPj49WrlypN998U99//72hfUuXLtWyZcsUFxenkiVLGtoCOIIB7TtAWlqakpKSNGHCBKWnp7vEf/L/5OqN9DmGPsfQ5xhX6atcubL69et3w4vKHjlyRKNGjaLvBly9T3L9RvocQ59j6HNMxYoVtXXrVlWtWtW67dSpU+rQoYNq166tN998U6GhofTRd8c20kef0f9Grjt37pzGjh2rY8eOqU+fPnrwwQclSW+99ZY8PT01dOhQQ/uaNWumI0eOKDs7W4GBgfLw8LDZb/SAO5BXHjc/BAXVhg0bFB8fr5UrV8rT01NdunRRTEyM0Vk2XL2RPsfQ5xj6HONqfaGhofLx8bFe8fyf9uzZ4+QiW/Q5ztUb6XMMfY6hzzF+fn46ePCgzWCOr6+vPv/8c7Vv317PPfecgXX0OcrV+yTXb6TPMfTln9KlS2vSpEk5to8YMcKAmpw6dOhgdAKQLxjQLmSOHj2qhIQEJSYm6ujRo2rWrJneffddRUVFqVixYkbnSXL9Rvroo4++f8NDDz2k9PT0G+738fFR9+7dnVhkiz7HuXojfY6hzzH0OeaBBx7QJ598opYtW9ps9/Pz0/Lly/Xoo48aE/b/0ecYV++TXL+RPsfQ9+84efKkMjMzbbYFBgYaVHPNsGHDDP38QH5hyZFCpGPHjtq8ebMqVKigHj16KCYmRtWrVzc6y4arN9LnGPocQ59jXL1v48aN1ovCuCL6HOfqjfQ5hj7H0OeYBQsWqFKlSta3rv/TiRMntG7dOvXs2dPJZdfQ5xhX75Ncv5E+x9CXf9LT0/XKK6/os88+yzGYLRl7vRegMGFAuxDp3r27YmNjFRkZKXd3d6NzcuXqjfQ5hj7H0OcYV+/z8fFRlSpVFBMTo549e6pSpUpGJ9mgz3Gu3kifY+hzDH2Ooc8x9DnO1Rvpcwx9+ec///mPfvrpJ40ePVoxMTGaMWOGjh8/rvfff19vvvnmDZe2cpaAgACZTKYb7j969KgTa4Dbx4A2AABOkJycrIULF2rx4sU6c+aMWrdurdjYWLVp08YlBuDpc5yrN9JHH3300Uff7XL1RvrocxV169bV3Llz1axZMwUGBmrjxo2qXr26lixZovj4eH322WeG9iUmJtp8fPXqVe3evVvLly/X0KFD9eyzzxpUBtwaBrQBAHCiq1ev6ssvv1RCQoLWrl2rsmXLqkePHurVq5dq1qxpdB59+cDVG+mjjz766KPvdrl6I330Ga1y5craunWrAgMDVa9ePc2fP1/33HOPDh8+rHvvvVfHjx83OjFXCxYs0KZNmzR37lyjU4A8YUAbAACDpKSkKDExUQkJCTp06JDCw8O1atUqo7Os6HOcqzfS5xj6HEOfY+hzDH2Oc/VG+hxD3+257777NGHCBD3wwAN67LHHVLt2bb311luaOXOm4uLi9MsvvxidmKtDhw7p/vvv17Fjx4xOAfKEAW0AAAyUlpampKQkTZgwQenp6S53oRj6HOfqjfQ5hj7H0OcY+hxDn+NcvZE+x9B362bOnCl3d3c999xz2rhxo7p3766srCyZzWZNmDBBffv2NToxV1OmTNH8+fO1e/duo1OAPPEwOgAAgDvRhg0bFB8fr5UrV8rT01NdunRRTEyM0VlW9DnO1Rvpcwx9jqHPMfQ5hj7HuXojfY6h7/YNGDDA+ucWLVrohx9+0I4dOxQcHKx69eoZWHZNs2bNcmw7deqUzp49q6lTpxpQBNweZmgDAOAkR48eVUJCghITE3X06FE1a9ZMsbGxioqKUrFixYzOoy8fuHojffTRRx999N0uV2+kjz7c3IQJE2w+dnNzU/ny5XX//ferVq1aBlUBt44BbQAAnKBjx47avHmzKlSooB49eigmJkbVq1c3OsuKPse5eiN9jqHPMfQ5hj7H0Oc4V2+kzzH0OWbGjBl5PnbgwIH/Yglw52DJEQAAnKBYsWJauHChIiMj5e7ubnRODvQ5ztUb6XMMfY6hzzH0OYY+x7l6I32Ooc8xH3zwQZ6OM5lMLjOgvXHjRiUnJ8tkMql27dp64IEHjE4CbgkztAEAAAAAAIBC7vjx4+rVq5d27typihUrSpJSUlJ01113KT4+3roNcHVuRgcAAAAAAAAABdXXX3+t0NBQpaen59iXnp6u0NBQrVu3zoAyW6+88orc3d31008/6ZdfftEvv/yin376Se7u7nrllVeMzgPyjBnaAAAAAAAAwG3q2rWrHn74YT3zzDO57p83b57WrFmjxYsXO7nMVmBgoFasWKGwsDCb7Tt27FBUVJSOHDliTBhwi5ihDQAAAAAAANymX3/9VS1btrzh/ubNm+vnn392XtAtMplMRicAt4QBbQAAAAAAAOA2nT59Wm5uNx5iM5lMOnPmjBOLcte8eXMNGzZMx44ds247evSohg8frubNmxtYBtwaBrQBAAAAAACA21SpUiW7M7B/+eUXl7jg4ttvv61Lly4pLCxM9evXV2hoqO666y5dunRJb7/9ttF5QJ6xhjYAAAAAAABwm1555RVt2LBBGzZskJeXl82+S5cuqVWrVmrZsqXLDBqvX79ev/32mywWi2rXrm13uRTAFTGgDQAAAAAAANymv/76S82bN5fJZFLfvn1Vs2ZNSdJvv/2mOXPmyGKxaOPGjfL19TW4FCgcGNAGAAAAAAAAHHDkyBENHTpUa9eulcVybajNZDIpIiJCkydPVtWqVQ0uvGbXrl3avHmzTp8+LbPZbLNvzJgxBlUBt4YBbQAAAAAAACAfpKWl6cCBA7JYLAoODpa3t7fRSVbTpk3TG2+8ocDAQPn6+spkMln3mUwmffXVVwbWAXnHgDYAAAAAAABQyIWEhGjYsGHq3bu30SmAQ9yMDgAAAAAAAADw7zKbzWrRooXRGYDDGNAGAAAAAAAACrmnnnpKCQkJRmcADmPJEQAAAAAAAKCQs1gs6tq1q06cOKG6deuqSJEiNvtnzpxpUBlwazyMDgAAAAAAAADw7xo7dqzWrVunhg0bKj093egc4LYxQxsAAAAAAAAo5KpUqaJ3331XnTp1MjoFcAhraAMAAAAAAACFnJeXlxo0aGB0BuAwBrQBAAAAAACAQq5///6Ki4uTxcJiDSjYWHIEAAAAAAAAKOS6deumLVu2qHTp0qpdu7Y8PGwvrbdo0SKDyoBbw0UhAQAAAAAAgEKuXLlyateundEZgMOYoQ0AAAAAAAAAKBBYQxsAAAAAAAC4A12+fFkJCQl65JFHjE4B8owlRwAAAAAAAIA7yE8//aQFCxZo6dKlMplMatOmjdFJQJ4xoA0AAAAAAAAUcmlpaVq0aJEWLlyogwcPKiMjQ++++6569OihIkWKGJ0H5BlLjgAAAAAAAACF1MaNG/XUU0+pTp06+uKLL9SvXz/t27dPbm5uatKkCYPZKHCYoQ0AAAAAAAAUUp06ddKAAQO0fft2BQQEGJ0DOIwBbQAAAAAAAKCQeuihhzRv3jwdPnxY3bp1U2RkpNzd3Y3OAm4bS44AAAAAAAAAhdSiRYv0008/KSwsTCNHjlStWrX04osvSpJMJpPBdcCtM6WlpVmMjgAAAAAAAADw79u0aZPi4+O1YsUKlS9fXlFRUerYsaPuueceo9OAPGFAGwAAAAAAALjDpKena/HixYqPj9eePXt05swZo5OAPGFAGwAAAAAAALiD7dq1Sw0bNjQ6A8gTLgoJAAAAAAAA3AEyMzP166+/6vTp0zKbzUbnALeFAW0AAAAAAACgkFu/fr2effZZ/fXXXzn2mUwmlhxBgcGSIwAAAAAAAEAh16hRIzVr1kwvvfSSfH19ZTKZbPZ7enoaVAbcGga0AQAAAAAAgEIuICBA3377rYKCgoxOARziZnQAAAAAAAAAgH9XZGSktm3bZnQG4DBmaAMAAAAAAACFXHp6uvr27avq1aurTp06KlKkiM3+Hj16GFQG3BoGtAEAAAAAAIBCbtmyZerXr5+uXLmi4sWL26yhbTKZdPToUQPrgLxjQBsAAAAAAAAo5OrXr6/HHntMw4YNU4kSJYzOAW4ba2gDAAAAAAAAhVx6erqeeuopBrNR4DGgDQAAAAAAABRy7du314YNG4zOABzGgDYAAIALCQ0N1fTp043OcEhhOIcbya9ze/TRR/XSSy/lQ5FrK8x/FwAAKGiCgoI0duxYPfPMM3r33Xc1Y8YMmxtQUDCgDQAAkA9Onz6toUOHKjQ0VL6+vqpZs6Y6dOig9evX5/vn2rNnj3r06KFatWrJz89P9evXV0xMjI4cOZLvn8tIBw8e1MCBA1WvXj35+voqNDRUsbGx2rZtm9FpeZaQkKDKlSvn2B4fH6/XX3/9X//8oaGh8vb2VlJSUo59rVu3lre39y0NOG/evFne3t5KTU3N0/Hr169Xnz598vz4AADg3xMfH6+SJUtq27Zt+vDDD/XBBx9Yb3PmzDE6D8gzD6MDAAAACoOYmBhdvnxZM2bMULVq1XT69Gl99913OnPmTL5+ntOnTysqKkoRERFavHixfHx8dPToUX311Vc6f/58vn4uI+3YsUNRUVGqVauWJk+erNq1a+vixYv66quv9PLLL2vjxo239biZmZkqWrRoju1ZWVkqUqSIo9l55uPj47TPFRAQoIULF6pbt27Wbb/++qv27dunsmXL/iuf8/rzXL58+X/l8QEAwK0xm81KSkpSYGCgSpYsaXQO4BBmaAMAADgoLS1NW7Zs0RtvvKEWLVqoSpUquvvuuzVo0CB17tzZelxuyy/ktvTEhQsX1LdvX1WuXFm1atWyuc/WrVuVlpammTNnKiwsTFWrVtX999+vMWPGqF69etbj3njjDd1zzz3y9/dXaGioXn/9dWVkZFj3jx8/Xvfee68SExMVGhqqypUrq3///srMzNTcuXNVr149VatWTSNGjJDZbLY5h/Hjx9+wLzfp6ekaPHiwatSooYCAALVt21Y7duy44fEWi0X9+/dX1apVtWbNGrVp00bVqlVT/fr1NWTIEH3++efWY3/55RdFRUXJ399fQUFB6tevn9LT0637+/Xrp27duundd99V3bp1VbduXR0+fFje3t5asmSJ2rdvL39/f3300UeSrs1cCg8Pl5+fnxo1aqSZM2fanP8/zZgxQ82aNVOlSpVUp04dDRo0SGlpaZKuzWYeMGCALl68KG9vb3l7e2v8+PGScn7d09LS9Nxzz6lq1ary9/dXVFSU9u7da91/fab3xo0bde+996pSpUpq166dDh06ZPe5l6QuXbpo+/btNscuXLhQHTp0yHFRqKSkJLVq1UoBAQGqUaOGnnjiCR0/flySdPjwYbVv316SFBwcLG9vb/Xr1896PkOGDNFrr72m4OBgRUZGSrL9O//tt9+qfPny2rx5s/XzffjhhwoMDMzTeQAAgNtnMpnUvHlznTp1yugUwGEMaAMAADioZMmSKlmypL788kubQePbNWvWLNWqVUsbN27U8OHDNWbMGC1fvlyS5OfnJ7PZrM8//1wWi+WGj1G8eHHNmDFD27Zt05QpU7R06VJNnjzZ5pgjR47oyy+/VFJSkhYsWKDPP/9cPXv21E8//aSlS5fqvffe0wcffKAVK1bkue+fLBaLunXrppSUFCUlJWnTpk1q1qyZOnTooBMnTuR6n927d2vv3r36z3/+I3d39xz7vb29JUmXLl1Sly5dVKJECa1du1bx8fH64YcfNHDgQJvjv/vuO/3yyy9asmSJzWD46NGj9fTTT2vr1q169NFHNX/+fI0dO1YjRozQtm3bNG7cOE2bNk1z58694fPs5uam8ePHa8uWLZozZ47+97//6eWXX5YkhYeHa/z48SpevLiSk5OVnJysQYMG5fo4/fr10//+9z8lJiZq7dq18vLyUpcuXXT58mXrMVeuXNHUqVM1Y8YMffXVV0pPT9eQIUNu2HZduXLl9Mgjjyg+Pl7StdnTixcvVkxMTI5jMzMzNXz4cH377bdKSkpSamqqdcmQgIAALViwQNK1X6wkJydrwoQJ1vsuXrxYFotFq1at0vvvv5/jse+//3795z//0XPPPaezZ8/qt99+02uvvaa3335bQUFBNz0PAABw+0wmk2rWrKnTp08bnQI4jCVHAAAAHOTh4aGZM2dq8ODBmj9/vho0aKDw8HB17NhR99xzzy0/XqNGjfTiiy9KkmrUqKGffvpJs2bNUocOHdS4cWMNHTpU/fr104svvqi7775b999/v7p27aoqVapYH+P6oKokVa1aVUOGDNH06dP12muvWbdnZ2dr5syZKlOmjOrWrauIiAh999132rt3r4oWLaqQkBCFh4fr22+/VVRUVJ76/mnTpk3as2eP/vjjD3l5eUmSXnvtNa1evVpJSUkaPHhwjvscOHBAklSrVi27z9Mnn3yiixcvavbs2SpVqpQk6d1331X79u114MABVa9eXZLk6empGTNmyNPTU9K1mcaS1LdvX5vzmjRpkkaPHm3dFhQUpIMHD2revHnq27dvrg39+/e3eZ7HjBmjnj176v3331fRokVVunRpmUwm+fn53fA89u/fr1WrVmnlypW67777JEmzZ89WaGioPvnkE8XGxkqSrl69qsmTJ6tmzZqSpEGDBmnAgAEym81yc7M/T6VXr1564YUXNGLECK1atUplypSxfq6/+/sgd1BQkKZOnaomTZrozz//VOXKla1LpVSoUEHlypWzuW+VKlX05ptv2u0YPny41q9fr0GDBunIkSOKjIxUz5497d4HAADkj9GjR+v111/XxIkTFRoaKpPJZHQScFsY0AYAAMgHUVFRioyM1JYtW/TDDz9o7dq1mjFjhkaOHKmhQ4fe0mM1btw4x8d/nyU9cuRIDRgwQJs2bdL27du1cOFCTZkyRf/973/VokULSdLnn3+uuLg4HThwQBcvXlR2drays7NtHjcgIEBlypSxfuzr66saNWrYrDHt6+urv/7665b6/m7Xrl26dOmSatSoYbM9IyNDBw8ezPU+9mae/11ycrLq1atnHcyWrs2KdnNz0759+6wD2nXq1LEOZv/dXXfdZf3z6dOndezYMb3wwgs2X6+rV6/a7dm4caPeeecd/fbbbzp37pyys7OVmZmpkydPqmLFink+Dzc3NzVp0sS67fovGfbt22fd5unpaR3MliR/f39lZWUpPT39pmtyR0REyGKxaP369Vq4cKF69eqV63E7d+7U22+/rT179igtLc167seOHcv14pZ/FxYWdrNTVZEiRTR37lw1bdpUFSpUuOHMfgAAkP969+6tjIwMtWzZUh4eHjm+Pzp69KhBZcCtYUAbAAAgnxQrVkytWrVSq1at9Morr2jQoEGaMGGCBg0apKJF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0aJFRqcAAAADLF26VBs3btRff/0ls9lss4/vD1BQMEMbAADgDvXXX3/p7rvv1ubNmxUUFGRoS8uWLVWrVi29/vrrKlGihHx8fAztuS44OFgxMTHq27evSpQooTJlyhid5JArV66oYcOG+vDDD9WsWTOjcwAAgBONHDlScXFxeuCBB+Tv7y+TyWSzf9asWQaVAbeGGdoAAAB3qAULFujuu+82fDBbuvbW16effloBAQG3/RiZmZkqWrToTY+7evWq3N3dc/wQ909paWlKTU1V69atValSpX+9yxk8PT3VpUsXzZ49mwFtAADuMIsWLdK8efMUFRVldArgEDejAwAAAGCMJUuW5Lii/aOPPqqXXnrJZlu/fv3UrVs368ffffedHnzwQVWuXFlVqlRRRESEfv31V+v+bdu2qW3btqpYsaLq1KmjIUOG6Ny5c7k2HD58WN7e3jp37pwGDhwob29vJSQkWD9PRESE/Pz8VLNmTQ0fPlyZmZk2rUOGDNFrr72m4OBgRUZG5vo5xo8fr3vvvVcJCQkKCwuTr6+vLl68qPT0dA0ePFg1atRQQECA2rZtqx07dkiSzaz1Dh06yNvbW5s3b87T+d2oa9++fXr88ccVEBCgGjVqqE+fPjp58mSO5zkuLk516tRR1apV1b9/f126dMl6jMVi0fTp03X33XfL19dXdevW1ejRo637jx8/rqeeekpVq1ZV1apV9fjjj2v//v02z0ebNm20atUqm8cFAACFn9lsVmhoqNEZgMMY0AYAALgDnT17Vvv27dNdd911S/e7evWqevbsqaZNm+rbb7/VN998o+eee07u7u6SpF9++UWdOnVSmzZt9O2332rhwoXas2ePBg4cmOvjBQQEKDk5WcWLF9f48eOVnJysTp066fjx4+ratasaNGigTZs2afr06fr0009tBm8lafHixbJYLFq1apXef//9G3YfPnxYS5Ys0ccff6xvv/1Wnp6e6tatm1JSUpSUlKRNmzapWbNm6tChg06cOKHw8HBt3bpV0rWZ7MnJyQoPD8/z+f2z68SJE2rbtq3q1KmjtWvX6rPPPtOFCxfUo0cPm/Urt2zZor179+qzzz7TRx99pC+++MLmvMaMGaNJkybphRde0NatW/Xxxx+rcuXKkqRLly6pffv28vT01MqVK/X111/Lz89PUVFRNoPXd911l65evart27fn5UsOAAAKiSeffFJJSUlGZwAOY8kRAACAO9DRo0dlsVjk5+d3S/c7f/680tPT9cgjj6hatWqSpFq1aln3v/fee3rsscc0aNAg67YpU6aoefPm+uuvv1ShQgWbx3N3d5efn59MJpNKly5t7Zk8ebL8/Pw0ZcoUubm5KSQkRKNGjdILL7ygV199VcWLF5ckValSRW+++eZNuzMzMzV79mz5+vpKkjZu3Kg9e/bojz/+kJeXlyTptdde0+rVq5WUlKTBgwdbW318fKxdeT2/f3a9+eabql+/vs2A/OzZsxUUFKQdO3aoUaNGkqRSpUpp6tSp8vDwUEhIiDp27KiNGzdqyJAhunDhgmbNmqXx48crJiZGklS9enU1adJEkvTpp5/KYrFo1qxZ1uVU3n33XdWoUUNr1qzRY489JkkqXry4SpcurcOHD9/0eQMAAIVHenq6PvnkE23YsEH16tWTh4ftsODEiRMNKgNuDQPaAAAAd6CMjAxJUrFixW7pfj4+PurZs6c6d+6sFi1aqHnz5urYsaN17etdu3bpwIEDWrZsmfU+Fsu1a5AfPHgwx4D2jSQnJ6tx48Zyc/u/NxTee++9yszM1IEDB1S/fn1JUlhYWJ4er1KlStbB7Oudly5dUo0aNWyOy8jI0MGDB2/4OHk9v3927dq1S99//711NvXfHTx40DqgHRISYvPDpb+/v3788UdJ156TK1euqEWLFjdsO3z4cI51yC9dupTjnLy8vKx/BwAAwJ1h37591iVHfvvtN5t9N7u2COBKGNAGAAC4A5UrV07StQsf+vv7W7e7ublZB2ivu3r1qs3Hs2bNUr9+/bR27VqtWrVK48aNU0JCgiIiImQ2mxUbG6v+/fvn+JwVK1bMc5/FYrnhD1Z/316iRIk8Pd4/jzObzfL19dWqVatyHFuqVKkbPk5ezy+3z/fwww9r3LhxOe7390H+IkWK2OwzmUzWr8c/vy65tYWGhurDDz/Msc/Hx8fm47Nnz6p8+fJ2Hw8AABQuX3zxhdEJQL5gQBsAAOAOVK1aNZUuXVrJycmqXbu2dXv58uV14sQJm2N//vlnValSxWZbaGioQkND9fzzz6tLly7673//q4iICDVs2FB79+5V9erVHeqrXbu2li1bJrPZbJ2lvWXLFhUtWtS61IkjGjZsqFOnTsnNzc168ce83u92zq9hw4ZatmyZAgMDcwxa51VISIg8PT21ceNGBQcH5/o5lixZorJly8rb2/uGj3Pw4EFlZGSoYcOGt9UBAAAAGImLQgIAANyB3Nzc1KJFC23ZssVme/PmzfXNN9/oyy+/1O+//64RI0bozz//tO4/dOiQ3njjDW3btk1HjhzRpk2b9MsvvygkJESSNHjwYP3000964YUXrMtzrF69Ws8///wt9fXp00cnTpzQ0KFDlZycrDVr1mj06NF65plnrOtnO6Jly5Zq2rSpevbsqa+//lqHDh3SDz/8oLfeekvff//9De93u+f39NNP69y5c+rdu7d+/PFHHTp0SBs2bNDgwYN1/vz5PDWXKlVKzz33nEaPHq34+HgdPHhQ//vf/zRv3jxJUteuXeXr66uePXvq22+/1aFDh/Tdd9/p1Vdf1f79+62P8/333ysoKCjXQXEAAADA1TGgDQAAcId68skntWzZMmVnZ1u39erVS7169dLAgQMVGRmpEiVK6NFHH7XuL168uP744w89+eSTuueee9S/f3917drVOqBbv359ffnllzpy5IjatWun+++/X2PGjMnz2tnXVapUSZ988ol2796tBx54QAMHDlTnzp31+uuv58u5m0wmLV68WA888IAGDx6sxo0bq3fv3vrjjz/sLo1yu+dXsWJFrVmzRm5uburcubOaNm2qF198UUWLFpWnp2eeu0eNGqXnn39ekyZNUpMmTRQbG6vjx49Luva1+fLLLxUUFKQnn3xSTZo0Ub9+/ZSWlmYzY/vTTz/VE088kefPCQAAALgSU1pamv3F+AAAAFBoPfTQQ+rTp4+6d+9udAqc4Ndff1VUVJR+/PFHlSlTxugcAAAA4JYxQxsAAOAO9s4778hsNhudASc5ceKE3n//fQazAQAA7mDfffedunfvrjp16sjb21sJCQk3vc8vv/yitm3byt/fX3Xq1NHbb7+d46Ll3377rVq0aCE/Pz81bNgw14uV5wcuCgkAAHAHq1+/vurXr290BpykdevWRicAAADAYBcvXlTdunXVo0cPPffcczc9/ty5c3rsscfUrFkzrVu3Tr///rsGDBig4sWLa9CgQZKuXWvn8ccfV3R0tD744ANt3bpVQ4cOVbly5RQVFZWv/Sw5AgAAAAAAAAB3oMqVK2vixImKjo6+4THz5s3TG2+8od9++01eXl6SpEmTJunDDz/Ur7/+KpPJpFGjRmnFihX66aefrPcbNGiQ9u3bp6+//jpfm1lyBAAAAAAAAACQqx9++EH33nuvdTBbkiIiIpSSkqLDhw9bj/nnuwEjIiK0Y8cOZWVl5WsPS464qEd77TQ6wa63jw0xOuGmJjZNNDrBrpkeI41OsGuQ5U2jE+x6qnc1oxPs+uCD341OsGv+wFSjE+ya8F0joxPsSk+7bHSCXW9lDjM6wa4Z1WcanWDXc412G51g14rjrv3vQ5KaVDlldIJdiRtKG51g15UrV41OKNDGBy00OsG+q/n7A11+O1XvIaMT7PK8esnoBLvcLNlGJ9hlMrt4n1z/DeSpXgFGJ9iVftW1X+MsFpPRCXbV3TDB6AS7LrR83OgEu7wqVjc6ocBw5rjfyvgwhx/j1KlTqlSpks22ChUqWPcFBQXp1KlTatmyZY5jrl69qtTUVPn7+zvccR0ztAEAAAAAAAAAN2Qy2f5C6voFIf++PS/H5AcGtAEAAAAAAAAAufL19dWpU7bvwjx9+rSk/5upfaNjPDw8VLZs2XztYUDbQd26dbvhlTqTk5Pl7e2t9evXa/LkyYqMjFSlSpXk7e3t3EgAAAAAAAAALsHk5ua0W35o0qSJtmzZooyMDOu29evXq2LFiqpatar1mA0bNtjcb/369brrrrtUpEiRfOm4jgFtB8XGxmrTpk3WBdD/buHChQoMDFSLFi105coVtWvXTv369TOgEgAAAAAAAACkCxcuaPfu3dq9e7fMZrOOHTum3bt36+jRo5Kk0aNHq0OHDtbju3TpIi8vL/Xv31+//vqrli9frnfffVf9+/e3LifSu3dvHT9+XMOGDVNycrIWLFigxMREDRw4MN/7GdB2UGRkpHx9fZWQkGCzPSsrS0lJSerVq5fc3Nz06quvatCgQWrQoIFBpQAAAAAAAACMZnIzOe2Wmx07dqh58+Zq3ry5Ll++rPHjx6t58+Z66623JEknTpzQwYMHrceXKVNGy5YtU0pKilq1aqWXXnpJAwYMsBmsDgoK0uLFi/X999/rgQce0OTJk/X222/fcGULR3jk+yPeYTw8PNSjRw8lJiZq2LBhcvv/U/lXrVql1NRURUdHG1wIAAAAAAAAANc88MADSktLu+H+uLi4HNvq1aunVatW2X3c+++/X5s2bXI076aYoZ0PYmJidOzYMZt1YuLj49W6dWsFBAQYFwYAAAAAAADApZhMbk67FUaF86ycLDg4WM2aNVN8fLwkKSUlRWvXrlVMTIzBZQAAAAAAAABQeDCgnU9iY2O1cuVKnT17VomJifLx8VHbtm2NzgIAAAAAAADgQoxeQ7ugY0A7n0RFRcnT01NJSUmKj49X9+7dVaRIEaOzAAAAAAAAAKDQ4KKQ+cTLy0tdu3bVhAkTlJaWlmO5kaNHj+rs2bM6cuSIJGn37t2SpOrVq6tkyZJO7wUAAAAAAADgfCY35hg7gmcvH8XExCgtLU3h4eEKCQmx2ffWW2+pefPmGjlypCSpefPmat68uXbs2GFEKgAAAAAAAAAUOMzQzkdhYWFKS0vLdV9cXJzi4uKcGwQAAAAAAADApbgV0rWtnYUZ2gAAAAAAAACAAoEZ2gAAAAAAAADgJCYTc4wdwbMHAAAAAAAAACgQmKENAAAAAAAAAE5iYg1thzCg7aLePjbE6AS7XgmYanTCTY1a2tnoBLumjNxqdIJdryxsbXSCXWHt+hqdYFfJddONTrDrygvzjU6w6+mNjxmdYNel1ItGJ9j1Ts+1RifY1Weda///nBk+wegEuzqse9LohJsq8WgHoxPsGrAjwegEuyxmi9EJdrl5uBudYNeOiHlGJ9h19rKX0Ql21TUdNDrBrix3T6MT7DqlikYn2OWrFKMT7Drv7mN0wk15Z/1ldIJdFc//bHSCXefKBBqdYNf8quONTrDrqRP/NTrBrsyK1Y1OwB2CAW0AAAAAAAAAcBKTG6tAO4JnDwAAAAAAAABQIDCg7aBu3bopKioq133Jycny9vbW/PnzNXDgQDVs2FD+/v5q2LChRo8ercuXLzu5FgAAAAAAAICRTG4mp90KI5YccVBsbKx69eqlw4cPq2rVqjb7Fi5cqMDAQFWsWFHZ2dmaOnWqgoODlZycrOeff15nzpzRtGnTDCoHAAAAAAAAgIKFGdoOioyMlK+vrxISbC8ulJWVpaSkJPXq1UsPP/yw4uLiFBERoaCgIEVGRmro0KFavny5QdUAAAAAAAAAjOBmcnParTAqnGflRB4eHurRo4cSExNlNput21etWqXU1FRFR0fner/z58/L29vbSZUAAAAAAAAAUPAxoJ0PYmJidOzYMW3YsMG6LT4+Xq1bt1ZAQECO448eParp06erT58+TqwEAAAAAAAAYDTW0HYMA9r5IDg4WM2aNVN8fLwkKSUlRWvXrlVMTEyOY0+dOqXOnTurVatWGjBggLNTAQAAAAAAAKDAYkA7n8TGxmrlypU6e/asEhMT5ePjo7Zt29occ/LkSbVv31516tTR7NmzZTIVzt+SAAAAAAAAAMgdM7Qdw4B2PomKipKnp6eSkpIUHx+v7t27q0iRItb9J06cULt27VSrVi3NmzdPHh4eBtYCAAAAAAAAQMHDqGo+8fLyUteuXTVhwgSlpaXZLDeSkpKidu3ayd/fX+PHj1dqaqp1X/ny5eXu7m5EMgAAAAAAAAAnM5mYY+wIBrTzUUxMjObNm6fw8HCFhIRYt69bt0779+/X/v37Vb9+fZv77Nq1S1WrVnV2KgAAAAAAAAAUOAxo56OwsDClpaXl2B4dHa3o6GjnBwEAAAAAAABAIcKANgAAAAAAAAA4SWG9WKOzsGALAAAAAAAAAKBAYIY2AAAAAAAAADiJmxtzjB3BswcAAAAAAAAAKBCYoe2iJjZNNDrBrlFLOxudcFOjm8w0OsGuaZ9GGp1g15vhrv13MNKthtEJdn3R+j6jE+z6b9rvRifY9VrdhUYn2JV++rzRCXaN+byN0Ql2jbt7kdEJdr195Q+jE+yaGBhndMJNdSt/2egEu+ZUbWV0gl0Wi8XoBLuKFnPtHyEm/bbA6AT7rmQYXWDXibs7GJ1gV7FM134Nrux+xegEu654FDc6wS6/iweMTripC17ljU6w61jZhkYn2JVlLmJ0gl1dv33S6AS7zsYOMTrBrhJGBxQgJhNraDuCGdoAAAAAAAAAgALBtadXAAAAAAAAAEAhYnJjhrYjmKHtoG7duikqKirXfcnJyfL29tbatWvVvXt31a9fX35+fgoJCVHfvn11/PhxJ9cCAAAAAAAAQMHFgLaDYmNjtWnTJh0+fDjHvoULFyowMFCtWrVS8+bN9dFHH2n79u1asGCBDh06pF69ehlQDAAAAAAAAMAoJjeT026FEQPaDoqMjJSvr68SEhJstmdlZSkpKUm9evWSm5ub+vfvr8aNG6tKlSoKDw/X888/r59++kkZGa59URgAAAAAAAAAcBUMaDvIw8NDPXr0UGJiosxms3X7qlWrlJqaqujo6Bz3OXv2rD755BPdc889KlasmDNzAQAAAAAAABjIzeTmtFthVDjPysliYmJ07NgxbdiwwbotPj5erVu3VkBAgHXbqFGjVKlSJVWrVk3Hjh1TUlKSAbUAAAAAAAAAUDAxoJ0PgoOD1axZM8XHx0uSUlJStHbtWsXExNgc95///EebNm3SsmXL5O7urr59+8pisRiRDAAAAAAAAMAArKHtGA+jAwqL2NhYDR48WGfPnlViYqJ8fHzUtm1bm2PKlSuncuXKqUaNGqpVq5bq1aunLVu2qFmzZgZVAwAAAAAAAEDBwQztfBIVFSVPT08lJSUpPj5e3bt3V5EiRW54/PX1tjMzM52VCAAAAAAAAMBgzNB2DDO084mXl5e6du2qCRMmKC0tzWa5kR9++EG7du1S06ZNVaZMGR08eFBvvfWWqlSpoqZNmxpYDQAAAAAAAAAFBzO081FMTIzS0tIUHh6ukJAQ6/ZixYrp888/V4cOHXTPPfdo0KBBqlevnlatWqVixYoZWAwAAAAAAADAmUwmk9NuhREztPNRWFiY0tLScmxv0KCBvvjiC+cHAQAAAAAAAEAhwoA2AAAAAAAAADiJWyFd29pZWHIEAAAAAAAAAFAgMEMbAAAAAAAAAJzE5MYcY0fw7AEAAAAAAAAACgRmaLuomR4jjU6wa8rIrUYn3NS0TyONTrBrcMm3jE6wa4nnG0Yn2FX6aHWjE+xqfnqL0Ql2HSk/3+gEu/p+2c7oBLssZovRCXa92XqZ0Ql29U18yOgEu9KjlxqdYFenj1oYnXBTNRuPNTrBrgGbJxidYJeHp2t/i168XAmjE+z6pvkSoxPsKupuNjrBrvrZ+4xOsOtEsWpGJ9hVSulGJ9hV6vJfRifYddXDy+iEmzpv8jY6wS7v7NNGJ9hVLPO80Ql2ve7zrtEJdr3sftLoBOQTk4k1tB3BDG0AAAAAAAAAQIHAgLaDunXrpqioqFz3JScny9vbW+vXr7duy8jI0H333Sdvb2/t2LHDWZkAAAAAAAAAXIDJzeS0W2HEgLaDYmNjtWnTJh0+fDjHvoULFyowMFAtWvzfW4NHjhypypUrOzMRAAAAAAAAAAoFBrQdFBkZKV9fXyUkJNhsz8rKUlJSknr16iW3/3/l0pUrV2rz5s0aO9a115UEAAAAAAAA8O9wczM57VYYMaDtIA8PD/Xo0UOJiYkym//vAi+rVq1SamqqoqOjJUl//vmnhg4dqg8++EDFihUzKhcAAAAAAAAACiwGtPNBTEyMjh07pg0bNli3xcfHq3Xr1goICFB2draeeeYZDRgwQA0aNDAuFAAAAAAAAAAKMAa080FwcLCaNWum+Ph4SVJKSorWrl2rmJgYSdKUKVNUpEgRDRw40MhMAAAAAAAAAAYzmUxOuxVGDGjnk9jYWK1cuVJnz55VYmKifHx81LZtW0nSxo0btXnzZpUvX17lypXT3XffLUl68MEH9cwzzxiZDQAAAAAAAAAFhofRAYVFVFSUXn75ZSUlJSk+Pl7du3dXkSJFJEkzZ87UpUuXrMeeOHFCnTp10pw5cxQeHm5UMgAAAAAAAAAnMxXSizU6CwPa+cTLy0tdu3bVhAkTlJaWZl1uRJKCgoJsji1RooQkqVq1aqpcubIzMwEAAAAAAACgwGLJkXwUExOjtLQ0hYeHKyQkxOgcAAAAAAAAAC7GzWRy2q0wYkA7H4WFhSktLU1r1qyxe1zVqlWVlpamu+66y0llAAAAAAAAAHDN3Llz1aBBA/n5+alFixb6/vvvb3js+PHj5e3tnevtr7/+kiRt3rw51/2//fZbvrez5AgAAAAAAAAAOInRa2gvXbpUw4YN05QpU9S0aVPNnTtXXbt21datWxUYGJjj+EGDBumpp56y2fbUU0/JZDKpQoUKNtu3bt0qHx8f68fly5fP935maAMAAAAAAADAHWLmzJnq2bOnnnjiCYWEhGjSpEny8/PThx9+mOvxJUuWlJ+fn/WWlZWlLVu26IknnshxbIUKFWyOdXd3z/d+BrQBAAAAAAAAwElMbian3f4pMzNTO3fuVOvWrW22t27dWtu2bctT/8KFC1WmTBl16NAhx76WLVsqJCREHTp00KZNm27vCboJlhwBAAAAAAAAgDtAamqqsrOzcywVUqFCBZ06deqm9zebzUpISFD37t3l6elp3e7v76+pU6fq7rvvVmZmppKSkhQVFaUvvvhC9913X76eAwPaLmqQ5U2jE+x6ZWHrmx9ksDfDE41OsGuJ5xtGJ9jVZd8zRifY9Xqve41OsGvOz38anWDX/OPfGp1g1xtPbDA6wa4zqReNTrDrvewRRifY9c4rPxidYNdLaf/OLIL8smmCaz9/klS6zHGjE+ya+9ByoxPsunQx0+iEAm3abtf+PtqtbP6vI5mfjtd7xOgEuypfSDY6wS6LW/6/rTo/Fck4Z3SCXRYv138TeTG3y0Yn2HU8O+fat67E4mHsusE3M9l/ptEJdmWea2J0gn0V/I0uKDBMJuP/LfyzwWKx5Knrq6++0rFjxxQbG2uzvWbNmqpZs6b14yZNmujIkSOaPn16vg9ou/6rBQAAAAAAAADAYeXKlZO7u3uO2dinT5/OMWs7N/Pnz1d4eLjq1Klz02MbNWqkAwcO3HbrjTCgDQAAAAAAAABO4uZmctrtn4oWLaqwsDCtX7/eZvv69esVHh5utzslJUVfffVVjtnZN7Jnzx75+fnl/YnJIwa0HdStWzdFRUXlui85OVne3t5av369QkND5e3tbXN74403nBsLAAAAAAAA4I42YMAAJSYmasGCBUpOTtYrr7yiEydOqHfv3pKk0aNH53rBx/j4eJUoUUKPPfZYjn2zZs3SF198of3792vv3r0aPXq0Vq5cqWeeyf8lbVlD20GxsbHq1auXDh8+rKpVq9rsW7hwoQIDA9WiRQtJ0ssvv6w+ffpY95coUcKprQAAAAAAAACMZfQa2p06ddKZM2c0adIknTx5UnXq1NHixYtVpUoVSdKJEyd08OBBm/tYLBYtXLhQXbt2VfHixXM8ZlZWlkaOHKmUlBQVK1bM+pgPP/xwvvczoO2gyMhI+fr6KiEhQSNG/N9FuLKyspSUlKSnn35abm7XJsKXKlXqX5lmDwAAAAAAAAB59fTTT+vpp5/OdV9cXFyObSaTSbt3777h4w0ePFiDBw/Otz57WHLEQR4eHurRo4cSExNlNput21etWqXU1FRFR0dbt02fPl3VqlXT/fffr8mTJyszkyvYAwAAAAAAAHcSk5vJabfCiAHtfBATE6Njx45pw4YN1m3x8fFq3bq1AgICJEnPPvus5s6dqxUrVqhv376aNWuWhg4dalAxAAAAAAAAABQ8LDmSD4KDg9WsWTPrIHZKSorWrl2rDz/80HrMwIEDrX+uX7++SpUqpd69e2v06NEqW7asEdkAAAAAAAAAnKyQTpx2GmZo55PY2FitXLlSZ8+eVWJionx8fNS2bdsbHt+oUSNJ0oEDB5yVCAAAAAAAAAAFGgPa+SQqKkqenp5KSkpSfHy8unfvriJFitzw+D179kgSF4kEAAAAAAAA7iCsoe0YlhzJJ15eXuratasmTJigtLQ0xcTEWPf98MMP2r59ux544AGVLl1aO3bs0IgRI9SmTRsFBgYaWA0AAAAAAAAABQcD2vkoJiZG8+bNU3h4uEJCQqzbixYtqmXLluntt99WZmamAgMDFRsbq8GDBxtYCwAAAAAAAMDZnDtz2uLEz+UcDGjno7CwMKWlpeW6/ZtvvnF+EAAAAAAAAAAUIgxoAwAAAAAAAICTuJmYoe0ILgoJAAAAAAAAACgQmKENAAAAAAAAAE7i3DW0Cx9maAMAAAAAAAAACgRmaLuop3pXMzrBrrB2fY1OuKlItxpGJ9hV+mh1oxPser3XvUYn2DXmtS1GJ9g1alxToxPsOhe/3OgEu5p27WJ0gl0XM7yMTrDLY5un0Ql2+fsWMTrBLovJtX/ffybd9dfA86502ugEu+5qWMnoBLtcfcJO8WJmoxPsu1LH6AK7MkqVNzrBLnfLVaMT7Dpf3NfoBLt+SqtldIJdTUvtNjrBrhIXThqdcFPFPYobnWCX77m9RifYdamUv9EJBdr+kncZnWBXsNEBuGMwoA0AAAAAAAAATuLUa0IWQq49BQkAAAAAAAAAgP+PAW0HdevWTVFRUbnuS05Olre3t9avXy9JWrv2/7F3/3FV1/f//++HHyKFefAHoIE6MYlpSrOksaWCJc2RLHsTEB56m27NtG/tbTV/bJX9UJr204psYjMPFPWelWbUNsO03++11LYVW6YIhZLI8UcpIOd8/3h/dvZm4OvgwPM8HG/Xy+V1ucjz+Xw9ub9QER88eJ7NuvzyyzVo0CANGTJE06ZN82dUAAAAAAAAAIaFhNj8dgUjCtpdVFhYqK1bt6q6urrd3Lp165SQkKCJEyfqlVde0fXXX6/c3Fxt3bpVv//97zVjxgwDiQEAAAAAAACgZ+IM7S7KzMxUTEyMSktLtWjRIu94S0uLysvLNXv2bHk8Hi1YsEB33323rrvuOu+apKQkE5EBAAAAAAAAGGLjEO0uoUO7i8LCwpSfn6+ysjK53f98xfeKigo1NDSooKBA27dvV21trXr16qUJEyZo5MiRuuqqq7Rjxw6DyQEAAAAAAACgZ6Gg3Q0cDodqa2u1ZcsW75jT6VRGRobi4+O1Z88eSdJ9992n+fPn6/nnn9fgwYOVlZWluro6M6EBAAAAAAAA+F2Izea3KxhR0O4GiYmJSktLk9PplCTV1dVp8+bNcjgckuTt3L711luVnZ2tlJQUPfLII+rbt6/Ky8uN5QYAAAAAAACAnoSCdjcpLCzUpk2b1NjYqLKyMkVHR2vq1KmSpNjYWEltz8wOCwvT8OHDVVtbayQvAAAAAAAAAP+zhfjvCkZB+lj+l52drYiICJWXl8vpdCovL0/h4eGSpJSUFEVEROjvf/+7d73b7dbu3buVkJBgKjIAAAAAAAAA9ChhpgMEi8jISOXk5KioqEgul8t73IgknXPOOZo5c6aKiop07rnnasiQIXrqqad06NAhXXPNNQZTAwAAAAAAAPAnW5Cebe0vFLS7kcPhUElJiVJTU9scLyJJ99xzj3r16qU5c+bo2LFjGjNmjDZs2KBBgwYZSgsAAAAAAAAAPQsF7W6UkpIil8vV4Vx4eLjuvvtu3X333f4NBQAAAAAAACBghITQod0VnKENAAAAAAAAAOgR6NAGAAAAAAAAAD/hCO2uoUMbAAAAAAAAANAj0KENAAAAAAAAAH5i4wztLqGgHaCeeurvpiNYinpjpekIPr2S8T3TESxNOPCu6QiWfv3nL0xHsHTnvZeYjmBpyS/eMx3B0uqHi0xHsPSK84DpCJYavzpkOoKliScC++/vJ7bA/vgdG97PdARLHrfHdASfqt3DTEew9PY7DaYjWGptaTUdwVKg/wcs6/wa0xEsRez9zHQES63fzTUdwVL/w7tNR7A0KXy/6QiWmt1RpiNYqu830nQEn0I8btMRLO2KmmQ6gqWzw46ZjmAp8kBg/z8kIqTZdAQfKDPCP/iTBgAAAAAAAAB+EuD9AQGPM7S7KDc3V9nZ2R3OVVVVyW63q7KyUna7vcPrpZde8m9gAAAAAAAAAOih6NDuosLCQs2YMUPV1dUaOnRom7l169YpISFB3/ve91RVVdVmbtWqVXrqqad02WWX+TMuAAAAAAAAAINsNlq0u4IO7S7KzMxUTEyMSktL24y3tLSovLxcM2bMUK9evRQbG9vm2rBhg66++mpFRQX2GWYAAAAAAAAAECgoaHdRWFiY8vPzVVZWJrf7ny8OUVFRoYaGBhUUFLS7Z9u2bfrss8/0n//5n35MCgAAAAAAAMC0kBD/XcEoSB/LvxwOh2pra7VlyxbvmNPpVEZGhuLj49utX7t2rUaPHq0LL7zQjykBAAAAAAAAoGejoN0NEhMTlZaWJqfTKUmqq6vT5s2b5XA42q09ePCgNm7cSHc2AAAAAAAAcAay2Wx+u4IRBe1uUlhYqE2bNqmxsVFlZWWKjo7W1KlT26179tlnFRISopycHAMpAQAAAAAAAKDnoqDdTbKzsxUREaHy8nI5nU7l5eUpPDy83bp169bpRz/6kfr27WsgJQAAAAAAAACTbCH+u4JRkD6W/0VGRionJ0dFRUXavXt3h8eNvPvuu/r000913XXXGUgIAAAAAAAAAD0bBe1u5HA45HK5lJqaqqSkpHbza9euVVJSki655BID6QAAAAAAAACgZwszHSCYpKSkyOVynXT+ySef9F8YAAAAAAAAAAEnJEhfrNFf6NAGAAAAAAAAAPQIdGgDAAAAAAAAgJ/QoN01dGgDAAAAAAAAAHoEOrQBAAAAAAAAwE/o0O4am8vl8pgOgfZa935sOoKlpvAo0xF86uPaazqCpb0DLjIdwdKwL98yHcHS4Y0bTEew5PppkekIlmbfsst0BEsvLws3HcGSOySwvx+8J3Sk6QiWkhu3mo5g6d3ITNMRLKV9vcl0BJ/+NnCC6QiWRh4I7H/jQpqOmY5gyR0RaTqCpe19LzMdwdJ5+tR0BEvhLV+bjmDpyFkxpiNY6rf5GdMRLNVd/lPTESx9cSywf38laUjkl6YjWPrsaILpCJZG9w7sz4HfhPUxHcFSuKfZdARL9oFxpiP0GE+87r+62o2ZR/32vvwlsP9HDgAAAAAAAABBJCSEFu2u4AxtAAAAAAAAAECPQEG7i3Jzc5Wdnd3hXFVVlex2uyorK/XZZ5/p2muv1fDhwxUfH6/LLrtMf/jDH/ycFgAAAAAAAIBJNpv/rmBEQbuLCgsLtXXrVlVXV7ebW7dunRISEjRx4kTl5uaqqalJL7/8srZu3apLLrlE1157rXbv3m0gNQAAAAAAAAD0PBS0uygzM1MxMTEqLS1tM97S0qLy8nLNmDFDjY2N2rVrl26++WZdcMEFGj58uO666y6dOHFCO3fuNJQcAAAAAAAAgL+F2Px3BSMK2l0UFham/Px8lZWVye12e8crKirU0NCggoIC9evXT0lJSSovL9fRo0fV2tqq3/zmN4qKilJqaqrB9AAAAAAAAADQc1DQ7gYOh0O1tbXasmWLd8zpdCojI0Px8fGy2Wx68cUX9cknnyghIUExMTEqKirSf//3fysuLs5ccAAAAAAAAAB+ZQux+e0KRhS0u0FiYqLS0tLkdDolSXV1ddq8ebMcDockyePxf2yBFAAAlrhJREFUaP78+erXr58qKiq0efNmZWdnq7CwUF9++aXJ6AAAAAAAAADQY4SZDhAsCgsLdfPNN6uxsVFlZWWKjo7W1KlTJUlbt27Va6+9pt27d8tut0uSUlJSVFlZqdLSUt12220GkwMAAAAAAADwF1twNk77DR3a3SQ7O1sREREqLy+X0+lUXl6ewsPDJUnffPONJCkkpO2HOyQkpM252wAAAAAAAACAk6NDu5tERkYqJydHRUVFcrlc3uNGJGn8+PGKjo7W3LlzdfvttysyMlJr167Vnj17lJmZaTA1AAAAAAAAAH8K0qOt/YYO7W7kcDjkcrmUmpqqpKQk73j//v3129/+Vl9//bWmTZum9PR0vfPOOyotLVVKSoq5wAAAAAAAAADOOKtXr9aYMWMUGxuriRMn6p133jnp2urqatnt9nbXH/7whzbr3nrrLU2cOFGxsbEaO3as1qxZc1qy06HdjVJSUuRyuTqcu/DCC7V+/Xr/BgIAAAAAAAAQUEyfob1+/XotWLBADzzwgC655BKtXr1aOTk5eu+995SQkHDS+377299q9OjR3rejo6O9v96zZ4+uueYaFRQU6KmnntJ7772n+fPnq3///srOzu7W/HRoAwAAAAAAAMAZ4vHHH9e1116r6667TklJSVq+fLliY2N9dlT369dPsbGx3qtXr17euaefflpxcXFavny5kpKSdN111yk/P1+PPfZYt+enoA0AAAAAAAAAfhIS4r/rXzU3N2v79u3KyMhoM56RkaH333/fMrfD4dCIESOUmZmpl19+uc3cBx980G7PyZMn66OPPlJLS8u/94E6CQraAAAAAAAAAHAGaGhoUGtrqwYOHNhmfODAgaqvr+/wnqioKN1zzz16+umn9cILL2jChAmaOXOmysvLvWvq6+s73PPEiRNqaGjo1mfgDG0AAAAAAAAA8BPTZ2j/b4a2ITweT7uxf+jfv79uuukm79sXXnihDh48qEceeUS5ubmWe3Y03lUUtANU0dvjTEewNPvNq0xH8OkX315nOoKln7yaZTqCpbuu22I6gqVLcv7DdARLrzgPmI5g6eVl4aYjWMpe2L0/jtTdYr4VYzqCpaV115iOYOn/G+00HcHSvXlVpiNY+sW2KaYj+FSYedx0BEvzt040HcFSS/MJ0xEs9Y4M7H9DHhheYjqCtZYm0wks/fXCWaYjWPr23k2mI1g6POFq0xEs9XZ/bTqCpfN6fWY6gk+RX7tMR7A06Owo0xEs1XsGmY5gafCa20xHsHRs5kLTERAE+vfvr9DQ0Hbd2AcOHGjXYW1l3LhxKi0t9b4dExPT4Z5hYWHq169f10L/C44cAQAAAAAAAAA/sdlsfrv+Va9evZSSkqLKyso245WVlUpNTe30M3z88ceKjY31vj1+/Hht2bKl3Z4XXnihwsO7tyGCDm0AAAAAAAAAOEPMnTtXN9xwg8aNG6fU1FStWbNG+/bt08yZMyVJS5Ys0YcffqgNGzZIksrKyhQeHq4xY8YoJCREr732mlavXq277rrLu+fMmTP161//WgsWLNDMmTP1/vvvq6ysTKtXr+72/BS0uyg3N1fHjx9v98qeklRVVaXU1FS9+OKLio6O1l133aU//elPCg0N1bRp03TfffcpKiqwfxwIAAAAAAAAQPcJMXyG9vTp03Xw4EEtX75c+/fvV3Jysp5//nkNGTJEkrRv3z7t3r27zT0rVqxQTU2NQkNDlZiYqMcee6zN+dnDhg3T888/r0WLFmnNmjWKi4vT/fffr+zs7G7PT0G7iwoLCzVjxgxVV1dr6NChbebWrVunhIQEnX/++frud7+rH/3oR1q+fLmOHDmihQsX6sYbb9QzzzxjKDkAAAAAAACAM9Hs2bM1e/bsDueKi4vbvH3ttdfq2muv9bnn97//fW3durVb8lnhDO0uyszMVExMTJtD0CWppaVF5eXlmjFjhl5//XWFhITogQce0HnnnafvfOc7evDBB7VhwwZ9/vnnhpIDAAAAAAAAQM9CQbuLwsLClJ+fr7KyMrndbu94RUWFGhoaVFBQoKamJoWHhys0NNQ7HxkZKUl69913/Z4ZAAAAAAAAgBk2m/+uYERBuxs4HA7V1ta2eSVPp9OpjIwMxcfHa8KECWpoaNBDDz2k5uZmuVwu76Hp+/fvNxMaAAAAAAAAAHoYCtrdIDExUWlpaXI6nZKkuro6bd68WQ6HQ5KUnJys4uJiFRcXa9CgQRo5cqSGDh2qmJiYNl3bAAAAAAAAAIIbHdpdQ0G7mxQWFmrTpk1qbGxUWVmZoqOjNXXqVO98Tk6O/va3v+mTTz7R559/rgULFujAgQPtXkgSAAAAAAAAANAxCtrdJDs7WxERESovL5fT6VReXp7Cw8PbrYuJiVFUVJTWr1+v3r17a9KkSf4PCwAAAAAAAMCIkBD/XcEozHSAYBEZGamcnBwVFRXJ5XJ5jxv5h6eeekrjx49XVFSUKisrdccdd+jOO++U3W43ExgAAAAAAAAAehgK2t3I4XCopKREqampSkpKajP34YcfatmyZfr666913nnn6aGHHlJeXp6hpAAAAAAAAABMCNazrf2FgnY3SklJkcvl6nBu1apV/g0DAAAAAAAAAEGGgjYAAAAAAAAA+EkIHdpdEqRHgwMAAAAAAAAAgg0d2gAAAAAAAADgJ5yh3TV0aAMAAAAAAAAAegQ6tAPUIdcx0xEsfdPwtekIPh06cMR0BEset8d0BEsHA/z3+OvjkaYjWGr86pDpCJbcIYH96T/mWzGmI1iq3/2F6QiWXPUu0xEsfdWvwXQESyGeVtMRLB05dNx0BJ9a3aGmI1g68GWj6Qg9WlNUb9MRLLldB01HsBQS1cd0BEtnh35jOoIld3hg//mzyW06gqVA/zeuJSTCdATfettNJ7B0whPY/waHBvjfkabGwK4jhLmbTUdAN6FDu2vo0AYAAAAAAAAA9AiB3aIHAAAAAAAAAEEkhA7tLqFD24fc3FxlZ2d3OFdVVSW73a7KykqtWLFCmZmZGjx4sOx2e4fra2pqlJubq8GDB2v48OG6/fbb1dzMj4sAAAAAAAAAQGdQ0PahsLBQW7duVXV1dbu5devWKSEhQRMnTlRTU5OysrI0Z86cDvdpbW1Vbm6ujh49qldffVUlJSXasGGDFi9efLofAQAAAAAAAECAsNn8dwUjCto+ZGZmKiYmRqWlpW3GW1paVF5erhkzZigkJESLFy/WTTfdpDFjxnS4zxtvvKFPPvlEq1atUkpKitLT07VkyRI988wzOnz4sD8eBQAAAAAAAAB6NAraPoSFhSk/P19lZWVyu//5arwVFRVqaGhQQUFBp/b54IMPlJSUpPj4eO/Y5MmT1dTUpO3bt3d3bAAAAAAAAAAByGbz+O0KRhS0O8HhcKi2tlZbtmzxjjmdTmVkZLQpUFupr6/XwIED24z1799foaGhqq+v7864AAAAAAAAABCUKGh3QmJiotLS0uR0OiVJdXV12rx5sxwOxyntYzvJwTUnGwcAAAAAAAAQXEJs/ruCEQXtTiosLNSmTZvU2NiosrIyRUdHa+rUqZ2+PyYmpl0ndkNDg1pbW9t1bgMAAAAAAAAA2qOg3UnZ2dmKiIhQeXm5nE6n8vLyFB4e3un7x48fr6qqKn3xxRfescrKSkVERCglJeU0JAYAAAAAAAAQaGw2/13BKMx0gJ4iMjJSOTk5KioqksvlanfcSE1NjRobG7V3715J0s6dOyVJw4cPV1RUlDIyMpScnKyf/vSnuvfee9XY2Kg77rhDhYWFOuecc/z+PAAAAAAAAADQ09ChfQocDodcLpdSU1OVlJTUZm7p0qWaMGGCfvnLX0qSJkyYoAkTJuijjz6SJIWGhqq8vFxnnXWWrrjiCs2cOVNZWVm69957/f4cAAAAAAAAAMygQ7tr6NA+BSkpKXK5XB3OFRcXq7i42PL+hIQElZeXn4ZkAAAAAAAAABD8KGgDAAAAAAAAgJ+E2DymI/RoHDkCAAAAAAAAAOgRKGgDAAAAAAAAAHoEjhwJUEubF5iOYOmhazebjuDT3S//wHQES/dlvGg6gqVHWxeZjmAp7P0I0xEsTTzxhekIlvaErjEdwdLSumtMR7DkqneZjmDp1pjlpiNYWrjmR6YjWPr65xWmI1i66q7vmY7g0/BpJaYjWPqvt39mOoKlXmcH9pfofeOjTUewtP6KTaYjWDo/xmU6gqUw9wnTESw1DBptOoKl8NYm0xEs9Tm6z3QESwfPGWo6gk+NIQNNR7DUciLcdARL/cIC+8/gA8lrTUew9HP3Z6YjoJsE64s1+gsd2gAAAAAAAACAHiGw2z8AAAAAAAAAIIjQod01dGj7kJubq+zs7A7nqqqqZLfbVVlZqRUrVigzM1ODBw+W3W7vcP3Pf/5zTZo0SbGxsbrgggtOY2oAAAAAAAAACD4UtH0oLCzU1q1bVV1d3W5u3bp1SkhI0MSJE9XU1KSsrCzNmTPnpHu53W7l5+crLy/vdEYGAAAAAAAAEKBCbP67ghEFbR8yMzMVExOj0tLSNuMtLS0qLy/XjBkzFBISosWLF+umm27SmDFjTrrX8uXLdcMNN2jEiBGnOzYAAAAAAAAABB0K2j6EhYUpPz9fZWVlcrvd3vGKigo1NDSooKDAYDoAAAAAAAAAPYlNHr9dwYiCdic4HA7V1tZqy5Yt3jGn06mMjAzFx8ebCwYAAAAAAAAAZxAK2p2QmJiotLQ0OZ1OSVJdXZ02b94sh8NhOBkAAAAAAACAnsRm898VjChod1JhYaE2bdqkxsZGlZWVKTo6WlOnTjUdCwAAAAAAAADOGBS0Oyk7O1sREREqLy+X0+lUXl6ewsPDTccCAAAAAAAA0IOE2Px3BaMw0wF6isjISOXk5KioqEgul6vdcSM1NTVqbGzU3r17JUk7d+6UJA0fPlxRUVGSpM8//1xHjx5VXV2dWlpavGvOP/989erVy49PAwAAAAAAAAA9DwXtU+BwOFRSUqLU1FQlJSW1mVu6dKmeffZZ79sTJkyQJG3cuFGXXnqpJOmmm27S22+/3W7Njh07NHTo0NMdHwAAAAAAAIBhNpvHdIQejYL2KUhJSZHL5epwrri4WMXFxZb3b9q06TSkAgAAAAAAAIAzAwVtAAAAAAAAAPATW5Cebe0vvCgkAAAAAAAAAKBHoEMbAAAAAAAAAPwkRJyh3RV0aAMAAAAAAAAAegQ6tAPUY8MfNx3B0qw3rjYdwad7v/Oc6QiWflJ2uekIlh76+QemI1iKiwk3HcHSJ7ZDpiNYur9xq+kIlv6/0U7TESx91a/BdARLC9f8yHQES8uueMp0BEsbjn5hOoKlp2ZvMB3Bp29F9jEdwdJ9KU+bjmApLDyw/42LskeZjmBp1c6fm45gKWTMxaYjWDpuH2Q6gqWQ1hOmI1g6fHas6QiWjvX9lukIlmwet+kIPg0+9pnpCJaaw882HcFSs6e36QiW5r53jekIlk5cscJ0BHQTztDuGjq0AQAAAAAAAAA9Ah3aAAAAAAAAAOAnNhtnaHcFHdo+5ObmKjs7u8O5qqoq2e12VVZWasWKFcrMzNTgwYNlt9vbrf344481a9YsjRo1SnFxcbrooov06KOPyu0O/B+pAgAAAAAAAIBAQEHbh8LCQm3dulXV1dXt5tatW6eEhARNnDhRTU1NysrK0pw5czrcZ/v27erfv7+efPJJvffee1q4cKF+9atf6aGHHjrdjwAAAAAAAAAgQITY/HcFI44c8SEzM1MxMTEqLS3VokWLvOMtLS0qLy/X7NmzFRISosWLF0uSXn755Q73cTgcbd4eNmyYduzYoQ0bNmj+/Pmn7wEAAAAAAAAAIEjQoe1DWFiY8vPzVVZW1uZ4kIqKCjU0NKigoODf3vvIkSMdHk8CAAAAAAAAIDjZ5PHbFYwoaHeCw+FQbW2ttmzZ4h1zOp3KyMhQfHz8v7Xn9u3bVVZWpuuvv76bUgIAAAAAAABAcKOg3QmJiYlKS0uT0+mUJNXV1Wnz5s3tjhHprL///e/Kzc3VnDlzTvqCkwAAAAAAAACCj83mvysYUdDupMLCQm3atEmNjY0qKytTdHS0pk6desr7/O1vf1NWVpamT5+uu+66q/uDAgAAAAAAAECQoqDdSdnZ2YqIiFB5ebmcTqfy8vIUHh5+Snt8+umnysrKUnZ2tpYtW3aakgIAAAAAAADAya1evVpjxoxRbGysJk6cqHfeeeeka7dt26b8/HwlJSVp0KBBSktL07p169qtsdvt7a6//e1v3Z49rNt3DFKRkZHKyclRUVGRXC5Xu+NGampq1NjYqL1790qSdu7cKUkaPny4oqKi9Mknn2jatGm69NJLNX/+fO3fv997b2xsrP8eBAAAAAAAAIAxNpvZF2tcv369FixYoAceeECXXHKJVq9erZycHL333ntKSEhot/6DDz7QqFGjdPPNNysuLk6bN2/WLbfcot69eysnJ6fN2vfee0/R0dHetwcMGNDt+SlonwKHw6GSkhKlpqYqKSmpzdzSpUv17LPPet+eMGGCJGnjxo269NJL9dJLL+mrr77S+vXrtX79+jb3ulyu054dAAAAAAAAAB5//HFde+21uu666yRJy5cv1+bNm7VmzRrdeeed7dbPnz+/zduzZs3Stm3btGHDhnYF7YEDB6p///6nL7woaJ+SlJSUkxafi4uLVVxcfNJ7Fy5cqIULF56mZAAAAAAAAAB6ghCZ69Bubm7W9u3bddNNN7UZz8jI0Pvvv9/pfY4cOaLBgwe3G580aZKam5uVlJSkW2+91dv0250oaAMAAAAAAADAGaChoUGtra0aOHBgm/GBAweqvr6+U3u89tprevPNN/X66697x+Li4vTggw/qO9/5jpqbm1VeXq7s7Gy98sor+t73vtetz0BBGwAAAAAAAAD8xGYznUCy/UsIj8fTbqwj7733nn784x/r/vvv17hx47zj5513ns477zzv2+PHj9fevXu1cuXKbi9oh3TrbgAAAAAAAACAgNS/f3+Fhoa268Y+cOBAu67tf/Xuu+8qJydHCxcu1KxZs3y+r3Hjxunzzz/vUt6OUNAGAAAAAAAAAD+x2Tx+u/5Vr169lJKSosrKyjbjlZWVSk1NPWnmt99+Wzk5Obr99tt14403duo5P/74Y8XGxp7aB6cTbC6Xy9wp5Dip419+ZjqCpeawSNMRfIpsOmQ6gqVDZ3X/X+juNMDV/d9B604eW2B/P+5YZD/TESz9uSnZdARLo3pXmY5gKcTTajqCpa979TUdwVL00S9MR7A0bbHbdARL5Q8E9ucXSQpxB/bfkcjjLtMRLIW4W0xHsORRAPyMrIVdUReajmApzhPYnwPdAf41VqA7+1iD6QiWjpwVYzqCpbAA//wnSa0hgX1ya1NIYP9ffeChXaYjWGrpdbbpCJaORFp3z5rWb0Bg5wsk22vD/fa+UuLbf25dv369brjhBj3wwANKTU3VmjVr5HQ69e6772rIkCFasmSJPvzwQ23YsEGStG3bNuXm5mrWrFmaN2+ed5/Q0FANGDBAkvTEE09oyJAhSk5OVnNzs55//nk99NBDeuaZZzRt2rRufabA/kwMAAAAAAAAAEHEJrP9xdOnT9fBgwe1fPly7d+/X8nJyXr++ec1ZMgQSdK+ffu0e/du7/qysjJ98803WrlypVauXOkdT0hI0McffyxJamlp0S9/+UvV1dWpd+/e3j2nTJnS7fnp0A5QdGh3HR3aXUOHdtfQod01dGh3DR3aXUOHdtfRod01dGh3DR3aXUOHdtfQod01dGh3HR3aXUOHdtfQod15O2r997lkbPwJv70vf+GrFR9yc3OVnZ3d4VxVVZXsdrsqKyu1YsUKZWZmavDgwbLb7e3WHjhwQNOnT9f555+vmJgYjRo1SrfeeqsOHQrsoisAAAAAAACA7hNi898VjCho+1BYWKitW7equrq63dy6deuUkJCgiRMnqqmpSVlZWZozZ06H+4SEhCgrK0vPPfec/vjHP+qJJ57Qm2++qZtvvvl0PwIAAAAAAAAABIXA/lmZAJCZmamYmBiVlpZq0aJF3vGWlhaVl5dr9uzZCgkJ0eLFiyVJL7/8cof79OvXT9dff7337SFDhmjWrFl66KGHTu8DAAAAAAAAAAgYps/Q7uno0PYhLCxM+fn5Kisrk9v9zzM1Kyoq1NDQoIKCgn9r37q6Om3cuFHf+973uisqAAAAAAAAAAQ1Ctqd4HA4VFtbqy1btnjHnE6nMjIyFB8ff0p7zZo1S4MGDVJycrKioqL0+OOPd3NaAAAAAAAAAIHKZvP47QpGFLQ7ITExUWlpaXI6nZL+t7t68+bNcjgcp7zX0qVL9eabb6q0tFTV1dVauHBhd8cFAAAAAAAAgKDEGdqdVFhYqJtvvlmNjY0qKytTdHS0pk6desr7xMbGKjY2ViNHjlS/fv30gx/8QLfeeuspd3oDAAAAAAAA6Hk4Q7tr6NDupOzsbEVERKi8vFxOp1N5eXkKDw/v0p7/OJO7ubm5OyICAAAAAAAAQFCjQ7uTIiMjlZOTo6KiIrlcrnbHjdTU1KixsVF79+6VJO3cuVOSNHz4cEVFRem1117TwYMHlZKSorPPPluffvqp7rjjDl188cUaPny4358HAAAAAAAAgP8F69nW/kJB+xQ4HA6VlJQoNTVVSUlJbeaWLl2qZ5991vv2hAkTJEkbN27UpZdeqt69e+vpp59WVVWVmpubde655yorK0s/+9nP/PoMAAAAAAAAANBTUdA+BSkpKXK5XB3OFRcXq7i4+KT3Tpo0SZMmTTo9wQAAAAAAAAD0CJwB3TV8/AAAAAAAAAAAPQId2gAAAAAAAADgJ/49Q9vmx/flH3RoAwAAAAAAAAB6BDq0A9TGL8eZjmBp2hv/aTqCT79KOPmZ5oFg+tMTTUewtLXoA9MRLB08FNivCOxxB3a+OXEvmY5g6RfbppiOYOnIoeOmI1i66q7vmY5g6anZG0xHsFT+QLPpCJZy5x80HcGn0ofiTEewtGB9YOeLPDvCdARL/QeeZTqCpV9GPmw6gqXjyammI1iKaPzCdARLtUO+bzqCJfdZoaYjWHKpv+kIls5t+cx0BJ96HT9sOoKl0G8CO9+JqGjTESx9cevPTUewFL1ylekI6CY20aHdFXRoAwAAAAAAAAB6BDq0AQAAAAAAAMBP/HuGdvChQ9uH3NxcZWdndzhXVVUlu92uyspKrVixQpmZmRo8eLDsdrvlng0NDUpOTpbdbldDQ8NpSA0AAAAAAAAAwYeCtg+FhYXaunWrqqur282tW7dOCQkJmjhxopqampSVlaU5c+b43PPGG2/UBRdccDriAgAAAAAAAEDQoqDtQ2ZmpmJiYlRaWtpmvKWlReXl5ZoxY4ZCQkK0ePFi3XTTTRozZozlfsXFxTp27Jjmzp17OmMDAAAAAAAACEA2efx2BSMK2j6EhYUpPz9fZWVlcrvd3vGKigo1NDSooKCg03vt2LFDjzzyiJ588kmFhPChBwAAAAAAAIBTQVW1ExwOh2pra7VlyxbvmNPpVEZGhuLj4zu1x9dff63Zs2fr/vvv1+DBg09TUgAAAAAAAACBzGbz+O0KRhS0OyExMVFpaWlyOp2SpLq6Om3evFkOh6PTe/z85z9XamrqSV9gEgAAAAAAAABgjYJ2JxUWFmrTpk1qbGxUWVmZoqOjNXXq1E7f/+abb6qsrEz9+/dX//79vYXtkSNH6p577jldsQEAAAAAAAAEEM7Q7pow0wF6iuzsbN1+++0qLy+X0+lUXl6ewsPDO33/iy++qObmZu/bf/rTnzRv3jy98sorSkxMPB2RAQAAAAAAACCoUNDupMjISOXk5KioqEgul6vdcSM1NTVqbGzU3r17JUk7d+6UJA0fPlxRUVEaMWJEm/UNDQ2S/rdDu3///n54AgAAAAAAAACmhQRp57S/cOTIKXA4HHK5XEpNTVVSUlKbuaVLl2rChAn65S9/KUmaMGGCJkyYoI8++shEVAAAAAAAAAAIOnRon4KUlBS5XK4O54qLi1VcXNzpvS699NKT7gUAAAAAAAAgONlsdGh3BR3aAAAAAAAAAIAegQ5tAAAAAAAAAPATG2dodwkd2gAAAAAAAACAHoEObQAAAAAAAADwEzq0u8bmcrn4CAag6vom0xEsJX71lukIPn06YJLpCJbOO/I/piNYqu17gekIluytB0xHsFTtHmY6gqXeYc2mI/Rore5Q0xEsDW/+i+kIlvZFDjcdwVL0iXrTESw1h0WajuBTwc/2mY5gqfShONMRLJ0ICTcdwVKIx206gqWa44NMR7CU0vSO6QiWPu+TYjqCpWFfB/a/cSfCIkxHsPRNhN10BEu9Wo+bjuDTN+HnmI5gqbE12nQES31Cj5qOYCn6eJ3pCJa+ikgwHcHSuQPPNh2hx6j96hu/va/4gWf57X35Cx3aAAAAAAAAAOAndGh3DWdoAwAAAAAAAAB6BAraPuTm5io7O7vDuaqqKtntdlVWVmrFihXKzMzU4MGDZbfbO1xvt9vbXWvWrDmN6QEAAAAAAAAEEpvN47crGFHQ9qGwsFBbt25VdXV1u7l169YpISFBEydOVFNTk7KysjRnzhzL/R599FFVVVV5r/z8/NMVHQAAAAAAAACCCmdo+5CZmamYmBiVlpZq0aJF3vGWlhaVl5dr9uzZCgkJ0eLFiyVJL7/8suV+ffv2VWxs7GnNDAAAAAAAACAwcYZ219Ch7UNYWJjy8/NVVlYmt/ufr+heUVGhhoYGFRQUnNJ+CxYs0PDhw5Wenq41a9a02RMAAAAAAAAAcHIUtDvB4XCotrZWW7Zs8Y45nU5lZGQoPj6+0/ssWrRIa9as0UsvvaTp06frF7/4hR544IHTkBgAAAAAAABAILLJ47crGHHkSCckJiYqLS3NW8Suq6vT5s2bT/kFHW+//Xbvr8eMGSO3260HHnhAt912W3dHBgAAAAAAAICgQ4d2JxUWFmrTpk1qbGxUWVmZoqOjNXXq1C7tOW7cOB0+fFj19fXdlBIAAAAAAABAIKNDu2soaHdSdna2IiIiVF5eLqfTqby8PIWHh3dpz48//li9e/dW3759uyklAAAAAAAAAAQvjhzppMjISOXk5KioqEgul0sOh6PNfE1NjRobG7V3715J0s6dOyVJw4cPV1RUlCoqKlRfX6+LL75YkZGR2rZtm5YtW6brrrtOERERfn8eAAAAAAAAAP4XrJ3T/kJB+xQ4HA6VlJQoNTVVSUlJbeaWLl2qZ5991vv2hAkTJEkbN27UpZdeqvDwcK1evVqLFy+W2+3WsGHDtHDhQv34xz/26zMAAAAAAAAAQE9FQfsUpKSkyOVydThXXFys4uLik9572WWX6bLLLjtNyQAAAAAAAAD0BDa5TUfo0ThDGwAAAAAAAADQI1DQBgAAAAAAAAD0CBw5AgAAAAAAAAB+YrPxopBdQYc2AAAAAAAAAKBHoEM7QJVtOcd0BEtzPyo1HcGnXw9NNx3B0txtRaYjWFp9+QbTESxdOHaw6QiW3n6nwXQESw+nv2U6gqX5WyeajmDpwJeNpiNY+q+3f2Y6gqX7Up42HcFS6TyX6QiWFqyPMx3Bp9KHTCewVvCzfaYjWIo4O9J0BEsRZwV2vqfOnmk6gqXQa681HcFS4okPTEewdCIiynQES8d79TEdwVJk82HTESwd6h1jOoJP5zQF9tf5Z4UE9u9xs6236QiWWlY/bDqCpf4/udV0BB9GmA7QY9g8dGh3BR3aAAAAAAAAAIAegQ5tAAAAAAAAAPATm+jQ7go6tH3Izc1VdnZ2h3NVVVWy2+2qrKzUihUrlJmZqcGDB8tut590v/Lycn3/+99XbGyshg8frhtuuOE0JQcAAAAAAACA4EJB24fCwkJt3bpV1dXV7ebWrVunhIQETZw4UU1NTcrKytKcOXNOuteTTz6pO+64QzfddJPeffddbdy4UVOnTj2d8QEAAAAAAAAEEJvH7bcrGHHkiA+ZmZmKiYlRaWmpFi1a5B1vaWlReXm5Zs+erZCQEC1evFiS9PLLL3e4j8vl0t13363S0lKlp//zxQpHjRp1eh8AAAAAAAAAAIIEHdo+hIWFKT8/X2VlZXK7//ldjYqKCjU0NKigoKBT+1RWVqq1tVX19fVKTU1VcnKyCgoKtGfPntOUHAAAAAAAAECgscnjtysYUdDuBIfDodraWm3ZssU75nQ6lZGRofj4+E7tsWfPHrndbq1YsUL33XefnE6nTpw4oaysLH3zzTenKTkAAAAAAAAABA8K2p2QmJiotLQ0OZ1OSVJdXZ02b94sh8PR6T3cbrdaWlp0//3367LLLtO4ceP01FNP6cCBA3rttddOV3QAAAAAAAAAAYQztLuGgnYnFRYWatOmTWpsbFRZWZmio6NP6QUdY2NjJUlJSUnesb59+youLk61tbXdnhcAAAAAAAAAOrJ69WqNGTNGsbGxmjhxot555x3L9X/5y180depUxcXFKTk5Wffff788nrZHmrz11luaOHGiYmNjNXbsWK1Zs+a0ZKeg3UnZ2dmKiIhQeXm5nE6n8vLyFB4e3un7L7nkEknSZ5995h07evSo9u/fr4SEhG7PCwAAAAAAACDwmD5De/369VqwYIHmz5+vrVu3avz48crJyVFNTU2H6w8fPqyrrrpKMTExeuONN1RUVKSVK1fqscce867Zs2ePrrnmGo0fP15bt27Vf/3Xf+n222/Xyy+/3O0fPwranRQZGamcnBwVFRVp9+7d7Y4bqamp0c6dO7V3715J0s6dO7Vz504dPXpUkjRixAhNnTpVCxYs0HvvvadPP/1Uc+fO1YABA5SZmen35wEAAAAAAABw5nn88cd17bXX6rrrrlNSUpKWL1+u2NjYk3ZUv/DCCzp27JiKi4v17W9/W9nZ2br55pv1xBNPeLu0n376acXFxWn58uVKSkrSddddp/z8/DZF7+5CQfsUOBwOuVwupaamtjk6RJKWLl2qCRMm6Je//KUkacKECZowYYI++ugj75pVq1bpoosuUl5enjIzM3X8+HFt2LBBZ511ll+fAwAAAAAAAIAZJs/Qbm5u1vbt25WRkdFmPCMjQ++//36HeT/44AN997vfVWRkpHds8uTJqqurU3V1tXfNv+45efJkffTRR2ppaenqh6yNsG7dLcilpKTI5XJ1OFdcXKzi4mLL+/v06aOVK1dq5cqVpyEdAAAAAAAAAJxcQ0ODWltbNXDgwDbjAwcOVH19fYf31NfXa/Dgwe3W/2Nu2LBhqq+v16RJk9qtOXHihBoaGhQXF9dtz0BBGwAAAAAAAAD85GRnW/s1g83W5m2Px9NuzNf6fx3vzJruwJEjAAAAAAAAAHAG6N+/v0JDQ9t1Yx84cKBd1/Y/xMTEdLhe+men9snWhIWFqV+/ft0VXxIFbQAAAAAAAADwG5vH47frX/Xq1UspKSmqrKxsM15ZWanU1NQO844fP17vvvuujh8/3mb9oEGDNHToUO+aLVu2tNvzwgsvVHh4eBc/Ym1x5EiAamo6YTqCJY/b/I9G+OLp4C9tIAmLCOy/ft983Ww6gqWQ7v1plW7X2tJqOoKlkKZjpiNYamkO7M+Bga7X2YH9+SWsm7+Y6W4h7u59wZLuFnl2hOkIPp0ICezf44izI30vMqjp68D+HB0SGmo6gqXQ6MD+HGg7EdifY06cFW06giWbO7C/xnLbAvzvR4D/GxeqwP79lSR3SGD/HreGBPbnwOaQ3qYjWOp9VmB/neW20ZeK7jF37lzdcMMNGjdunFJTU7VmzRrt27dPM2fOlCQtWbJEH374oTZs2CBJ+o//+A/df//9uvHGG3Xrrbfqs88+08MPP6zbb7/de5zIzJkz9etf/1oLFizQzJkz9f7776usrEyrV6/u9vyB/ZkOAAAAAAAAAIKIzeM2+v6nT5+ugwcPavny5dq/f7+Sk5P1/PPPa8iQIZKkffv2affu3d71ffv21Ysvvqhbb71V6enpstvtmjt3rubNm+ddM2zYMD3//PNatGiR1qxZo7i4ON1///3Kzs7u9vwUtAEAAAAAAADgDDJ79mzNnj27w7ni4uJ2Y6NGjVJFRYXlnt///ve1devWbslnhZ9V8CE3N/ek30moqqqS3W5XZWWlVqxYoczMTA0ePFh2u73d2tLSUtnt9g6vP/3pT6f5KQAAAAAAAAAEAps8fruCEQVtHwoLC7V161ZVV1e3m1u3bp0SEhI0ceJENTU1KSsrS3PmzOlwn+nTp6uqqqrNdc0112jo0KG68MILT/djAAAAAAAAAECPx5EjPmRmZiomJkalpaVatGiRd7ylpUXl5eWaPXu2QkJCtHjxYknSyy+/3OE+kZGRioz85wsQffPNN3rttdd08803ew9PBwAAAAAAABDcTJ+h3dPRoe1DWFiY8vPzVVZWJrf7n3/YKioq1NDQoIKCgn9r3xdffFHffPPNv30/AAAAAAAAAJxpKGh3gsPhUG1trbZs2eIdczqdysjIUHx8/L+159q1a5WZmam4uLhuSgkAAAAAAAAg0Nk8Hr9dwYiCdickJiYqLS1NTqdTklRXV6fNmzfL4XD8W/t98skn+uCDD3Tdddd1Z0wAAAAAAAAACGoUtDupsLBQmzZtUmNjo8rKyhQdHa2pU6f+W3v95je/UXx8vC677LJuTgkAAAAAAAAAwYuCdidlZ2crIiJC5eXlcjqdysvLU3h4+Cnvc/z4cZWXl6ugoEAhIXz4AQAAAAAAgDOJTW6/XcEozHSAniIyMlI5OTkqKiqSy+Vqd9xITU2NGhsbtXfvXknSzp07JUnDhw9XVFSUd93LL7+sw4cPa8aMGf4LDwAAAAAAAABBgIL2KXA4HCopKVFqaqqSkpLazC1dulTPPvus9+0JEyZIkjZu3KhLL73UO7527VpNnjxZCQkJ/gkNAAAAAAAAIHAE6Ys1+gsF7VOQkpIil8vV4VxxcbGKi4t97vHqq692cyoAAAAAAAAAODNQ0AYAAAAAAAAAP7F5gvNsa3/hVQkBAAAAAAAAAD0CHdoAAAAAAAAA4Cc2cYZ2V9ChDQAAAAAAAADoEejQxr8lJCzUdASfevUO7D/eZ/U/23SEHu2s3oF93pQtxGY6giV3RKTpCJZ6R4abjmCpKaq36QiW+sZHm45gKcoeZTqCJY8C++9v/4FnmY7gU4inwXQESxFnBfbnwJDQwP4669jho6YjWAr7VoTpCJbc4YGdz2ML7J6nEE+L6Qg9WljLMdMRLJ1QYH8NKEmtIYH9/8ym0MD+OqHZE9ifAyNj+puOYOmILbC/RkDncYZ21wT2VysAAAAAAAAAAPw/gf2tRQAAAAAAAAAIJh7O0O4KOrR9yM3NVXZ2dodzVVVVstvtqqys1IoVK5SZmanBgwfLbrd3uP5Pf/qTsrOzNXToUA0ZMkTTpk3Thx9+eBrTAwAAAAAAAEDwoKDtQ2FhobZu3arq6up2c+vWrVNCQoImTpyopqYmZWVlac6cOR3uc/ToUV199dWKi4vT7373O/3+979XXFycpk+friNHjpzuxwAAAAAAAAAQAGwet9+uYERB24fMzEzFxMSotLS0zXhLS4vKy8s1Y8YMhYSEaPHixbrppps0ZsyYDvf5+9//rsbGRi1cuFBJSUlKSkrSokWLdOjQIX322Wf+eBQAAAAAAAAA6NEoaPsQFham/Px8lZWVye3+53c1Kioq1NDQoIKCgk7tM2LECA0YMEBOp1NNTU1qamrSM888o/j4eJ1//vmnKz4AAAAAAACAAGLzePx2BSMK2p3gcDhUW1urLVu2eMecTqcyMjIUHx/fqT369OmjV155RevXr9egQYM0aNAgrV+/Xi+99JIiIyNPU3IAAAAAAAAACB4UtDshMTFRaWlpcjqdkqS6ujpt3rxZDoej03scO3ZM8+bN08UXX6w//OEPev311zVmzBhde+21+vrrr09XdAAAAAAAAACBxOP23xWEwkwH6CkKCwt18803q7GxUWVlZYqOjtbUqVM7ff8LL7yg3bt36/XXX1doaKgkafXq1Ro2bJheeeUV5ebmnq7oAAAAAAAAABAU6NDupOzsbEVERKi8vFxOp1N5eXkKDw/v9P3Hjh2TzWZTSMg/P+QhISGy2WxtzuYGAAAAAAAAELxsHrffrmBEQbuTIiMjlZOTo6KiIu3evbvdcSM1NTXauXOn9u7dK0nauXOndu7cqaNHj0qS0tPTdeTIEc2fP19VVVX65JNPdOONNyo0NFQTJkzw+/MAAAAAAAAAQE9DQfsUOBwOuVwupaamKikpqc3c0qVLNWHCBP3yl7+UJE2YMEETJkzQRx99JEkaOXKknnvuOf31r3/V5ZdfriuuuEJffvmlXnjhBZ177rl+fxYAAAAAAAAA/meTx29XMOIM7VOQkpIil8vV4VxxcbGKi4st709PT1d6evppSAYAAAAAAAAAwY+CNgAAAAAAAAD4S5Cebe0vHDkCAAAAAAAAAOgR6NAGAAAAAAAAAD+xefx3tnUwnqJNhzYAAAAAAAAAoEewuVyuYCzU93gR28pMR7D0UWKB6Qg+Xfi3Z0xHsPSHhJ+ajmApY+d9piNYG55sOoGlkP01piNY+nDUDaYjWBr3aYnpCJbcroOmI1haf95dpiNY+tHOn5uOYOlvlwd2vm9/uNp0BJ8+GDPPdARLw5+caTqCpdBegf1DlGGREaYjWLpqx3+ajmBp3cPnmo5g6ainj+kIljwem+kIlvp79puOYOlgSIzpCJaGf/Wu6Qg+1cWMNR3Bks0W2CWec44fMB3B0s7WMaYjWDr3rMD++A0aENj/hgSSkL+97bf35R75Pb+9L3+hQxsAAAAAAAAA0CNQ0AYAAAAAAAAA9AgUtH3Izc1VdnZ2h3NVVVWy2+2qrKzUihUrlJmZqcGDB8tut3e4/s0339SUKVMUHx+vpKQk3XnnnTpx4sRpTA8AAAAAAAAgoHg8/ruCEAVtHwoLC7V161ZVV1e3m1u3bp0SEhI0ceJENTU1KSsrS3PmzOlwnz//+c/KyclRenq6tm7dqpKSElVUVOiuu+46zU8AAAAAAAAAAMGBgrYPmZmZiomJUWlpaZvxlpYWlZeXa8aMGQoJCdHixYt10003acyYjl9AYP369UpKStLChQs1fPhwff/739eSJUu0evVqHTlyxB+PAgAAAAAAAMAwm8fttysYUdD2ISwsTPn5+SorK5Pb/c8/BBUVFWpoaFBBQUGn9mlqalLv3r3bjEVGRur48ePavn17d0YGAAAAAAAAgKBEQbsTHA6HamtrtWXLFu+Y0+lURkaG4uPjO7XH5MmT9cc//lHPPfecTpw4oS+//FL333+/JGn//v2nIzYAAAAAAACAQMMZ2l1CQbsTEhMTlZaWJqfTKUmqq6vT5s2b5XA4Or1HRkaG7rnnHt1+++2KjY3VRRddpClTpkiSQkNDT0tuAAAAAAAAAAgmFLQ7qbCwUJs2bVJjY6PKysoUHR2tqVOnntIe8+bNU3V1tf785z9r165d3vuHDh16OiIDAAAAAAAACDCcod01FLQ7KTs7WxERESovL5fT6VReXp7Cw8NPeR+bzaZBgwYpMjJS//3f/634+HiNHTv2NCQGAAAAAAAAgOASZjpATxEZGamcnBwVFRXJ5XK1O26kpqZGjY2N2rt3ryRp586dkqThw4crKipKkvToo49q8uTJCgkJ0caNG/Xwww/r6aef5sgRAAAAAAAA4EwRpJ3T/kJB+xQ4HA6VlJQoNTVVSUlJbeaWLl2qZ5991vv2hAkTJEkbN27UpZdeKkn6/e9/rxUrVqi5uVmjR49WWVmZLr/8cv89AAAAAAAAAAD0YBS0T0FKSopcLleHc8XFxSouLra8f+PGjachFQAAAAAAAICewubxmI7Qo3GGNgAAAAAAAACgR6BDGwAAAAAAAAD8xc0Z2l1BhzYAAAAAAAAAoEegQztQnWgxncBS47FI0xF8azpuOoGlXqGB/d24kH4DTEewdLxPYOeL2PuZ6QiWztOnpiNYa2kyncBSSFQf0xEsnR/jMh3BUsiYi01HsBTn+cJ0BEvHk1NNR/Appekd0xEshV57rekIlmwB/nWgOzzCdARL62aeazqCJcctgf055uVx60xHsLQpfbXpCJYuOveE6QiWBp740nQESw0DkkxH8OnslkOmI1iKOrrPdARL9f1Gmo5gqV/rUdMRLH3T2gNqMegcztDuEjq0AQAAAAAAAAA9Ah3aAAAAAAAAAOAvnsD+qf1AR4e2D7m5ucrOzu5wrqqqSna7XWvXrtW8efM0duxYxcXFaezYsVqyZImOHTvWZn1NTY1yc3M1ePBgDR8+XLfffruam5v98RgAAAAAAAAA0OPRoe1DYWGhZsyYoerqag0dOrTN3Lp165SQkKBBgwaptbVVDz74oBITE1VVVaVbbrlFBw8e1COPPCJJam1tVW5urqKjo/Xqq6+qsbFRc+bMkcfj0fLly008GgAAAAAAAAA/s3GGdpfQoe1DZmamYmJiVFpa2ma8paVF5eXlmjFjhqZMmaLi4mJNnjxZw4YNU2ZmpubPn68NGzZ417/xxhv65JNPtGrVKqWkpCg9PV1LlizRM888o8OHD/v7sQAAAAAAAACgx6Gg7UNYWJjy8/NVVlYmt/uf59tUVFSooaFBBQUFHd535MgR2e1279sffPCBkpKSFB8f7x2bPHmympqatH379tMVHwAAAAAAAEAg8bj9dwUhCtqd4HA4VFtbqy1btnjHnE6nMjIy2hSo/6GmpkYrV67UrFmzvGP19fUaOHBgm3X9+/dXaGio6uvrT1t2AAAAAAAAAAgWFLQ7ITExUWlpaXI6nZKkuro6bd68WQ6Ho93a+vp6XX311UpPT9fcuXPbzNlstg73P9k4AAAAAAAAgCBDh3aXUNDupMLCQm3atEmNjY0qKytTdHS0pk6d2mbN/v37deWVVyo5OVmrVq1qU6iOiYlp14nd0NCg1tbWdp3bAAAAAAAAAGBaU1OTbrvtNg0fPlyDBw9WXl6evvjiC8t71q5dqx/84AcaNmyYhgwZoqysLL377rtt1ixbtkx2u73NNXLkyE5loqDdSdnZ2YqIiFB5ebmcTqfy8vIUHh7und+3b5+ysrI0cuRIlZSUKCwsrM3948ePV1VVVZvf8MrKSkVERCglJcVfjwEAAAAAAADAIJvH47erqxYuXKiNGzeqpKREr776qo4cOaLc3Fy1trae9J633npLV111lV5++WVt3rxZ5513nq6++mrt2rWrzbrzzjtPVVVV3uudd97pVKYw30sgSZGRkcrJyVFRUZFcLleb40bq6uqUlZWluLg4LVu2TA0NDd65AQMGKDQ0VBkZGUpOTtZPf/pT3XvvvWpsbNQdd9yhwsJCnXPOOSYeCQAAAAAAAAA6dOjQIa1bt06PP/640tPTJUmrVq3SBRdcoC1btmjy5Mkd3vfrX/+6zdsPPvigNm3apD/84Q9KTEz0joeFhSk2NvaUc9GhfQocDodcLpdSU1OVlJTkHX/jjTe0a9cuvf322xo9erSSkpK8V21trSQpNDRU5eXlOuuss3TFFVdo5syZysrK0r333mvqcQAAAAAAAAD4m9vtv6sLtm/frpaWFmVkZHjH4uPjlZSUpPfff7/T+zQ3N+v48eOy2+1txvfs2aPk5GSNGTNG119/vfbs2dOp/ejQPgUpKSlyuVztxgsKClRQUODz/oSEBJWXl5+GZAAAAAAAAADQferr6xUaGqr+/fu3GR84cGC71wq0cu+99yoqKko/+MEPvGMXXXSRnnjiCZ133nk6cOCAli9frilTpui9995Tv379LPejoA0AAAAAAAAA/tINZ1t3xb333qsVK1ZYrtm4ceNJ5zwej2w2W6feV3FxsX7zm9/opZdeanPs8uWXX95m3UUXXaSUlBSVlZVp3rx5lntS0AYAAAAAAACAM8ScOXN0zTXXWK6Jj4/X//zP/6i1tVUNDQ0aMGCAd+7AgQNKS0vz+X6Ki4t133336YUXXtC4ceMs10ZFRen888/X559/7nNfCtoAAAAAAAAAcIbo379/u2NEOpKSkqLw8HBVVlYqJydHkvTFF1+oqqpKqamplvc+9thjWrZsmZ5//nl997vf9fm+jh8/rr///e+69NJLfa6loA0AAAAAAAAA/uLp2os1+kvfvn3lcDh0xx13aODAgYqOjtbixYs1atQoTZo0ybtu2rRpGjdunO68805J0qOPPqp77rlHTz31lEaMGKH9+/dLknr37q2+fftKkn7xi1/oiiuuUHx8vPcM7W+++Ub5+fk+c1HQDlD1oy73vcigb9t2m47g077vTDMdwdLo1k9NR7D05agrTEewFOo5YTqCpdbv5pqOYCn6aK3pCJb+euEs0xEsnR36jekIlsLcgf3347h9kOkIlty2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AAAAAMD/c0Z3aN96663yeDyqrKzUVVddpfPOO09JSUn6yU9+orfeeqvDex577DGlpaVp8ODBSk5O1k033SSXy+WdP3TokH7yk59oxIgRio2N1dixY/XEE094559++mmNGzdOsbGxSkxM1PTp03XixInT/agAAAAAAAAAAoC71eO3KxidsR3ajY2N+sMf/qBf/OIXioqKajdvt9s7vC8kJETLli3TsGHDVFNTo9tvv1233367nnrqKUnSvffeq7/+9a8qLy/XgAEDtHfvXjU0NEiSPvroI916660qLi7WJZdcokOHDmnr1q2n7RkBAAAAAAAAIJicsQXtzz//XB6PRyNHjjyl+2688Ubvr4cOHaq7775b1157rZ588kmFhISopqZGY8aM0bhx47xr/qGmpkZnn322fvCDH6hPnz6S/vdFJwEAAAAAAACcGTzu4Oyc9pcztqDt8fx7f3DefPNNPfTQQ/rb3/6mw4cPq7W1Vc3Nzdq/f78GDRqkWbNm6brrrtOOHTuUnp6uK664Qt///vclSenp6YqPj9fYsWM1efJkpaen68orr/QWtwEAAAAAAAAAJ3fGnqGdmJgom82mv/3tb52+Z+/evcrNzdXIkSP1m9/8Rlu2bNFjjz0mSWpubpYkXX755fr444910003qaGhQbm5ud6u7j59+mjr1q16+umnFR8fr4ceekjjx49XXV1d9z8gAAAAAAAAgIDDGdpdc8YWtKOjozV58mT9+te/1tGjR9vN/98XevyHjz76SM3NzVq2bJnGjx+vESNGdFiM7t+/v/Ly8lRcXKyVK1fq2WefVVNTkyQpLCxMEydO1J133qm3335bX3/9tV5//fVufz4AAAAAAAAACDZn7JEjkrRixQplZmYqPT1dixcv1qhRo+TxeLRt2zY99NBD+vOf/9xmfWJiotxut5544gldeeWV+uMf/6gnn3yyzZr77rtPY8eOVXJysk6cOKGNGzdq2LBhioiI0Guvvabdu3crLS1N0dHR2rZtm44ePXrK53gDAAAAAAAA6Jk8rW7TEXq0M7qgPWzYML355pt64IEHdOedd6qurk79+vXT6NGj9dBDD7VbP3r0aBUVFemRRx7Rfffdp/Hjx+uee+7RzJkzvWsiIiJ07733qrq6WhEREbr44ov13HPPSZL69u2rTZs26Ve/+pWOHTumb33rW3r00UeVlpbmt2cGAAAAAAAAgJ7qjC5oS1JcXJyWL1+u5cuXdzj/8ccft3n7pz/9qX7605+2Gbvqqqu8v7711lt16623drjXd7/7Xb3yyitdTAwAAAAAAACgp/K4g/Nsa385Y8/QBgAAAAAAAAD0LGd8hzYAAAAAAAAA+Iu7lQ7trqBDGwAAAAAAAADQI9ChDQAAAAAAAAB+4qFDu0tsLpeLj2AAOlb3uekIlvru+9R0BJ8aB40yHcFSS2iE6QiW7IdrTEewtCvqQtMRLEWENJuOYClKh01H6NHC3IH9+xvibjUdwdKJ0F6mI1g6ERLY+ULdLaYj+HTU1td0BEv9W+pMR7DksQX2D1G6baGmI1g6HNbPdARL37RGmo5gqV9Ig+kIliJOfGM6gqVzGveYjmDpq4HfNh3Bkv3ol6Yj+HQiLLD/H3cw8lzTESz1O/aF6QiWDkQmmI5gafYtu0xHsLTJmWI6Qo+x/ZKJfntfKe+96bf35S90aAMAAAAAAACAn7hP0F/cFYHd/gEAAAAAAAAAwP/TYwvaeXl56tevnyorK01HOWVz5sxRbm6u6RgAAAAAAAAA/MzT4vHbFYz8XtB2u91qbe3a2Z779u3T1q1bdeONN+qZZ57ppmQAAAAAAAAAgEDms6D9wx/+UPPnz9fdd9+t4cOHa8SIEfrFL34ht9stSXK5XPrpT3+qoUOHKi4uTtnZ2frkk0+895eWlurcc8/V7373O333u9/VwIEDVVVVpQsuuED333+/5syZo/j4eI0aNUrr16+Xy+XS9ddfr3PPPVff+c539MYbb7TLVFZWpsmTJ+uGG25QRUWFDh482Gb+Hx3QDz/8sEaOHKkhQ4borrvuktvt1rJlyzRixAiNHDlSDz/8cJv7ampqVFBQoPj4eMXHx2vGjBn64ot/vmDBsmXL9N3vfrfNPf94vn9d89vf/lYpKSmKj4/Xtddeq4aGBu/8s88+q9dff112u112u13btm3z9dsAAAAAAAAAIAi4T3j8dgWjTnVov/DCCwoNDdXvfvc7LV++XMXFxVq/fr2k/y0ef/jhhyorK9PmzZsVGRmp//iP/9CxY8e89x8/flwrVqzQQw89pPfff18JCf/7qrHFxcUaN26c3nzzTf3oRz/SnDlz9OMf/1iXX365tm3bprS0NP3kJz/R8ePHvXt5PB45nU5dc801SkhI0Lhx4/Tcc8+1y/zOO++ourpar7zyih588EE98sgjysnJUXNzs1577TUtWLBAd911l7Zv3+7dt6CgQF999ZU2bNigjRs3at++fSooKJDHc2q/+Xv37tX69evldDq1fv167dy5U/fcc48k6aabbtJVV12lSZMmqaqqSlVVVUpNTT2l/QEAAAAAAADgTBTWmUVJSUlavHixJGnEiBFau3at3nzzTV144YWqqKjQpk2b9L3vfU+StGrVKl1wwQV64YUXVFhYKElqbW3Vr371K6WkpLTZd/LkyZo9e7YkaeHChXr88cf1rW99S/n5+ZKk2267TU6nU5988okuvPBCSdK2bdvU2NiozMxMSf97lnZxcbFuvPHGNnufc845WrFihUJDQzVy5Eg99thjqqur029/+1vvczz00EPatm2bUlJStGXLFv35z3/WRx99pKFDh0qSVq9erQsvvFBvvvmmJk2a1OkP6okTJ/TEE0+ob9++kqT//M//VGlpqSQpKipKvXv3VkREhGJjYzu9JwAAAAAAAICeL1jPtvaXTnVojxo1qs3bcXFx+uqrr1RVVaWQkBCNHz/eO9e3b199+9vf1qeffuodCwsL0wUXXGC5b1RUlM4666w2YzExMZKkr776yjvmdDp11VVXqVevXpKk7Oxs7d69W3/84x/b7J2UlKTQ0NA2e/3rc8TExHj3rqqq0qBBg7zFbEkaNmyYBg0a1OZZOiMhIcFbzJb+9+N14MCBU9oDAAAAAAAAANBWpzq0w8PD27xts9nk8Xgsj+Kw2WzeX0dERLQpLlvtGxYW1uZtSW3O696wYYOam5u1du1a77rW1lY988wzuuiiizq99z/G/rG3x+Npk7mjZwkJCWn3zCdOnOjUc/3j/QAAAAAAAAA4cwXr2db+0qkO7ZM5//zz5Xa79cEHH3jHDh8+rL/+9a9KSkrqcrh/9cILL2jAgAF66623tG3bNu/1yCOP6MUXX9TXX3/9b+99/vnn68svv1R1dbV3bM+ePaqrq9P5558vSRowYIDq6+vbFLU//vjjU35fvXr1Umtr67+dFQAAAAAAAADORF0qaCcmJmrq1Kn62c9+pnf+//buPbqmO+/j+OdISIKqKIImpHHJxS2qLnVJhRql4tKmCDJUK5rSpy19pqEMxfRR1KhxaYuOCVFUFWGGzrikqmhmiKgqWlJRhCBRIW4nzx8mZ5xxqdjkdxLv11pZS87Oat7rqPxOfmfv7/76a+3evVsxMTF64IEH9Nxzz92tRof58+erS5cuCgkJcfqIioqSzWZz3KjyTrRp00b16tVTTEyMUlJStGPHDg0cOFANGzZUWFiYJKlVq1Y6ffq03nvvPR08eFDx8fFasWJFgb9X9erVtWfPHu3fv18nT57UpUuX7rgbAAAAAAAAQNGRd8leaB/FkaUNbUmaOXOmHn30UUVFRaldu3Y6f/68li5dKi8vr7vR55CSkqLU1FR17dr1umOlSpVSx44dNX/+/Dv+79tsNiUkJOihhx5S586dFRERocqVKyshIcExciQwMFBTpkzRvHnz1LJlS23cuFFDhw4t8Pfq16+f6tSpo/DwcNWsWVNbt269424AAAAAAAAAuF/YsrKyGNrigs4fPWA64ZYePFawG2WacLpq3V//IoMuuXmYTril8mfSTSfc0o9lG5lOuCWPEhdNJ9xSWZ0xnVCkudtd+++3hN21x1pdditlOuGWLpdw7T43u+tf2XXW9uCvf5FBD106ajrhlvJsls85uafstuvvjeNKzrhXMJ1wS+eu3N0Tf+62CiVOmk64JY/L50wn3FK502mmE27pRKUQ0wm3VP7sEdMJv+qyu2v/HnfK62HTCbdU4fzPphNuKdPLz3TCLb342o+mE25p9YJQ0wlFxha/5oX2vR5PL34n0t7WTSEBAAAAAAAAANZxU0hrXPv0DwAAAAAAAAAA/o0ztAEAAAAAAACgkORd4gxtKzhDGwAAAAAAAABQJHBTSAAAAAAAAABAkcAZ2gAAAAAAAACAIoENbQAAAAAAAABAkcCGNgAAAAAAAACgSGBDGwAAAAAAAABQJLChDQAAAAAAAFhkt9tlt9sdn2dkZCg+Pl5bt241WAUUP+6mA1B4cnJylJKSopYtW5pOuSm73a6ff/5Zfn5+RjsuXryoUqVKOT7funWrLly4oMcff9zpcVcRExOjt99+W1WrVjWdcp2srCwdOHBAPj4+evjhh03nOMnIyNDixYuVnp6u6tWrq0ePHvLx8THWs2LFCrVv316lS5c21vBr0tLSlJqaqubNm6ty5co6duyYEhISZLfb1aFDBzVo0MB0otLS0rRlyxZlZGTIzc1N1atXV3h4uMqVK2c6TZJ09uxZpaSk6Pjx47LZbKpUqZJCQ0NVtmxZ02m3dPnyZR09etT4z+ei6MiRI7p48aL8/f1Np9zQ+PHj9dJLL6lixYqmU24oMzNTDz74oEqWLGk6xUlubq7WrFnjWEM6dOggT09PYz0pKSkKDQ019v1vR05Ojvbv36/g4GB5eHjo/PnzWrVqlex2u8LCwlzidUz+6+X8NaRGjRpq2LChbDab6TRJrCH3G1dfPyTWkDvFGlJwRWENkaQePXqoXbt2io2N1dmzZxUeHq6cnBzl5OToT3/6k6Kiooz2bd68Wc2aNZO7u/N24OXLl7Vt2zaX3i8CrmXLysrKMx2BwrFr1y498cQTOnXqlLGG3NxcDR8+XCtXrlT58uX14osvKjY21nH8+PHjCgoKMtZ49OhRRUdHa/v27WrSpIkWLVqkgQMHat26dZIkf39//fWvfzW2WKakpNzw8fbt22vOnDmqUaOGJBl7MTJ27Fi98cYbKl26tC5duqQ33nhD8+fPV15enmw2mzp16qQ5c+YYe7HWrVs39e3bV5GRkUpNTVXnzp1Vrlw51axZUwcPHlR2drZWrVql+vXrG+nz9vbWAw88oMjISPXr108NGzY00nEz69atU+/evXX58mU98MADWrp0qaKjo1W6dGmVKFFCP/30kxYuXKgnn3zSSF9OTo5efvllrVy5UpIcv+hnZmbKy8tLo0eP1sCBA420SVdfJL711luKj49Xbm6u3NzcJElXrlyRp6en+vXrp3HjxrncL1z5TK8heXl5mjp1qmP9eOGFF9S5c2fHcdPrhySdOXNGr732mrZs2aJWrVpp+vTpiouL07x582Sz2dS0aVMtXrxYDz74oJG+06dPX/dYXl6eAgMDtXr1atWuXVvS1Z9FJsybN09RUVHy8PBQXl6epkyZomnTpumXX36Rp6en+vfvr/Hjx6tECTMXGMbGxqpTp06KiIhQWlqaIiIilJmZqSpVqigjI0OVKlXSihUrjG08eXt7y9/fX/369VPv3r1VuXJlIx03s337dj3zzDPKzs5W9erV9fnnnysqKkqHDx+WzWaTm5ubPvvsMz322GNG+ux2u8aMGaM5c+YoNzdX0tV/H5Lk6+uriRMnqmPHjkbaJNYQq1x9DXH19UNiDbGKNcQaV19DrlWrVi2tWLFCdevW1SeffKKpU6fqq6++0pIlSzRjxgx9/fXXRvsqVKigvXv3qlKlSk6Pnzp1SrVq1TL6WhooCEaOoFBNnDhRa9eu1YgRI9S3b19NnjxZMTExTpfk5P/yYMLo0aPl5uamhIQEPfzww+rVq5dycnK0e/dupaamysfHR1OmTDHWFx4errZt2yo8PNzp4/Lly+rfv7/juClTp05VTk6OJGnatGlatWqVPv74Y6Wmpmr+/Pnavn27pk2bZqxvx44djk3iMWPGqEuXLtq5c6dWrFihnTt3qkePHhoxYoSxPkkaOHCgNm3apPDwcIWFhenPf/6zfvnlF6NN+SZMmKCYmBgdP35cI0eOVO/evdW5c2f961//UnJysgYNGqR3333XWN9bb72ljIwMbd68Wf/6178UERGhXr16KT09Xf/3f/+n0aNH69NPPzXat3LlSr3//vv64YcflJmZqczMTP3www+aNm2aVq5cqVGjRhnrc3XTp0/XlClTFBYWpkceeUQvvviixo0b5/Q1JtcPSRo3bpy+/fZbvfbaazp69Kj69++vbdu26W9/+5sSExOVlZWl999/31hfzZo1r/uoVauWLl++rKeeekoBAQGqWbOmsb6hQ4fqzJkzkq5uTEyZMkXDhg1TYmKiRo0apQULFmjOnDnG+r744gvVqlVLkjRy5EiFhIRo79692rFjh/bt26fQ0FANHz7cWJ8kNW3aVH/84x9Vr149RUdHO96QdwVvv/22fvOb3yglJUXPPvusIiMjFRwcrLS0NKWlpalDhw4aO3assb6xY8dq7dq1+vjjj7Vs2TI1b95cY8aM0bZt29SrVy/1799f69evN9bHGmKNq68hrr5+SKwhVrGGWOPqa8i1zp4963jzacOGDercubNKliypsLAwpaWlmY2THCeb/bdTp06pTJkyBoqAO8MZ2sVIhQoVbuvrTL7jFhoaqkmTJql9+/aSpPT0dEVGRqpu3bqaM2eOMjMzjZ4dERQUpPnz56tJkyY6ffq0AgICtHz5cj3xxBOSpKSkJL366qs3PVP6XmvZsqV8fX01fvx4eXh4SLq6IDVu3FhLly5VQECAJKl69epG+ry9vbVv3z5VqlRJrVu3VkxMjKKjox3HP//8c02YMEHbtm0z0letWjV99dVXCggIUGBgoJYsWeJ0FvQPP/ygtm3b6tChQ0b6rn3+Nm3apL/85S9atWqV3Nzc1L17d/Xr109NmjQx0iZJfn5+2rRpk/z9/WW321W5cmWtX7/eMWbkxx9/VHh4uLHnr2bNmvrss88cVyhkZWUpKChIBw4cUOnSpTV79mzFx8dr06ZNxvo+/vhjx8+T/7Zx40a98MIL+vHHHwu57KpfuyLg0qVLOnbsmLGfz02bNtXw4cPVvXt3SVevWOnVq5ciIyM1fvx442fXSVK9evU0c+ZMhYWF6ejRowoJCdHChQsdZ3WuXbtWI0eOVHJyspG+4OBgNWjQQIMHD3acoZaXl6du3bpp2rRpjqt8WrVqZaTv2p+Bbdu21bPPPqvBgwc7jsfHx+vDDz/U5s2bjfRVqVJFW7dulb+/v0JCQpSQkKBGjRo5jn///ffq2LGjDh48aKQv//krW7asli1bpvj4eH3zzTfy9fVVdHS0+vTpY3T0V40aNfSPf/xDtWvX1oULF1StWjV98cUXaty4sSRpz5496tSpk7HnLzg4WHPnzlWLFi0kXR310LRpU/3444/y8PDQxIkT9Y9//ENffPGFkT7WEGtcfQ1x9fVDYg2xijXEGldfQ6712GOPafjw4XrqqafUoEEDzZs3T61bt1Zqaqq6d+9u7Od0r169JF19c6VNmzZOo1Ttdru+++47BQYG6rPPPjPSBxQUM7SLES8vL8XGxt50XMKhQ4c0evToQq5ylpGRoTp16jg+9/PzU2Jiorp06aIBAwboD3/4g8G6qxtg+eNEvL29Vbp0aadZfwEBATp27JipPK1fv14jR45Uv379NHv2bNWtW9dxrEqVKsY2sq+V/27vzz//7HiBke/RRx9Venq6iSxJV39ZSEpKUkBAgKpUqaJDhw45/QJ26NAhl5lf3bp1a7Vu3VqnT5/WwoULtWDBAiUkJCg4ONjYZWqlSpXSuXPnJEnnz5+X3W7XhQsXHMfPnz9v9FLn/FEo+cqUKaPLly/r3LlzKl26tNq2bWv07LXc3NxbvvFYoUIFx2XuJmRkZKhXr143Pbvq6NGj+uCDDwq56j/S09P16KOPOj4PDQ1VYmKiIiIidOXKFb3++uvG2vKdOHHC8cZi1apV5eXl5bgEW7q6GfDzzz+bytPmzZv18ssva8qUKfrwww8d9wyw2Wxq3LixgoKCjLXly19Dfvrpp+s27sLCwoxeRVO7dm3985//lL+/v8qVK6esrCyn49nZ2S4xZ9nLy0t9+vRRnz59tGfPHs2bN0+zZs3SxIkT1a5dOy1evNhI17XPTf6f88dm5P/Z5BmyZ8+eVbVq1Ryf+/j4KDc3V1lZWfLx8VGXLl00depUY32sIda4+hri6uuHxBpiFWuINa6+hlxr8ODBGjRokMqUKSM/Pz/HTOqvv/5aISEhxrry15C8vDyVL1/eaQxoqVKl1Lx5c/Xr189UHlBgbGgXI/Xr15e3t7e6du16w+O7du0q5KLr+fj46ODBg4538CWpcuXKWrFihSIiIvTSSy8ZrJMqVqyojIwM+fr6Sro6/uHaOXDZ2dlGL8Px8PDQpEmTtHr1akVGRuqVV17Ryy+/bKznRubOnasyZcqoVKlS153lcubMGaM31XzzzTf1wgsvyN3dXbGxsXrrrbd0+vRpBQYGav/+/ZowYYLjnWsTbvQi1tvbW4MHD9bgwYO1ZcsWxcfHGyi7qlmzZho9erReffVVLV68WI0aNdKkSZM0d+5c2Ww2TZo0yelMk8L26KOPaubMmXrvvfckSTNmzFDFihUdNyk6e/as0X+/rVq10ogRI/TRRx9dN4f/6NGjGjVqlFq3bm2o7uovy3Xr1r3pnPFdu3YZ3Yx46KGHdPjwYaf1o3bt2lq5cqUiIiJ04sQJY235KlSooJMnTzrWkE6dOjnNO83JyTH6M7BChQpatGiRZs2apfDwcE2cONFphqwrWLNmjcqVKydPT0/HCKt858+fNzb7VJKGDBmiUaNGqVKlSho6dKji4uI0ceJE1alTR/v371dcXJwiIiKM9d1oDQkODta7776rsWPHavny5UbXkNDQUP3xj39UXFyc5s+fL39/f3300UeaOXOmJOnDDz9UcHCwsb6QkBAtWbJEv/vd7yRJS5cuVZkyZRybdna73ei/X9YQa1x9DXH19UNiDbGKNcQaV19DrvX8888rNDRUhw8fVnh4uOP/u0ceeURvvfWWsa7856p69ep65ZVXGC+CIo8N7WKkffv2ys7Ovulxb29vo5t10tWzTj/99FO1adPG6XEfHx+tXLlSTz/9tJmwf6tfv76Sk5MdZxaPGTPG6fjWrVuNvqua7+mnn1ZoaKhiYmL097//3XSOg6+vrxISEiRdfZc3NTXV6bLDTZs2OZ1tUtjatWvnuMnOkSNHlJeXp1dffVXS1TcLnn/+eaNn8P7aWQWPP/64Hn/88UKqud64cePUo0cPRUREKCgoSMuWLdPQoUMdN6/x9vbW0qVLjfWNGTNG3bp104oVK1SyZEmdPHlSs2bNchz/5ptvHOOOTHjvvffUo0cP1atXT4GBgapUqZJsNpuOHz+uvXv3KigoSEuWLDHW16xZM/3www83PV62bFnHpfgmNG/eXImJidfdeb1OnTqON0VNCwkJcZrV/9+zOlNSUpyuUjIlNjZWLVq00MCBA42NT7iZV155xfHnTZs2qVmzZo7Pk5OTjd0sS5J69uyp06dPq3fv3rLb7bpy5YpjfIEkdezYUe+8846xvlutIR4eHurZs6d69uxZiEXOfv/73ysyMlKLFi1SxYoVlZiYqCFDhqh27dqy2Wz65ZdftGjRImN9I0aMUI8ePbR69Wp5enrqn//8p9OM5XXr1jlGbJnAGmKNq68hRWX9kFhD7hRriDWuvob8t0aNGjmd6HPp0iV16NDBYNF/xMXFmU4A7gpmaBcjSUlJN52r5yri4+NVrVo1Pfnkkzc8fuzYMa1fv169e/cu5LKrNmzYoPDw8JseT05Olqen503Hutxr//13bLfbNXnyZH355ZeaOXOm8ZEjv/b/YHJyskqVKvWrcxbvlfw+u92ulJQUpaWlyW63y8fHR6GhoU7jKkz1tWzZUu7urvleY/7zd+rUKafLnpOSknT+/Hk1bdr0tmf536u+wMBArV27VhcuXFBYWJhLXP56LbvdrnXr1ik5OVnHjx+XdPUqlaZNm6pt27ZGzxxydd9++61SUlLUt2/fGx7fs2ePVqxYYfRF+smTJ1WiRAmnK3uutXbtWnl6errMWn3u3Dn97ne/05dffqnly5c7Lnd3VWvWrFHJkiXVrl07ox3Z2dnasGGD0xrSvHlzozdDk6SFCxfq2WefddxjwxXl5ORo//79qlWrlsqWLavc3FwtWbJEubm5Cg8PN/qmt3T158znn3+uCxcuqF27drd8TWgCa8idc/U1pKitHxJryJ1iDblzrr6G5Pvggw9UtWpVx5XzQ4YM0SeffKJHHnlEn3zyifHO06dPa9y4cUpKStKJEyeuezPD5IhQoCDY0C5GvL29Vb16dUVHR6t3795OcwBdhas30mfNtX19+vS57pJY04rS8+fqfa7899u3b1/16dPH5Z4/AAAAAMVbo0aNNH36dLVs2VKbN29Wz5499ac//UkrV67UuXPnjM0hz9enTx+lpqaqf//+qlKlynXjZkydXAgUFG/jFyNbt25VRESEPvroIzVo0EA9evTQqlWrdOXKFdNpDq7eSJ811/bVr1/fpftc/flz9T5X/vudPXu2Sz5/vyYnJ0ebN282nXFT9Fnn6o30WUOfNfRZY7fbXfqsOvqsycvLc+k+iefQKld//ui7fUePHnVcOb1mzRp17dpV3bt3V1xcnJKTkw3XSV9++aX+/Oc/a9iwYerTp4969+7t9AEUFWxoFyOBgYEaP368vvvuO3388cey2Wzq37+/goODNXr0aO3fv990oss30kcfffSZcuDAAeMzPG+FPutcvZE+a+izhr5by83N1euvv66aNWuqcePGTveIkKTMzExjI90k+qz6tb4TJ04Y7ZN4Dq0q6s8ffbfvgQce0MmTJyVdHWmaPyqoZMmSunDhgsk0SVLFihW5ISSKBTa0iyF3d3d16dJFixcv1q5duzRo0CAlJiaqWbNm6tixo+k8Sa7fSB999NEHAICrmDhxotauXasRI0aob9++mjx5smJiYmS32x1f82s3l6aPPitcvZE++kz/G8kXHh6u//mf/9GQIUN08OBBx03p9+zZoxo1ahiuk0aNGqV33nlHZ8+eNZ0CWMIM7ftAVlaWFi9erAkTJig7O1unTp0ynXQdV2+kzxr6rKHPGlfpu90bZtJ3Y67eJ7l+I33W0GcNfdaEhoZq0qRJjo2R9PR0RUZGqm7dupozZ44yMzMVFBREH333bSN99Jn+N5LvzJkzGjdunA4fPqwXXnhBTz75pCTpnXfekYeHh4YNG2a0r0WLFjp06JCuXLkiPz8/ubu7Ox3/+uuvDZUBBeP+61+Comrjxo1asGCBVq9eLQ8PD0VGRio6Otp0lhNXb6TPGvqsoc8aV+vz8vJSbGys6tevf8Pjhw4d0ujRowu56j/os87VG+mzhj5r6LMmIyNDderUcXzu5+enxMREdenSRQMGDNAf/vAHY20SfVa5ep/k+o30WUPf3VOuXDlNmjTpusdHjBhhoOZ6Xbp0MZ0A3BVsaBcz6enpSkhI0MKFC5Wenq4WLVpo6tSp6tq1qzw9PU3nSXL9Rvroo4++e6F+/fry9vZW165db3h8165dhVzkjD7rXL2RPmvos4Y+a3x8fHTw4EGny9UrV66sFStWKCIiQi+99JLBOvqscvU+yfUb6bOGvnsjIyNDFy9edHrMz8/PUM1VcXFxRr8/cLewoV2MdOvWTZs2bVKlSpUUFRWl6OhoBQQEmM5y4uqN9FlDnzX0WePqfe3bt1d2dvZNj3t7e6tXr16FWOSMPutcvZE+a+izhj5rWrdurU8//VRt2rRxetzHx0crV67U008/bSbs3+izxtX7JNdvpM8a+u6e7Oxsvfnmm1q+fPl1m9mS2fF4QHHChnYx4unpqfnz56tDhw5yc3MznXNDrt5InzX0WUOfNa7e99hjjznucn4jvr6+mjlzZiEWOaPPOldvpM8a+qyhz5omTZqoWrVqNzxWpUoVrV69WuvXry/kqv+gzxpX75Ncv5E+a+i7e0aNGqVvv/1WCQkJio6O1vTp03XkyBF98MEHLjEaxdfXVzab7abH09PTC7EGuHPcFBIAgELg7e2t6tWrKzo6Wr17977pi3JT6LPO1Rvps4Y+a+izhj5r6LPO1Rvps4a+uyckJERz5sxRixYt5Ofnp6SkJAUEBGjp0qVasGCBli9fbrRv4cKFTp9fvnxZqampWrlypYYNG6ZBgwYZKgMKpoTpAAAA7gdbt25VRESEPvroIzVo0EA9evTQqlWrdOXKFdNpkui7G1y9kT5r6LOGPmvos4Y+61y9kT5r6Lt7srOzHXOyy5Ur5xgx0qRJE33zzTcm0yRJvXv3dvr47W9/q8mTJ2vkyJFKTk42nQfcNs7QBgCgEF2+fFl//etflZCQoHXr1qlChQqKiopS3759Vbt2bdN59N0Frt5IH3300UcffXfK1Rvpo8+0li1basKECWrdurW6d++uoKAgvfPOO5oxY4ZmzZql3bt3m068obS0NLVq1UqHDx82nQLcFja0AQAw5OjRo1q4cKESEhKUlpamZs2a6W9/+5vpLAf6rHP1Rvqsoc8a+qyhzxr6rHP1Rvqsoe/OzJgxQ25ubnrppZeUlJSkXr166dKlS7Lb7ZowYYJiYmJMJ97Qe++9p7/85S9KTU01nQLcFja0AQAwKCsrS4sXL9aECROUnZ3tcnc+p886V2+kzxr6rKHPGvqsoc86V2+kzxr6rEtPT9eOHTtUs2ZN1a1b13SOWrRocd1jx48f1+nTpzVlyhT169fPQBVQcO6mAwAAuB9t3LhRCxYs0OrVq+Xh4aHIyEhFR0ebznKgzzpXb6TPGvqsoc8a+qyhzzpXb6TPGvruHj8/P8dMbVfQpUsXp89LlCihihUrqlWrVqpTp46hKqDgOEMbAIBCkp6eroSEBC1cuFDp6elq0aKFfvvb36pr167y9PQ0nUffXeDqjfTRRx999NF3p1y9kT76TJk+ffptf+2QIUPuYQlw/2BDGwCAQtCtWzdt2rRJlSpVUlRUlKKjoxUQEGA6y4E+61y9kT5r6LOGPmvos4Y+61y9kT5r6LOmQYMGt/V1NptNO3fuvMc1tycpKUl79+6VzWZTUFCQWrdubToJKBBGjgAAUAg8PT01f/58dejQQW5ubqZzrkOfda7eSJ819FlDnzX0WUOfda7eSJ819FlTlG6keOTIEfXt21cpKSmqWrWqpKs32GzUqJEWLFjgeAxwdZyhDQAAAAAAANyhv//97xo6dKi++uorPfjgg07HsrOz1apVK73//vtq27atocKroqOjdezYMc2ePVv+/v6SpLS0NMXExKhKlSqKj4832gfcLja0AQAAAAAAgDv03HPP6Te/+Y0GDhx4w+Nz587V2rVrtWTJkkIuc+bn56fExESFhoY6Pb5jxw517dpVhw4dMhMGFFAJ0wEAAAAAAABAUfXdd9+pTZs2Nz0eFhamb7/9tvCCCshms5lOAAqEDW0AAAAAAADgDmVmZqpEiZtvsdlsNp06daoQi24sLCxMcXFxOnz4sOOx9PR0DR8+XGFhYQbLgIJhQxsAAAAAAAC4Q9WqVbvlGdi7d+92iRsuvvvuuzp37pxCQ0NVr1491a9fX40aNdK5c+f07rvvms4DbhsztAEAAAAAAIA79Oabb2rjxo3auHGjvLy8nI6dO3dO4eHhatOmjctsGm/YsEH79u1TXl6egoKCbjkuBXBFbGgDAAAAAAAAd+jEiRMKCwuTzWZTTEyMateuLUnat2+fZs+erby8PCUlJaly5cqGS4HigQ1tAAAAAAAAwIJDhw5p2LBhWrdunfLyrm612Ww2tWvXTpMnT1aNGjUMF161c+dObdq0SZmZmbLb7U7Hxo4da6gKKBg2tAEAAAAAAIC7ICsrSwcOHFBeXp5q1qyp8uXLm05yeP/99zVmzBj5+fmpcuXKstlsjmM2m01ffPGFwTrg9rGhDQAAAAAAABRzgYGBiouL0/PPP286BbCkhOkAAAAAAAAAAPeW3W7XE088YToDsIwNbQAAAAAAAKCYGzBggBISEkxnAJYxcgQAAAAAAAAo5vLy8vTcc8/p2LFjCgkJUcmSJZ2Oz5gxw1AZUDDupgMAAAAAAAAA3Fvjxo3T+vXr1bBhQ2VnZ5vOAe4YZ2gDAAAAAAAAxVz16tU1depUPfPMM6ZTAEuYoQ0AAAAAAAAUc15eXmrQoIHpDMAyNrQBAAAAAACAYu7ll1/WrFmzlJfHsAYUbYwcAQAAAAAAAIq5nj17asuWLSpXrpyCgoLk7u58a71FixYZKgMKhptCAgAAAAAAAMXcQw89pM6dO5vOACzjDG0AAAAAAAAAQJHADG0AAAAAAADgPnT+/HklJCToqaeeMp0C3DZGjgAAAAAAAAD3ke3btys+Pl7Lli2TzWZTx44dTScBt40NbQAAAAAAAKCYy8rK0qJFizR//nwdPHhQubm5mjp1qqKiolSyZEnTecBtY+QIAAAAAAAAUEwlJSVpwIABCg4O1qpVqxQbG6vvv/9eJUqUUNOmTdnMRpHDGdoAAAAAAABAMfXMM89o8ODBSk5Olq+vr+kcwDI2tAEAAAAAAIBiqn379po7d65++ukn9ezZUx06dJCbm5vpLOCOMXIEAAAAAAAAKKYWLVqk7du3KzQ0VKNGjVKdOnX0xhtvSJJsNpvhOqDgbFlZWXmmIwAAAAAAAADce19++aUWLFigxMREVaxYUV27dlW3bt302GOPmU4Dbgsb2gAAAAAAAMB9Jjs7W0uWLNGCBQu0a9cunTp1ynQScFvY0AYAAAAAAADuYzt37lTDhg1NZwC3hZtCAgAAAAAAAPeBixcv6rvvvlNmZqbsdrvpHOCOsKENAAAAAAAAFHMbNmzQoEGDdOLEieuO2Ww2Ro6gyGDkCAAAAAAAAFDMNW7cWC1atND//u//qnLlyrLZbE7HPTw8DJUBBcOGNgAAAAAAAFDM+fr66quvvpK/v7/pFMCSEqYDAAAAAAAAANxbHTp00LZt20xnAJZxhjYAAAAAAABQzGVnZysmJkYBAQEKDg5WyZIlnY5HRUUZKgMKhg1tAAAAAAAAoJj7/PPPFRsbqwsXLqh06dJOM7RtNpvS09MN1gG3jw1tAAAAAAAAoJirV6+eunfvrri4OJUpU8Z0DnDHmKENAAAAAAAAFHPZ2dkaMGAAm9ko8tjQBgAAAAAAAIq5iIgIbdy40XQGYJm76QAAAAAAAAAA95a/v7/GjRunr7/+WnXr1pW7u/O24JAhQwyVAQXDDG0AAAAAAACgmGvQoMFNj9lsNu3cubMQa4A7x4Y2AAAAAAAAUIzZ7Xbt3btXfn5+Klu2rOkcwBJmaAMAAAAAAADFmM1mU1hYmI4fP246BbCMDW0AAAAAAACgGLPZbKpdu7YyMzNNpwCWsaENAAAAAAAAFHNvv/22fv/73ys1NVV5eUwgRtHFDG0AAAAAAACgmPP19VVubq7sdrvc3d3l4eHhdDw9Pd1QGVAw7qYDAAAAAAAAANxbEydONJ0A3BWcoQ0AAAAAAAAAKBI4QxsAAAAAAAC4D1y4cEFLlizR3r17ZbPZFBQUpMjIyOvGjwCujDO0AQAAAAAAgGLu+++/V2RkpM6cOaO6detKknbv3q1y5crps88+U2BgoOFC4PawoQ0AAAAAAAAUc926dZOXl5c+/PBDlStXTpJ05swZxcTE6OLFi1q2bJnhQuD2sKENAAAAAAAAFHNVq1bV+vXrFRwc7PT47t271b59ex05csRQGVAwJUwHAAAAAAAAALi3PDw8lJ2dfd3jZ86cYYY2ihQ2tAEAAAAAAIBi7qmnntKrr76qrVu36sqVK7py5Yq2bNmi119/XR07djSdB9w2Ro4AAAAAAAAAxVxWVpZiY2O1Zs0aubm5SZKuXLmiTp06acaMGSpfvrzZQOA2saENAAAAAAAA3CcOHDigvXv3Ki8vT0FBQQoICDCdBBSIu+kAAAAAAAAAAPfesmXLlJSUpBMnTshutzsdW7RokaEqoGDY0AYAAAAAAACKuVGjRmnWrFlq3bq1qlSpIpvNZjoJuCOMHAEAAAAAAACKudq1a2vy5Mnq2rWr6RTAkhKmAwAAAAAAAADcW3a7XfXr1zedAVjGhjYAAAAAAABQzPXv31+LFy82nQFYxgxtAAAAAAAAoJjLzs7Wp59+qo0bN6pu3bpyd3feFpw4caKhMqBg2NAGAAAAAAAAirnvv//eMXJk3759Tse4QSSKEm4KCQAAAAAAAAAoEpihDQAAAAAAAAAoEtjQBgAAAAAAAAAUCWxoAwAAAAAAAACKBDa0AQAAAAAAAABFAhvaAAAAAAAAAIAi4f8BPcNJK1XOi3kAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 1728x1440 with 4 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Make sure we use the subsample in our correlation\n",
    "\n",
    "f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))\n",
    "\n",
    "# Entire DataFrame\n",
    "corr = normal_sample.corr()\n",
    "sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)\n",
    "ax1.set_title(\"Imbalanced Correlation Matrix \\n (don't use for reference)\", fontsize=14)\n",
    "\n",
    "sub_sample_corr = under_sample_data.corr()\n",
    "sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)\n",
    "ax2.set_title('SubSample Correlation Matrix \\n (use for reference)', fontsize=14)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "2a972f3e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x288 with 4 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "f, axes = plt.subplots(ncols=4, figsize=(20,4))\n",
    "\n",
    "# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)\n",
    "sns.boxplot(x=\"Class\", y=\"V17\", data=under_sample_data, palette='Blues', ax=axes[0])\n",
    "axes[0].set_title('V17 vs Class Negative Correlation')\n",
    "\n",
    "sns.boxplot(x=\"Class\", y=\"V14\", data=under_sample_data, palette='Blues', ax=axes[1])\n",
    "axes[1].set_title('V14 vs Class Negative Correlation')\n",
    "\n",
    "\n",
    "sns.boxplot(x=\"Class\", y=\"V12\", data=under_sample_data, palette='Blues', ax=axes[2])\n",
    "axes[2].set_title('V12 vs Class Negative Correlation')\n",
    "\n",
    "\n",
    "sns.boxplot(x=\"Class\", y=\"V10\", data=under_sample_data, palette='Blues', ax=axes[3])\n",
    "axes[3].set_title('V10 vs Class Negative Correlation')\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c3d0bd6",
   "metadata": {},
   "source": [
    "### Train Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "039ed480",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of transactions in the training dataset:  199364\n",
      "Number of transactions in the testing dataset:  85443\n",
      "Total number of transactions:  284807\n",
      "\n",
      "Number of transactions in the training dataset:  688\n",
      "Number of transactions in the testing dataset:  296\n",
      "Total number of transactions:  984\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Split the whole dataset into training and testing\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)\n",
    "\n",
    "# Print the size of each dataset\n",
    "print(\"Number of transactions in the training dataset: \", len(X_train))\n",
    "print(\"Number of transactions in the testing dataset: \", len(X_test))\n",
    "print(\"Total number of transactions: \", len(X_train) + len(X_test))\n",
    "\n",
    "# Split the undersampled dataset into training and testing\n",
    "X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)\n",
    "\n",
    "# Print the size of each dataset\n",
    "print(\"\\nNumber of transactions in the training dataset: \", len(X_train_undersample))\n",
    "print(\"Number of transactions in the testing dataset: \", len(X_test_undersample))\n",
    "print(\"Total number of transactions: \", len(X_train_undersample) + len(X_test_undersample))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8b493d4e",
   "metadata": {},
   "source": [
    "### Logistic Regression\n",
    "\n",
    "We are highly focused on recall score as it helps us identify the maximum number of fraudulent transactions. Recall, along with accuracy and precision, is a key metric for evaluating a confusion matrix. Recall calculates the number of true positive instances divided by the sum of true positive instances and false negative instances.\n",
    "\n",
    "Since our dataset is imbalanced, many normal transactions may be predicted as fraudulent, leading to false negatives. Recall helps us address this issue. Increasing recall may lead to a decrease in precision, but that is acceptable in our scenario, as predicting a fraudulent transaction as normal is not as critical as the opposite.\n",
    "\n",
    "We can also apply a cost function to assign different weights to false negatives and false positives, but we'll leave that for another time."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "734ddcae",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import KFold, cross_val_score\n",
    "from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "415790d3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C parameter:  0.01\n",
      "Recall:  0.821917808219178\n",
      "Recall:  0.8493150684931506\n",
      "Recall:  0.9152542372881356\n",
      "Recall:  0.9324324324324325\n",
      "Recall:  0.8787878787878788\n",
      "C parameter:  0.1\n",
      "Recall:  0.8356164383561644\n",
      "Recall:  0.863013698630137\n",
      "Recall:  0.9491525423728814\n",
      "Recall:  0.9324324324324325\n",
      "Recall:  0.8939393939393939\n",
      "C parameter:  1\n",
      "Recall:  0.8356164383561644\n",
      "Recall:  0.863013698630137\n",
      "Recall:  0.9661016949152542\n",
      "Recall:  0.9324324324324325\n",
      "Recall:  0.8939393939393939\n",
      "C parameter:  10\n",
      "Recall:  0.8493150684931506\n",
      "Recall:  0.863013698630137\n",
      "Recall:  0.9491525423728814\n",
      "Recall:  0.9324324324324325\n",
      "Recall:  0.8939393939393939\n",
      "C parameter:  100\n",
      "Recall:  0.8493150684931506\n",
      "Recall:  0.863013698630137\n",
      "Recall:  0.9661016949152542\n",
      "Recall:  0.9459459459459459\n",
      "Recall:  0.8787878787878788\n",
      "Best model to choose from cross validation is with C parameter = 100\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "100"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import KFold\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.metrics import recall_score\n",
    "\n",
    "def print_kfold_scores(x_train_data, y_train_data):\n",
    "    k_fold = KFold(n_splits=5, shuffle=False)\n",
    "    c_param_range = [0.01, 0.1, 1, 10, 100]\n",
    "    results_table = pd.DataFrame(index=range(len(c_param_range)), columns=['C_parameter', 'Mean recall score'])\n",
    "    j = 0\n",
    "    for c_param in c_param_range:\n",
    "        print('C parameter: ', c_param)\n",
    "        recall_accs = []\n",
    "        for train, test in k_fold.split(x_train_data):\n",
    "            lr = LogisticRegression(C=c_param, penalty='l2')\n",
    "            lr.fit(x_train_data.iloc[train], y_train_data.iloc[train])\n",
    "            y_pred = lr.predict(x_train_data.iloc[test])\n",
    "            recall = recall_score(y_train_data.iloc[test], y_pred)\n",
    "            recall_accs.append(recall)\n",
    "            print('Recall: ', recall)\n",
    "        results_table.loc[j, 'C_parameter'] = c_param\n",
    "        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)\n",
    "        j += 1\n",
    "    best_c = results_table.loc[results_table['Mean recall score'].astype(float).idxmax()]['C_parameter']\n",
    "    print('Best model to choose from cross validation is with C parameter =', best_c)\n",
    "    return best_c\n",
    "\n",
    "best_c = print_kfold_scores(X_train_undersample, y_train_undersample)\n",
    "best_c"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5b45d6dd",
   "metadata": {},
   "source": [
    "#### Make a function to create confusion Matrix's"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "4250c01a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import itertools\n",
    "\n",
    "def plot_confusion_matrix(cm, classes,\n",
    "                          normalize=False,\n",
    "                          title='Confusion matrix',\n",
    "                          cmap=plt.cm.Blues):\n",
    "    \"\"\"\n",
    "    This function prints and plots the confusion matrix.\n",
    "    Normalization can be applied by setting `normalize=True`.\n",
    "    \"\"\"\n",
    "    plt.imshow(cm, interpolation='nearest', cmap=cmap)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick_marks = np.arange(len(classes))\n",
    "    plt.xticks(tick_marks, classes, rotation=0)\n",
    "    plt.yticks(tick_marks, classes)\n",
    "\n",
    "    if normalize:\n",
    "        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]\n",
    "        #print(\"Normalized confusion matrix\")\n",
    "    else:\n",
    "        1#print('Confusion matrix, without normalization')\n",
    "\n",
    "    #print(cm)\n",
    "\n",
    "    thresh = cm.max() / 2.\n",
    "    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):\n",
    "        plt.text(j, i, cm[i, j],\n",
    "                 horizontalalignment=\"center\",\n",
    "                 color=\"white\" if cm[i, j] > thresh else \"black\")\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.ylabel('True label')\n",
    "    plt.xlabel('Predicted label')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "878444e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recall metric in the testing dataset:  0.9251700680272109\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAUkAAAEmCAYAAADvKGInAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAA0mklEQVR4nO3de1zP9///8du7pIX0TjpJBxTK0JKxwhxaGMpZZg6ZOW5jB6OPdjJTMcam4ae1k+bMCHOYhTTCJoxhFnI+NEU5lXe/P8z7673q3bu8e79753F1eV0u3q/X8/16PV65uPd8vZ6vgyIrK6sAIYQQRTIzdgFCCFGRSUgKIYQWEpJCCKGFhKQQQmghISmEEFpISAohhBYSkibkwIED9OrVC09PT5RKJU2bNi33bSYnJ6NUKomKiir3bT1JunXrhlKpNHYZQgdVjF1ARXTy5Eni4uJITk7m7Nmz3L59G6VSydNPP03Xrl0JCwujZs2aBq3pxo0b9O/fnxs3bjBgwACcnZ2xsbExaA3igTNnztC8eXMCAwPZsGGDscsR5UwhF5NrmjVrFp988gkqlYoWLVrg5+dHzZo1uXbtGrt37+bEiRPUqlWL9PR0g9a1fft2evbsSXh4OJ999pnBtnvr1i3OnTuHnZ0ddnZ2BttuRaaPkHz4y7dhw4Z6rk7om/QkHzFnzhw+/vhjXFxciI+Pp1WrVoXapKSk8O677xq8tosXLwLg4OBg0O1Wq1ZN/iOXA1dXV2OXIHQk5yT/lZGRwSeffIKFhQVLly4tMiABAgMD+eWXXwrN37lzJ/369aNevXo4ODjQvHlzJk2axNWrVwu1HTNmDEqlkuTkZNauXUvHjh1xdnbGw8OD8PBwzp8/r2575swZlEolY8aMASAmJgalUolSqSQhIQHQfn6ruHOK6enpvPHGGzzzzDM4OTnh7u7Os88+y7hx4zh79myJ33+4jrFjx+Lj44O9vT1eXl4MGzaMw4cPF2qbkJCgXs+hQ4fo378/bm5uODs707VrV/bs2VNk/UV5+DPp1q0bV65cYdy4cXh5eVGnTh2Cg4NJSUkBICcnh//97388/fTTODg40KpVK3788cdC68vOzmbu3Ll0794db29v7O3tadCgAWFhYaSmphbaj+bNmwMPfmE+/Ld49Gf0aH0XLlxg9OjRNGzYkFq1arF+/Xqg8L9ZXl4eQUFBKJVK1q5dW6jGiRMnolQqmTx5ss4/J6Ef0pP8V0JCAnl5efTq1avEARFLS0uNz19//TVvvfUWVlZWhIaG4uTkRGpqKgsXLmTDhg389NNPRfYcvvrqK3766SdefPFFAgMD2b9/P2vWrOHw4cOkpKRgaWmJjY0NkyZN4vDhw2zcuJHAwEDatGkDUOaBm4sXL9KxY0dycnLo1KkTPXr04N69e5w7d47ExET69etXYk/nwIEDhIaGcuPGDTp37kyTJk04deoUiYmJ/PTTTyxevJgXXnih0PfS0tL4/PPPadWqFUOGDOHcuXOsW7eO0NBQdu7cSaNGjXTej+zsbDp37oytrS39+vXjwoULrF27lj59+rBlyxYmTJjArVu3ePHFF7l58yarVq0iPDwcFxcXWrZsqV7PiRMnmDZtGgEBAXTu3BmlUsnZs2fZuHEjW7duZcmSJQQHBwMPfuajR49mwYIFuLq68tJLL6nX8/Df5aHr168THBxMzZo16dmzJ/n5+dja2ha5LxYWFsTHx9O2bVtef/11mjdvjoeHBwDr1q1j0aJFPPPMM0ydOlXnn4/QDwnJf+3evRuADh06lOp7GRkZTJo0iWrVqvHzzz/j7e2tXjZt2jQ+/fRT3n77bZYvX17ou7/88gs7duygcePG6nkjRoxg5cqVbNiwgd69e6NUKomIiCAhIYGNGzfSpk0bIiIiyriXD6xdu5asrCw++eQTxo0bp7Hs7t275OXlaf1+QUEBo0eP5saNG3z55ZcaQbF9+3Z69erF6NGjOXz4MNWqVdP47ubNm1m4cCEDBgxQz/v666958803WbhwIbNnz9Z5P/744w9GjRpFdHQ0CoUCgNmzZzN16lS6d+9Ohw4diIuLw8LCAoCOHTvy6quvMmfOHHUvHKBhw4YcO3as0DnXjIwMgoKCmDJlijokmzVrho2NDQsWLMDNzU3rv8XRo0cZMGAAsbGxVKlS8n81Nzc35s2bx+DBgxk+fDibNm3iwoULvP7669SsWZP4+HiqVq2q889H6Iccbv/r8uXLANSpU6dU31u+fDn37t3jlVde0QhIeHCI5OzszJYtW7hw4UKh744aNUojIAGGDh0KwO+//16qOkrDzOzBP/t/Awwe9JJr1Kih9fupqakcP34cPz8/jYAEaN++Pd27dyczM7PIQY3nnntOIyABXn75ZapUqVLqfa5evTrvv/++OiAB+vfvDzy4GmDatGnqgATo3bs3FhYWhU4H2NjYFDko5ebmRmhoKH/99ZfGKQhdVa1alWnTpukUkA/16NGDkSNH8vvvvzNlyhReeeUV9emAevXqlboG8fgkJP9VUPBgkP/R/3C6OHjwIADt2rUrtMzS0pLWrVsDcOjQoULLfX19C81zcXEBICsrq1R1lEbXrl2xtrZm4sSJDBo0iPj4eA4fPoxKpdLp+9r2GR4E5aPtHlXUPltYWODg4FDqfW7QoAHVq1fXmOfk5ASAUqksdMrA3Nwce3v7In9h7dmzh2HDhtGkSRMcHBzU5xkXLVoE/N/AWWm4ublhb29f6u9NmzYNX19fFi1axG+//cbw4cPp1atXqdcj9EMOt//l5OTEiRMnNAZNdHHjxg2g+FFnR0dHjXaPKupaS3NzcwDu379fqjpKw9XVlV9++YWYmBi2bt2q7vE5ODgwcuRI3nzzTXUdRdH3PsOD/S7tPltbWxea97DXVtSyh9vJz8/XmJeYmMjQoUN56qmn6NChAx4eHlSrVg0zMzN27dpFSkoKd+/eLVVtUPYrEapWrUqXLl1IS0sDUA/aCeOQkPzXc889x86dO9mxYwdDhgzR+XsP/9NfuXKlyOUPD+PL8+Lzh4fP+fn5hQ7tsrOzi/yOl5cXcXFx3L9/nyNHjrBz507i4uKYNm0aKpVK62VOFWGf9Wn69OlUrVqVpKSkQgNHEyZMUI+Wl1Zpj0oe2r9/P7NmzcLOzo7MzEzGjh3LTz/9VKrDdqE/crj9r0GDBmFhYcG6des4evSo1raP9ioeXg6SnJxcZLuHl5A8bFceHl5Kcu7cuULLDhw4oPW75ubmNGvWjNdee42VK1cCqC9TKY62fQbYsWMHUPShdUWUnp5Oo0aNCgWkSqUq8tKkh71sXU9PlEZWVhbDhw8HYOXKlQwfPpx9+/bJqLYRSUj+y83NjSlTppCXl0f//v3Zt29fke327NlDUFCQ+nP//v2pWrUqX331FSdOnNBoO3v2bC5cuEBwcDDOzs7lVvvDy1ni4+M15h86dIgFCxYUav/bb7+pe3uPejjvqaee0rq9Vq1a0ahRI3777TeWLVumsWzHjh0kJiZiZ2fHiy++WKr9MBY3NzfS09M1zlUWFBQQHR3NsWPHCrW3tbVFoVCUaTCnJK+99hoZGRl8+OGHPPPMM0yfPp2nn36aL774gq1bt+p9e6Jk0n9/xIQJE8jPz2f69Om88MIL+Pv74+fnh7W1NZmZmezdu5ejR49qjIS6ubkRExPDW2+9RYcOHejZsyeOjo6kpqaSkpKCi4sLs2bNKte6Bw0axLx58/j88885cuQITZo04fTp0/z000+EhISwatUqjfYrVqwgLi6O5557jgYNGlCrVi31dYFmZma88cYbWrenUCiYP38+PXv2ZPTo0axZs0Z9neS6deuoWrUqCxYsKHL0vCIaO3Ysb775Js8//zwhISFUqVJFPYLfpUsXNm3apNG+evXqtG7dmt27dzNgwAB8fX2pUqUKAQEBBAYGlrmOBQsWsH79ejp37qy+NOupp57im2++oX379owePZrk5ORSX4EhHo/0JP/jnXfeYe/evYwePZpbt26xdOlS5s6dy/r166lduzYxMTHqE+oPhYeH8+OPP/Lcc8+xYcMGvvjiC86ePcvIkSNJSkoq91vQ7Ozs2LBhA127dmXfvn0sWrSICxcu8PXXXzNs2LBC7fv27cvQoUO5fv06a9euJTY2ltTUVDp37szWrVvp3r17idv08/Nj+/bthIWFcfDgQT7//HN27txJt27d2Lp1a5EXkldU4eHhxMbG4ujoyJIlS1ixYgUuLi78/PPPxZ4mWbBgAd27d2ffvn3MnDmTTz75hJ07d5a5hrS0NN5//31cXFyYP3++xjJPT09mz55NZmYmI0aMKNdBPVGYPOBCCCG0kJ6kEEJoISEphBBaSEgKIYQWEpJCCKGFhKQQQmghISmEEFpISAohhBYmd8eNU/spxi5Br5IXDaPtq98Yu4xyc33fPGOXUC7+SNvP077+xi5D7+7mFv1AFH1w6vi+1uWXfqmY96ebXEgKIUyUWfGP36vIJCSFEIahMM2zexKSQgjDKOPzNY1NQlIIYRjSkxRCCC2kJymEEFpIT1IIIbSQ0W0hhNBCDreFEEILOdwWQggtJCSFEEILMzncFkKI4klPUgghtJDRbSGE0EJGt4UQQgs53BZCCC2kJymEEFpIT1IIIbSQnqQQQmgho9tCCKGFHG4LIYQWEpJCCKGFnJMUQggtpCcphBBaSE9SCCG0kNFtIYQonsJEe5KmeZJACGFyFAqF1qk4KSkphIWF4e3tjVKpJCEhQb0sLy+PDz74gICAAOrUqUOjRo0YMWIEZ8+e1VhHt27dUCqVGtPw4cN1qltCUghhGIoSpmLk5ubi4+NDdHQ0VlZWGstu3brFwYMHeeedd9ixYwc//PAD58+fp2/fvuTn52u0HTRoEMePH1dPn332mU5ly+G2EMIgynq4HRwcTHBwMABjx47VWGZjY8OPP/6oMe+zzz6jdevWHD9+nCZNmqjnV6tWDUdHx1JvX3qSQgiDKOvhdmndvHkTAKVSqTF/1apV1K9fn9atWxMZGaluVxLpSQohDMLMrPz7ZPfu3SMyMpIuXbrg4uKint+vXz9cXV1xcnLi2LFjfPTRR/zxxx+FeqFFkZAUQhhGOQ9u5+fnM3LkSLKzs1myZInGsmHDhqn/3qRJEzw8POjUqRNpaWn4+vpqXa8cbgshDKI8D7fz8/N55ZVXOHLkCGvXrqVWrVpa2z/zzDOYm5uTnp5e4rqlJymEMIjyuk4yLy+P4cOH8+eff7J+/XqdBmeOHDnC/fv3dWorISmEMIiyhmROTo66x6dSqTh37hyHDh3C1tYWZ2dnhg4dyoEDB1iyZAkKhYLLly8DULNmTaysrDh16hTLly8nODiYWrVqcfz4cSIjI2nWrBmtW7cucfsSkkIIgyhrSB44cIAePXqoP0dFRREVFcXAgQOZPHkyGzduBKB9+/Ya34uNjWXQoEFYWFiwY8cOFixYQG5uLi4uLgQHBzN58mTMzUu+VVJCUghhEAqzsoVk27ZtycrKKna5tmUAdevWVQdpWUhICiEMwlTv3ZaQFEIYhISkEEJoY5oZKSEphDAM6UkKIYQWphqScsdNOQj0a8CKOaP4e/M0bh+Yx8s9WhXb1t3dndsH5jFhcCf1PDfnWtw+MK/I6c0hnYpdlyhfM2OiCGzdEodaNXnhhRfo07MHR/74w9hlmQwzMzOtU0UlPclyUKOaJUdPXuCH9anETR1SbLteQb5Ur16dC1eyNOafu3wdj6AIjXkhHZszZ3J/Vv+cVg4VC13s3LGdUaPH0sK/JX8d+4Mfli6jW5cgfj90tMTb4ARyTlL8n827jrJ511EA/t9Hg4ts4+Zsy6cT+5Keno51LWeNZSpVAZczNR/jFNrRl19Sj3PmQmb5FC1KlLhxs/rvBfl3iP/mexztbNj9awrduvfQ8k0BcrgtSsHc3Ixvo8KJjtvEnTt3SmzvXseODs82JH51igGqE7q6efMmKpUKpdLW2KWYBEM9T1LfJCSN4L3R3cjMymXRil06tR/eO4BrWbkkbj9UzpWJ0njnrfE0b+5L6+eeM3YpJsFUQ1IOtw2sTQtPBoe0olVYtE7tzc3NGBzSmsXr9pCfryrn6oSuPvvsM35N2cUv23fpdP+vwGTPSRq9JxkXF0ezZs1wdHTk+eef59dffzV2SeXqef+GONWuyaktn3Bz31xatGiBex07po0P5eSmjwu179buaZztbfh6zW4jVCuKMvHtN9m8eTObtvxCvfr1jV2OyZDR7TJYvXo1kydPZtasWbRu3Zq4uDj69evHnj17cHV1NWZp5eb/Ld/Jmp8PqD9/+2EotRzqsnzTb0WecwzvHcjO/X9xMuOKIcsUxXj7zfGsXL6U+fPn06hxY2OXY1Iq8iG1NkYNydjYWF566SWGDh0KwMyZM9m2bRvx8fF88MEHxiztsVS3qkoDV3sAzBQKXJ1tadbQhes3bnH20nWuXs9Rt71z5w55+fe5fO0Gf53RDEJXJ1teeM6bEe99Z9D6RdEmvD6OHxK+Z/mqH+H+PS5dugRAjRo1qFGjhnGLMwGmGpJG6+Peu3ePtLQ0OnbsqDG/Y8eOpKamGqkq/fDzcSd1WQSpyyKoZlWV98d0J3VZBO+N6Vaq9Qzt+RzZObdZsy2tfAoVpbJwwZfcvHmTrsGd6Nq1K/Vcnann6syc2Z8auzTTUMb3bhubIisrq8AYG7548SLe3t5s2LCBwMBA9fyYmBhWrFjB/v37i/zeX3/9ZagShXjieHl5ldu6n/lI+yVsBz4I1LrcWIw+uv3fLnhBQYHWbnnbV78p54oMK3nRsEq3T4+6vm+esUsoF3+k7edpX39jl6F3d3Ozy23dpnq4bbSQtLOzw9zcnCtXNM/DXbt2DXt7eyNVJYQoL2ZlfDK5sRntnGTVqlXx9fUlKSlJY35SUhKtWhX/QAghhGlSKLRPFZVRD7fHjRvHqFGjaNGiBa1atSI+Pp5Lly4RHh5uzLKEEOVADrfLoHfv3vzzzz/MnDmTy5cv4+3tzfLly3FzczNmWUKIcmCiGWn8gZsRI0YwYsQIY5chhChnpnpO0ughKYR4MkhPUgghtJCepBBCaCEDN0IIoYWEpBBCaGGiGWn850kKIZ4MZX0yeUpKCmFhYXh7e6NUKklISNBYXlBQQFRUFI0bN8bJyYlu3brx559/arS5e/cuEydOpH79+tSpU4ewsDDOnz+vU90SkkIIgyjrHTe5ubn4+PgQHR2NlZVVoeVz584lNjaWmJgYfvnlF+zt7enVqxc3b/7fy/QiIiJITEzkq6++YuPGjdy8eZMBAwZw//79EuuWkBRCGISZmULrVJzg4GDef/99QkNDCz3BvKCggPnz5zNhwgRCQ0Px8fFh/vz55OTksHLlSgCys7P5/vvvmTp1Kh06dMDX15eFCxdy5MgRtm/fXnLdj7XXQgiho/J4EdiZM2e4fPmyxnNpraysCAgIUD+XNi0tjby8PI02devWpVGjRjo9u7bYgZslS5aUqeiBAweW6XtCiMqtPAZuLl++DFDoyWH29vZcvHgRgCtXrmBubo6dnV2hNv99CllRig3JsWPHlrpghUIhISmEKFJ5XgJU2ufS6toGtITkwYMHdSxPCCFKVh4Z6ejoCDzoLdatW1c9/9Hn0jo4OHD//n0yMzOpXbu2RpuAgIASt1FsSMqTeIQQ+lQePUl3d3ccHR1JSkrCz88PePByvd27dzN16lQAfH19sbCwICkpiX79+gFw/vx5jh8/rtOza0t9Mfnt27c5cOAAV69eJTAwUCOZhRCiOGW9dzsnJ4f09HQAVCoV586d49ChQ9ja2uLq6sqYMWOYNWsWXl5eeHp68umnn1K9enX69u0LgI2NDYMHD+b999/H3t4eW1tbpkyZQpMmTWjfvn3JdZem2AULFtCoUSO6d+9OeHg4R44cASAzMxM3Nze++05efSqEKFpZr5M8cOAA7dq1o127dty+fZuoqCjatWvH9OnTARg/fjxjx45l4sSJdOjQgUuXLrF69Wqsra3V65g+fbo6t7p06UL16tVZunQp5ubmJdatc08yISGBiIgIevXqRadOnXjttdfUy+zs7OjQoQNr1qxhyJAhuq5SCPEEKevhdtu2bcnKytK63oiICCIiIopt89RTTzFz5kxmzpxZ6u3r3JOMjY2lc+fOxMfH07Vr10LLfX19OX78eKkLEEI8GcrjOklD0Dkk//77bzp37lzscjs7OzIzM/VSlBCi8qn0LwKztrYmO7v4d/L+/fffMogjhChWRe4taqNzT7Jdu3YkJCRw9+7dQsvOnz/Pt99+S1BQkF6LE0JUHmW9d9vYdO5JRkZG0qlTJ9q3b0/Pnj1RKBRs3bqVpKQkvvnmGywsLHj33XfLs1YhhAkz0Y6k7j3J+vXrs2nTJpycnIiJiaGgoIDY2Fjmzp1L8+bN2bRpEy4uLuVZqxDChJkpFFqniqpUF5M3atSINWvWkJWVRXp6OiqVCg8PDzkXKYQoUQXOQa3K9PoGpVKpvgVICCF0YaoDN6UKyaysLObNm8eWLVs4e/YsAK6urgQHBzNu3DhsbW3LpUghhOmrwGMzWul8TvLkyZMEBAQwa9Ys8vPzadOmDYGBgeTn5zNr1iwCAgL466+/yrNWIYQJq/Sj2xMnTiQnJ4e1a9fSrl07jWU7duxg8ODBTJo0idWrV+u9SCGE6VNQcYNQG517kqmpqYwePbpQQAI8//zzjBo1ij179ui1OCFE5WGm0D5VVDr3JG1sbFAqlcUuVyqVWpcLIZ5spjpwo3NPcvDgwSxevFjjNY0PZWdns3jxYgYPHqzX4oQQlUelu3d7zZo1Gp8bNmyIQqHA39+fgQMHUr9+feDBPdtLly7F3t4eLy+v8q1WCGGyKvIF49oUG5LDhw9HoVBQUFAAoPH3uXPnFmp/5coVRo4cqX4asBBCPKoij2BrU2xIJiYmGrIOIUQlZ6IdyeJDsk2bNoasQwhRyVW6w20hhNAn04zIUobk1atX+f7770lLSyM7OxuVSqWxXKFQsG7dOr0WKISoHEz1EiCdQ/LYsWN069aN3NxcGjRowJ9//knjxo3Jysri4sWL1KtXTx6VJoQolomO2+h+neSHH36IhYUFe/bsYd26dRQUFBAVFcXRo0dZtGgRWVlZfPzxx+VZqxDChJnqvds6h+Tu3bsJDw/Hw8MDM7MHX3t4SVDfvn3p3bs37733XvlUKYQweZX+bYl5eXk4OzsDD95hC2i8GKxp06YcOHBAz+UJISoLU713W+eQrFu3LhkZGQBYWVnh5OTE3r171cuPHj1K9erV9V+hEKJSMNWepM4DN23btmXjxo1ERkYC0K9fP7788ktu3LiBSqVi2bJlcu+2EKJYFTcGtdM5JCdMmEC7du24c+cOTz31FFOmTOHGjRusWbMGc3NzBgwYIAM3QohiVfqLyV1dXXF1dVV/trS0ZM6cOcyZM6c86hJCVDJlHcFu2rSp+nUxjwoODmb58uWMGTOGJUuWaCzz9/fn559/LtP2/kvuuBFCGERZO5JJSUncv39f/fnSpUu0b9+enj17que1b9+ehQsXqj9XrVq1rGUWUmxI/jeZdTVw4MAyFyOEqLzKerj931dWf//991hbW2uEpKWlJY6Ojo9TXrGKDcmxY8eWemUKhUJCUghRJH2ckiwoKOD7779nwIABVKtWTT1/9+7deHp6YmNjQ2BgIO+99x729vaPv0G0hOTBgwf1sgF9O5s8x9gl6NXpY2mVbp8eZdslxtgllIvk6E6Vct8urRpdbuvWx2U+SUlJnDlzRuNKmqCgIHr06IG7uzsZGRlMmzaNkJAQtm/fjqWl5WNvs9iQdHNze+yVCyHEQzpflK3Ft99+i5+fH82aNVPP69Onj/rvTZo0wdfXl6ZNm7J582ZCQkIee5v6qFsIIUpkbqbQOpXk6tWrbNy4kaFDh2pt5+zsTJ06dUhPT9dL3TK6LYQwiMe99TAhIQFLS0t69+6ttV1mZiYXL17U20CO9CSFEAbxOLclFhQU8N1339G7d2+sra3V83NycoiMjGTv3r2cOXOG5ORkwsLCsLe3p3v37nqpW3qSQgiDeJyeZHJyMunp6SxatEhjvrm5OUePHmXp0qVkZ2fj6OhI27Zt+frrrzXC9HFISAohDOJxBrfbtWtHVlZWoflWVlasXr267CvWgYSkEMIgTPXe7VKdk8zIyOCNN97A19cXV1dXdu3aBTw4Ufr222+TlpZWHjUKISoBc4X2qaLSuSd5/PhxunTpgkqlwt/fn4yMDPX9lHZ2duzbt4+7d+8yb968citWCGG6TLUnqXNIfvDBB1hbW/Pzzz9jbm6Op6enxvLg4GB+/PFHfdcnhKgkTDQjdT/c/vXXXxkxYgQODg5FDte7urpy8eJFvRYnhKg8TPX1DTr3JPPz87W+nuH69euYm5vrpSghROVjqofbOvckfXx8SE5OLnJZQUEBiYmJ+Pr66qsuIUQlo1BonyoqnUNyzJgxrF27lhkzZvDPP/8AoFKpOHHiBMOHD+fAgQO8/vrr5VaoEMK0mSsUWqeKSufD7T59+nD27Fk++eQToqOj1fPgwVXv06ZN44UXXiifKoUQJq8in3fUplQXk0+YMIG+ffuybt060tPTUalU1KtXj5CQENzd3curRiFEJfBEhCQ8eP92WZ5aLoR4slXkd2trI7clCiEMotL3JG1tbXX6TfBwUEcIIR5loh1J3UPy3XffLRSS9+/f58yZM/z00094enrSuXNnvRcohKgcqphoV1LnkIyIiCh22YULFwgKCqJhw4Z6KUoIUfmYak9SL08mr1OnDuHh4cyYMUMfqxNCVEJmKLROFZXeBm6USiWnTp3S1+qEEJWMqfYk9RKS165d49tvv5XX0AohimWipyR1D8kePXoUOT87O5sTJ06Ql5dHfHy83goTQlQupvqAC51DUqVSFRrdVigUuLu706FDB4YMGUKDBg30XqAQonLQ5d3aFZHOIblhw4byrEMIUcmZaEdSt9Ht27dv06NHDxYvXlze9QghKimzEqaKSqfarKysOHjwoPqdNkIIUVoKhULrVFHpHOBt2rTh119/Lc9ahBCVmKKEqaLSOSRjYmL4/fffee+99zh9+jQqlao86xJCVDJmCoXWqaLSOnCzZMkSAgICcHd3p2XLlhQUFBAbG0tsbCxmZmZYWFhotFcoFFy4cKFcCxZCmCYTHdzWHpLjxo1j4cKFuLu706tXrwp93kAIUbGVJT+ioqKIiYnRmOfg4MCJEyeAB+/Xio6O5ttvvyUrK4sWLVrw6aef4u3trZeaoYSQLCgoUP99/vz5etuoEOLJU9YRbC8vL9avX6/+/OhbWefOnas+uvXy8mLGjBn06tWLffv2YW1t/ZgVP1CRR96FEJVIWUe3q1SpgqOjo3qqXbs28KATN3/+fCZMmEBoaCg+Pj7Mnz+fnJwcVq5cqbe6SwxJOcQWQuhDWUe3T58+jbe3N82aNWP48OGcPn0agDNnznD58mU6duyobmtlZUVAQACpqal6q7vEO27GjRun86tiZeBGCFGcsrw21t/fny+//BIvLy+uXbvGzJkzCQ4OZs+ePVy+fBkAe3t7je/Y29tz8eJFvdQMOoRkixYt8PDw0NsGhRBPprIclf73NdX+/v74+vryww8/0LJlyyLXW1BQoNcj4BJDMjw8nH79+ultg0KIJ5M+YqtGjRo0btyY9PR0unfvDsCVK1eoW7euus21a9cK9S4fhwzcCCEMQqHQPunizp07/PXXXzg6OuLu7o6joyNJSUkay3fv3k2rVq30Vre8UlYIYRBleUVDZGQkXbp0oW7duupzkrdu3WLgwIEoFArGjBnDrFmz8PLywtPTk08//ZTq1avTt29fvdUtISmEMIiynCa8cOECI0aMIDMzk9q1a+Pv78/WrVvVb0EYP348t2/fZuLEieqLyVevXq23ayShhJC8fv263jYkhHiyleX+7JLedqBQKIiIiND6NtfHJT1JIYRBVOQ3ImojISmEMAhTvS9FRrcN4NddybzcvxdNG7pjb23BksXfaixfv3YN/Xq+SGMPZ+ytLUhJ3mGkSp9cgU3rsmJqb/5eOpbbP0/i5eCnNZa/P6wtafEjuJb4JhfWjKdhw4a09nEptB7/Rs6sjxnA1cQ3ubJuAklzX8auppWhdqNC08fotjFISBpAbm4OjX2a8MmM2VhZFf4Pc+tWLs+2eo6pUTONUJ0AqGFVlaOnr/FO7DZu3ckrtPzE2UwmfLEV/5HxdJqQwN27d1kb1Q8HZTV1m5aNnUmM6c/Ogxk8//r3BIz5ljkr9pInT/QHQFHCn4pKDrcN4IXOXXmhc1cAXh/9SqHl/Qe+DEDmtWsGrUv8n81709m8Nx2A//fui4WWL912VOPz2bNn8fPzo5mnIz/vPwXAjDGdWLj2d2b8sFvd7uR5Gfx8yFSfJyk9SSFKyaKKGfb29mTn3uXQyX/vH1ZWo3UTFy79k8u2OYM4veI1fv7sJdo/427kaisOU30yuYSkEDrq2qoBVxPfJGvjOzg6OtL93WVcyboFQD1nJQCRQ9vw7aZDhEYsJ+XwORKj+9O0vv5ukTNlpnq4bdSQTElJISwsDG9vb5RKJQkJCcYsRwitdhzMoNWor+kwfjHZ2dksfi8Up1rVgf+7BvCr9Wl8t+kwB09e4YP4new/dpERPZ4xZtkVhplC+1RRGTUkc3Nz8fHxITo6usgBDSEqklt38ki/kMXePy9w5swZ8vLvM6xrcwAu/pMDwJ9nNM8rH8vIxNWhpsFrrYhMtSdp1IGb4OBggoODARg7dqwxSxGi1MzMFFhWffAqgTOXsrlw7SYNXe002njVteWPU1eNUV6FU4FPO2olo9sGkJOTw6n0kwAUqFScP3eWw4fSsLWtBcD1f/7h3LkMbmRnA5D+90lq2tjg4OiEo6OT0ep+klR/yoIGLrbAg0NnV4eaNGvgwPWbt8nKuctbA1qxcfdJLv2TQ22banh4eGBd05pV24+p1/HZ8r1EDm3DH+lXSDt5mT7PN+ZZ7zq8+cVWY+1WhWKiGYkiKyuroORm5c/FxYUZM2YwaNAgre3++usvA1WkP7/99hujR48uNL9bt258+OGHJCYmMnXq1ELLX331VUaOHGmIEp941tbWNGrUqND8a9eukZGRQb169ahevTpVqlQhPz+f3NxcLl26RG5urkZ7Jycn7O3tqVKlCnfu3OHcuXPcvHnTULvx2Ly8vMpt3X9e1R413vYVM0ZNLiTzzKsbqCLDOH0sDY/GvsYuo9y49pxl7BLKRXJ0J9pO3mbsMvTu0qrCv8z15c9rJYRk7YoZknK4LYQwiIo8OKONhKQQwiBk4KYMcnJySE9/cCuYSqXi3LlzHDp0CFtbW1xdXY1ZmhBCz0w0I417neSBAwdo164d7dq14/bt20RFRdGuXTumT59uzLKEEOWhrC/eNjKj9iTbtm1LVlaWMUsQQhhIyfdnV4gx5ELknKQQwiAqcGdRKwlJIYRhmGhKSkgKIQyi5EuA5HBbCPEEk0uAhBBCCxPNSAlJIYRhKEy0KykhKYQwCBPNSAlJIYRhmGhGSkgKIQzERFNSXgQmhDCIsry+Yfbs2XTo0AFXV1caNGjAgAEDOHpU8/W+Y8aMQalUakxBQUF6q1t6kkIIgyjLOcldu3bxyiuv4OfnR0FBAdOnT6dnz56kpqZia2urbte+fXsWLlyo/ly1alV9lAxISAohDKQsIbl69WqNzwsXLsTNzY09e/bQtWtX9XxLS0scHR0ft8QiyeG2EMIg9PG2xJycHFQqFUqlUmP+7t278fT0pEWLFrzxxhtcvaq/l69JT1IIYRD6uARo8uTJNG3alGeffVY9LygoiB49euDu7k5GRgbTpk0jJCSE7du3Y2lp+djblJAUQhjE42bk//73P/bs2cOmTZswNzdXz+/Tp4/6702aNMHX15emTZuyefNmQkJCHnOrEpJCCEN5jJSMiIhg9erVJCYm4uHhobWts7MzderUUb/14HFJSAohDKKsLwKbNGkSq1evZv369TRs2LDE9pmZmVy8eFFvAzkSkkIIgzArQ0a+8847LFu2jMWLF6NUKrl8+TIA1atXp0aNGuTk5BAdHU1ISAiOjo5kZGQwdepU7O3t6d69u17qlpAUQhhGGUIyLi4OgNDQUI35kyZNIiIiAnNzc44ePcrSpUvJzs7G0dGRtm3b8vXXX2Ntba2PqiUkhRCGUZbD7ZLegWVlZVXoWkp9k5AUQhiEPAVICCG0MNGMlJAUQhiIiaakhKQQwiBKfu92xSQhKYQwCNOMSAlJIYSBmGhHUkJSCGEoppmSEpJCCIOQnqQQQmhhohkpISmEMAwZ3RZCCG1KysgCg1RRahKSQgiDMM1+pISkEMJASjzalp6kEOJJVtaH7hqbhKQQwjBMMyMlJIUQhlGWJ5NXBBKSQgiDkMNtIYTQwkQvk8TM2AUIIURFJj1JIYRBmGpPUkJSCGEQck5SCCG0KGl0W2WYMkpNQlIIYRim2ZGUkBRCGIYcbgshhBYycCOEEFqYaEZKSAohDMREU1KRlZVVQR9QJIQQxid33AghhBYSkkIIoYWEpBBCaCEhKYQQWkhICiGEFhKSRhQXF0ezZs1wdHTk+eef59dffzV2SUKLlJQUwsLC8Pb2RqlUkpCQYOyShAFISBrJ6tWrmTx5Mm+//TY7d+7k2WefpV+/fpw9e9bYpYli5Obm4uPjQ3R0NFZWVsYuRxiIXCdpJJ06daJJkyZ8/vnn6nl+fn6EhobywQcfGLEyoQsXFxdmzJjBoEGDjF2KKGfSkzSCe/fukZaWRseOHTXmd+zYkdTUVCNVJYQoioSkEWRmZnL//n3s7e015tvb23PlyhUjVSWEKIqEpBEp/vNYlIKCgkLzhBDGJSFpBHZ2dpibmxfqNV67dq1Q71IIYVwSkkZQtWpVfH19SUpK0piflJREq1atjFSVEKIo8qg0Ixk3bhyjRo2iRYsWtGrVivj4eC5dukR4eLixSxPFyMnJIT09HQCVSsW5c+c4dOgQtra2uLq6Grk6UV7kEiAjiouLY+7cuVy+fBlvb2+mT59OYGCgscsSxUhOTqZHjx6F5g8cOJD58+cboSJhCBKSQgihhZyTFEIILSQkhRBCCwlJIYTQQkJSCCG0kJAUQggtJCSFEEILCcknSFRUFEqlUmNe06ZNGTNmjHEKKoZSqSQqKkpv7f7rzJkzKJVKPvvss7KUp/d6RMUmIWkgCQkJKJVK9WRnZ4ePjw+vvfYaly5dMnZ5pZKTk0NUVBTJycnGLkWIcie3JRrY5MmTqVevHnfv3mXPnj388MMPpKSk8Ouvvxrladf79+/HzKx0vytzc3OJiYkBoG3btuVRlhAVhoSkgXXq1ImWLVsCMGTIEGxtbYmNjWXjxo306dOnyO/cunWLatWqlUs9lpaW5bJeISoLOdw2snbt2gFw+vRpAMaMGYOjoyMZGRm89NJLuLm50a9fP3X7VatW0alTJ5ydnXFzc2PAgAEcO3as0Ho3b95MYGAgjo6OtGjRgu+++67I7Rd1TvLevXvMnDmTli1b4uDggJeXFwMHDuTPP//kzJkzNGrUCICYmBj16YNH13Hp0iXGjx9P48aNcXBwwM/Pj7lz51JQoHkH7I0bNxg/fjweHh64uroyePDgxzr1cP36dSIjIwkICKBu3bq4uLjQvXt39uzZU+x3Fi5cSLNmzXByciIoKIj9+/cXaqPr/ojKSXqSRnbq1CkAatWqpZ6nUqno3bs3fn5+fPTRR5ibmwMwZ84cPvzwQ3r06EFYWBi5ubnExcXRuXNnduzYgYeHBwA7duzgpZdeon79+kyZMoU7d+7w8ccf4+joWGI9KpWKgQMHsm3bNkJCQnj11Ve5ffs2ycnJpKWlERISwsyZM5k4cSLdu3dXP/ChXr16AFy9epWgoCDy8/MZOnQoTk5O7N69mw8++ICLFy8SHR0NPHjA8Msvv0xycjKDBw+madOmbN++XeMXQmmdPn2atWvXEhoaSv369cnOzua7774jNDSUpKQkfHx8NNqvWLGC7OxsXnnlFVQqFXFxcfTs2ZPt27fj6elZqv0RlZeEpIHduHGDzMxM7ty5Q2pqKjNmzMDKyorOnTur2+Tl5REcHMz06dPV886ePcu0adOYNGkSERER6vlhYWE8++yzfPrpp8ybNw+A999/H6VSyZYtW7C1tQUgNDSUgICAEutbsmQJ27ZtIzIyknfeeUc9f/z48eonp4eEhDBx4kSaNGnCgAEDNL4/bdo07t69S0pKCg4ODgCEh4fj5OTEvHnzGDNmDO7u7mzatImdO3fyv//9j3fffReAV199lVdffZXDhw+X9scKgI+PD2lpaepfKgDDhg2jZcuWLFiwQOOlawAnT55k3759uLu7A9CzZ09at25NdHQ0cXFxpdofUXnJ4baB9enThwYNGtCkSROGDx+Oo6Mjy5Yto06dOhrtRowYofE5MTGR/Px8+vTpQ2ZmpnqysLDA39+fnTt3AnD58mUOHjxIWFiYOiABGjVqRKdOnUqsb926ddjY2PD6668XWlbSqyUKCgpYu3YtnTt3xtzcXKPOTp06oVKpSElJAR6cDjAzM2PUqFEa63icy5EsLS3VAXnnzh3++ecfVCoVLVq0IC0trVD7rl27agScp6cnnTp1YuvWraXeH1F5SU/SwGJiYmjUqBGWlpbUrVuXunXrFgofMzMz3NzcNOb9/fffADz77LNFrvfhwE5GRgYAXl5ehdp4enqyZcsWrfWdOnUKT0/PMg3oXLt2jaysLBYvXszixYuLbQMPesYODg7Y2NgUqrGsVCoVc+fO5ZtvvuHMmTMay4rq7TVo0KDIeZs3byY7O5t79+7pvD+i8pKQNDA/Pz/16HZxLCwsqFJF859GpVIBsHLlykLLAPVlPA8HE4rq9eky0PA4LyN7WGPfvn15+eWXi2xTv379x95OcebMmcPUqVMZOHAgkZGR1KpVC3Nzc2bPnq0+9/uokn5GpdkfUXlJSJqIhwMjdevWpXHjxsW2e9hjOnHiRKFlD3uj2tSvX5/U1FTu3btH1apVi2xTXLjVrl2bmjVrkp+fT/v27bVux83Nje3bt5Odna3Rmzx58mSJNRZn9erVtGnTptBTwou7C6aobaWnp2NjY4ONjQ01atTQeX9E5SXnJE1ESEgIVapUISoqSt3DedTDwz5HR0eaNWvG0qVLuX79unr58ePH2bZtm07bycrKIjY2ttCyh72sh4f2WVlZGsvNzc0JCQlh/fr1RZ4DzM7OJi8vD4Dg4GBUKhULFy7UaPM4r0EwNzcv1FtOTU1l7969RbbftGmTxmH5yZMn2bZtG0FBQaXeH1F5SU/SRHh4ePDRRx8xZcoUgoKC6NGjB7a2tpw9e5YtW7bg7++vvhf5o48+ok+fPgQHBzNkyBBu377NokWL8Pb25o8//tC6nbCwMJYvX85HH33EwYMHCQwM5M6dO+zatYtevXoRFhZGjRo18PLyYvXq1Xh6elKrVi3c3d3x9/fnww8/JCUlhS5dujB48GB8fHy4efMmR48eJTExkd9//x1HR0e6du1KYGAgUVFRnDt3jmbNmpGUlFToXGJpdO3alejoaEaNGkVAQAB///0333zzDY0bNyYnJ6dQ+wYNGvDiiy8yYsQIVCoVixYtwtLSkkmTJqnb6Lo/ovKSkDQh48aNw9PTky+++ILZs2eTn5+Ps7MzrVu3ZvDgwep2HTp0ICEhgY8//piPP/4YV1dX3nvvPc6fP19iSJqbm7Ns2TJmzZrFypUr2bBhA7a2tvj7++Pr66tuFxsbS0REBJGRkdy9e5eBAwfi7+9P7dq12bZtGzNnzmTDhg1888032NjY4OnpyeTJk9Uj7gqFgh9++IHIyEh+/PFH1qxZw/PPP8+KFSvw9vYu08/nrbfe4vbt26xYsYK1a9fi7e1NfHw8q1atYteuXYXa9+vXj2rVqhEbG8vly5d5+umnmT59Og0bNlS30XV/ROUlLwITQggt5JykEEJoISEphBBaSEgKIYQWEpJCCKGFhKQQQmghISmEEFpISAohhBYSkkIIoYWEpBBCaCEhKYQQWvx/7KSG+SK1AmIAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def plot_confusion_matrix(cm, classes,\n",
    "                          title='Confusion matrix',\n",
    "                          cmap=plt.cm.Blues):\n",
    "    plt.imshow(cm, interpolation='nearest', cmap=cmap)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick_marks = np.arange(len(classes))\n",
    "    plt.xticks(tick_marks, classes, rotation=0)\n",
    "    plt.yticks(tick_marks, classes)\n",
    "\n",
    "    thresh = cm.max() / 2.\n",
    "    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):\n",
    "        plt.text(j, i, cm[i, j],\n",
    "                 horizontalalignment=\"center\",\n",
    "                 color=\"white\" if cm[i, j] > thresh else \"black\")\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.ylabel('True label')\n",
    "    plt.xlabel('Predicted label')\n",
    "\n",
    "# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test\n",
    "# dataset\n",
    "lr = LogisticRegression(C=best_c, penalty='l2')\n",
    "lr.fit(X_train_undersample, y_train_undersample.values.ravel())\n",
    "y_pred_undersample = lr.predict(X_test_undersample.values)\n",
    "\n",
    "# Compute confusion matrix\n",
    "cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)\n",
    "np.set_printoptions(precision=2)\n",
    "\n",
    "recall = cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])\n",
    "print(\"Recall metric in the testing dataset: \", recall)\n",
    "\n",
    "# Plot non-normalized confusion matrix\n",
    "class_names = [0, 1]\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0d66a2b3",
   "metadata": {},
   "source": [
    "Our model predicts a 93.2% recall on the undersampled test set. We'll try it now with our whole data to see if it still works."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "3a8650a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recall metric in the testing dataset:  0.9251700680272109\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def plot_confusion_matrix(cm, classes,\n",
    "                          title='Confusion matrix',\n",
    "                          cmap=plt.cm.Blues):\n",
    "    plt.imshow(cm, interpolation='nearest', cmap=cmap)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick_marks = np.arange(len(classes))\n",
    "    plt.xticks(tick_marks, classes, rotation=0)\n",
    "    plt.yticks(tick_marks, classes)\n",
    "\n",
    "    thresh = cm.max() / 2.\n",
    "    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):\n",
    "        plt.text(j, i, cm[i, j],\n",
    "                 horizontalalignment=\"center\",\n",
    "                 color=\"white\" if cm[i, j] > thresh else \"black\")\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.ylabel('True label')\n",
    "    plt.xlabel('Predicted label')\n",
    "\n",
    "# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test\n",
    "# dataset\n",
    "lr = LogisticRegression(C=best_c, penalty='l2')\n",
    "lr.fit(X_train_undersample, y_train_undersample.values.ravel())\n",
    "y_pred = lr.predict(X_test.values)\n",
    "\n",
    "# Compute confusion matrix\n",
    "cnf_matrix = confusion_matrix(y_test, y_pred)\n",
    "np.set_printoptions(precision=2)\n",
    "\n",
    "recall = cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])\n",
    "print(\"Recall metric in the testing dataset: \", recall)\n",
    "\n",
    "# Plot non-normalized confusion matrix\n",
    "class_names = [0, 1]\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8eb54f36",
   "metadata": {},
   "source": [
    "Still almost identical! That's great. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "de239945",
   "metadata": {},
   "source": [
    "#### We'll check it again with our skewed data using Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "901867f4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C parameter:  0.01\n",
      "Recall:  0.5373134328358209\n",
      "Recall:  0.6164383561643836\n",
      "Recall:  0.6666666666666666\n",
      "Recall:  0.6\n",
      "Recall:  0.5\n",
      "C parameter:  0.1\n",
      "Recall:  0.5522388059701493\n",
      "Recall:  0.6164383561643836\n",
      "Recall:  0.7166666666666667\n",
      "Recall:  0.6153846153846154\n",
      "Recall:  0.5625\n",
      "C parameter:  1\n",
      "Recall:  0.5522388059701493\n",
      "Recall:  0.6164383561643836\n",
      "Recall:  0.7333333333333333\n",
      "Recall:  0.6153846153846154\n",
      "Recall:  0.575\n",
      "C parameter:  10\n",
      "Recall:  0.5522388059701493\n",
      "Recall:  0.6164383561643836\n",
      "Recall:  0.7333333333333333\n",
      "Recall:  0.6153846153846154\n",
      "Recall:  0.575\n",
      "C parameter:  100\n",
      "Recall:  0.5522388059701493\n",
      "Recall:  0.6164383561643836\n",
      "Recall:  0.7333333333333333\n",
      "Recall:  0.6153846153846154\n",
      "Recall:  0.575\n",
      "Best model to choose from cross validation is with C parameter = 1\n"
     ]
    }
   ],
   "source": [
    "best_c = print_kfold_scores(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "2b88eebf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Recall metric in the testing dataset:  0.6190476190476191\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def plot_confusion_matrix(cm, classes, title):\n",
    "    \"\"\"\n",
    "    Plot confusion matrix using matplotlib.\n",
    "    \"\"\"\n",
    "    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick_marks = np.arange(len(classes))\n",
    "    plt.xticks(tick_marks, classes, rotation=0)\n",
    "    plt.yticks(tick_marks, classes)\n",
    "\n",
    "    thresh = cm.max() / 2.\n",
    "    for i, j in np.ndindex(cm.shape):\n",
    "        plt.text(j, i, format(cm[i, j], 'd'),\n",
    "                 horizontalalignment=\"center\",\n",
    "                 color=\"white\" if cm[i, j] > thresh else \"black\")\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.ylabel('True label')\n",
    "    plt.xlabel('Predicted label')\n",
    "\n",
    "# Fit logistic regression model using \"best_c\" as the regularization strength\n",
    "lr = LogisticRegression(C=best_c, penalty='l2')\n",
    "lr.fit(X_train, y_train.values.ravel())\n",
    "\n",
    "# Make predictions on the test data\n",
    "y_pred = lr.predict(X_test.values)\n",
    "\n",
    "# Compute confusion matrix\n",
    "cnf_matrix = confusion_matrix(y_test, y_pred)\n",
    "np.set_printoptions(precision=2)\n",
    "\n",
    "# Calculate recall metric\n",
    "recall = cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])\n",
    "print(\"Recall metric in the testing dataset: \", recall)\n",
    "\n",
    "# Plot non-normalized confusion matrix\n",
    "class_names = [0, 1]\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "197ccb54",
   "metadata": {},
   "source": [
    "## Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "b205460c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9746146553842913\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Create a random forest classifier\n",
    "clf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "\n",
    "# Fit the classifier to the training data\n",
    "clf.fit(X_train_undersample, y_train_undersample)\n",
    "\n",
    "# Predict the target variable for the test data\n",
    "y_pred = clf.predict(X_test)\n",
    "\n",
    "# Convert the predicted target variable into a Pandas Series\n",
    "y_pred = pd.Series(y_pred)\n",
    "\n",
    "# Convert the test target variable into a Pandas Series\n",
    "y_test = pd.Series(y_test.values.ravel())\n",
    "\n",
    "# Compare the two Series element-wise\n",
    "accuracy = np.mean(y_pred == y_test)\n",
    "print('Accuracy:', accuracy)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3f3b655",
   "metadata": {},
   "source": [
    "This is so much better."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "b2c89aeb",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9995318516437859\n"
     ]
    }
   ],
   "source": [
    "import xgboost as xgb\n",
    "\n",
    "# Create an XGBoost model\n",
    "clf = xgb.XGBClassifier(random_state=42)\n",
    "\n",
    "# Fit the model to the training data\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "# Predict the target variable for the test data\n",
    "y_pred = clf.predict(X_test)\n",
    "\n",
    "# Calculate the accuracy of the model\n",
    "accuracy = clf.score(X_test, y_test)\n",
    "print('Accuracy:', accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8627ffc6",
   "metadata": {},
   "source": [
    "And This is even better!"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "421c1fa6",
   "metadata": {},
   "source": [
    "### Conclusion:\n",
    "Our undersampled dataset performed well in our Logistic Regression models however, it will likely missclassify nonfraudulent transactions as fraudulent which will make our customers unhappy. Our Random Forest model did so much better. But the best model was our XGBoost. It seems to do much better with skewed data.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd421a46",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

# Applying Reinforcement Learning in Quantitative Trading

## Overview  

This is the repository of my graduate thesis which aims to use reinforcement learning in quantitative trading.
Two types of RL models were experimented and could make good performance in the back-test:
1. Policy Gradient
    - Vanilla Policy Gradient (not implemented in this repo)
    - Multi-Task Recurrent Policy Gradient (RPG)
2. Direct RL
    - Vanilla DRL (DRL)
    - A hybrid network involves news embedding vectors (DRL with news)

## Experiments

```.ipynb``` files were details of experiments.

This repository contains 3 types of environments:  
1. CryptoCurrency (Huobi): ```env/crc_env.py```  
2. End of day US stock prices (quandl): ```env/stock_env.py```  
3. Continuous Futures (quandl): ```env/futures_env.py```  

And, 2 types of agents:  
1. DRL: ```agents/drl_agent.py``` and ```agents/drl_news_agent.py```  
2. RPG: ```agents/rpg_agent.py``` and ```agents/rpg_news_agent.py```  


Also, there are some history codes in ```history``` and ```model_archive``` which have been deprecated, but contains some early ideas, please feel free to use them.

## Reference

[1] [Deep Direct Reinforcement Learning for Financial Signal Representation and Trading](http://ieeexplore.ieee.org/document/7407387/)  
[2] [Using a Financial Training Criterion Rather than a Prediction Criterion](http://www.worldscientific.com/doi/abs/10.1142/S0129065797000422)  
[3] [A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](http://arxiv.org/abs/1706.10059)  
[4] [Recurrent Reinforcement Learning: A Hybrid Approach](http://arxiv.org/abs/1509.03044)  
[5] [Reinforcement Learning for Trading](http://dl.acm.org/citation.cfm?id=340534.340841)  
[6] [Continuous control with deep reinforcement learning](http://arxiv.org/abs/1509.02971)  
[7] [Memory-based control with recurrent neural networks](https://arxiv.org/abs/1512.04455)  
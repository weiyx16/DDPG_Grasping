# Deep Reinforcement Learning for robotic manuplation

Tensorflow implementation of **DDPG** for our manuplation system  

**A WIP Repo**

This implementation contains:

1. DDPG with **input of image**(use CNN to extract state presentation)  
2. Experience replay memory and history for consecutive 4 frames  
    - to reduce the correlations between consecutive updates  
3. V-Rep simulation for robotic grasping  

## What Remains

Pause since 2019.3.30  
left things:  
1. action definition and how to use ounoise  
2. what to print out and what to inject to summary and when to save both two networks  
3. the replay memory function need to be adapted to our new model(different action dimension)  
4. about the sess.run and .eval()  
5. main function(where to add session graph)  
6. debug with simulation  

And to be honest, although things remaining are not too much and actually the framework is established, maybe recently I'm not going to supplement this algorithm for our grasping system.
But it still can provide some ideas on how to use image as input of DDPG.  

## About V-rep Simulation(environment of deep RL)
please ref to our [DQN version](https://github.com/weiyx16/Active-Perception)  

## License

MIT License.

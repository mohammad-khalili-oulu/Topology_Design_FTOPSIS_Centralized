# OWC/RF HetNet Topology Design Simulator


## Overview

This repository contains the simulator for the research paper titled "An Efficient Topology Design for OWC/RF HetNet by Fuzzy TOPSIS." The simulator is designed to support the experiments and findings presented in the paper.

## Paper Abstract

In the rapidly evolving domain of wireless communications, topology optimization for optical wireless communication and radio frequency heterogeneous network (OWC/RF HetNet) confronts unprecedented challenges. These arise from the distinct characteristics of OWC and RF networks, combined with the swift rise of the Internet of Things nodes. Traditional topology designs, with their limited focus on a few parameters, often overlook a wider array of essential factors. Addressing this, we introduce a Fuzzy Technique for Order Preference by Similarity to the Ideal Solution algorithm as a multi-criteria approach to concurrently consider different attributes like achievable data rates, node-access point distances, and cost-effectiveness for the topology design in the OWC/RF HetNet. Simulation results show that the proposed approach can provide a trade-off between various attributes like data rate, coverage range, and diverse requirement satisfaction. Furthermore, this study paves the way by providing a robust dataset, primed for machine learning integrations in subsequent network design pursuits.

## Simulator Features

- [Feature 1]: It generates feasible topologies and assesses them by Fuzzy TOPSIS.
- [Feature 2]: It is multi-treading Python code.
- [Feature 3]: Fuzzy TOPSIS is implemented in C language to have better computation performance,

## Getting Started
To run this code on server

#!/bin/bash
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --job-name="Python-sim2"
#SBATCH --partition=normal

cp $HOME/ftopsis.so .
source $HOME/myenv/bin/activate
python3 $HOME/parallel_main.py

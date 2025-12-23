#!/bin/bash

# Set your Hugging Face credentials
export HF_TOKEN="your_token_here"
export HF_USERNAME="your_username_here"
export HF_ENDPOINT=https://hf-mirror.com

# Download evaluation datasets
cd data

./hfd.sh Auraithm/AIME2024 --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv AIME2024/AIME2024.json ./
./hfd.sh Auraithm/AIME2025 --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv AIME2025/AIME2025.json ./
./hfd.sh Auraithm/MATH500 --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv MATH500/MATH500.json ./
./hfd.sh Auraithm/GSM8K --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv GSM8K/GSM8K.json ./
./hfd.sh Auraithm/OlympiadBench --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv OlympiadBench/OlympiadBench.json ./

# Download training datasets
./hfd.sh Auraithm/GLM4.6-OpenR1Math-SFT --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv GLM4.6-OpenR1Math-SFT/GLM4.6-OpenR1Math-SFT.json ./
./hfd.sh Auraithm/BigMath-RL --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv BigMath-RL/BigMath-RL.json ./
./hfd.sh Auraithm/Light-OpenR1Math-SFT --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv Light-OpenR1Math-SFT/Light-OpenR1Math-SFT.json ./
./hfd.sh Auraithm/Light-MATH-RL --dataset --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv Light-MATH-RL/Light-MATH-RL.json ./
cd ..

# Download models
cd public

./hfd.sh JetLM/SDAR-8B-Chat --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv JetLM/SDAR-8B-Chat ./
rm -rf JetLM
./hfd.sh OpenMOSS-Team/DiRL-8B-Instruct --tool aria2c -x 10 --hf_token $HF_TOKEN --hf_username $HF_USERNAME
mv OpenMOSS-Team/DiRL-8B-Instruct ./
rm -rf OpenMOSS-Team

cd ..

echo "All datasets and models downloaded successfully!"
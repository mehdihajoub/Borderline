# Project Name

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

Project Borderline aims to address hate speech handling online in a comical way. The project consists of a unique pipeline that combines four models to achieve the following tasks (if certain criteria are met *):

- Extract and transcribe the audio from a video into text.
- Classify the verbal content of the video as either hate speech or not hate speech using a classifier.
- If the video is classified as not hate speech, no further action is taken. Otherwise, the text is passed to the third model.
- The third model, LLM (Llamma 3), is fine-tuned on a custom dataset to reverse the meaning of a phrase. For example, the input phrase 'i hate you' would be transformed to 'i love you'.
- The newly generated opposite phrase is then passed to the fourth and final model, which re-voices the phrase using an AI-generated voice and adjusts the lip movement of the protagonist to match the new phrase.

## Classification

The classifier is trained in the Classifier.ipynb notebook.

## Opposite Generator

The LLM for generating opposite phrases is fine-tuned in the Opposite_Generator.ipynb notebook using unsloth, which allows for faster and more memory-efficient training.

## Final Pipeline

All the building blocks are combined in the Pipeline.ipynb notebook. To load a desired video, the path to the video needs to be changed.

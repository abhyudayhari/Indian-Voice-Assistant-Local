# Indian-Voice-Assistant-Local
This project is a locally hosted Indian Voice Assistant designed to process and respond to voice commands in various Indian languages. The models used in this assistant are all loaded locally, ensuring that the system works offline and that data privacy is maintained. The assistant exhibits a low latency of about 2-3 seconds, providing a smooth user experience.




_**Features**_

**Multilingual Support**: The assistant supports multiple Indian languages, making it accessible to a wide audience.

**Offline Functionality**: All models are loaded locally, allowing the assistant to operate without an internet connection.

**Low Latency**: The system is optimized for quick responses, with an average latency of 2-3 seconds.

**Customizable**: Easily add or modify commands and responses to tailor the assistant to your specific needs.




_**All the models used here are listed below**_



For ASR task: [ASR Model](https://github.com/Open-Speech-EkStep/vakyansh-models) 
Navigate to FineTuned Models and download them into folder structure as 
```markdown

Models/Language
```

For Translation task: [Translational Model](https://github.com/AI4Bharat/IndicTrans2) 

For Speech to Text task: [ Speech to Text Model](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) 


Due to the gpu constraint i.e. limited to 4gb for running it locally we are currently picking MMS (Speech to text) but for larger gpu we can use [INDIC TTS](https://github.com/AI4Bharat/Indic-TTS) which gives natural voices)



_**Steps to run the project**_

```markdown

python3 app.py
```

Note: This project is right now focussed on kannada, hindi and english language but u can download the relevant models and place them in correct  directories and change some code in **app.py**.

## CONTRIBUTING

We welcome contributions to this project! If you'd like to contribute, follow these steps:

1. Fork the repository.
2. Clone your forked repository to your local machine.
3. Create a new branch for your feature or fix.
4. Make your changes and commit them with descriptive messages.
5. Push your changes to your forked repository.
6. Create a pull request to the main repository.

## LICENSE
This project is licensed under the MIT License

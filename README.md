# Capstone-Brick-Modeling

**Project description incoming.**

We also have a [poster](https://www.canva.com/design/DAFZKQlLOLo/2ALw0oHRO8qrPj--Q-8huw/view?utm_content=DAFZKQlLOLo&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink) and website associated with this project (**links to be added soon - if they are visible, the work may not be complete yet.**).


## Getting the Data
Ideally in our pipeline, data would be obtained by pairing sensor data with mappings from our [Brick Schema](https://brickschema.org/) in order to query our desired building and floor in a building for relevant sensors to perform our calculation, then using UCSD's Brick server.

However, we were not able to obtain access to the Brick server in the time we had for the project, so we used data from [a data pull from a previous project](https://github.com/HYDesmondLiu/B2RL/tree/master/real_building_buffers). This represents data on floors 2, 3, and 4 of UC San Diego's EBU-3B (Computer Science) building with data from July 2017 to early January 2019. This data should download automatically when the data part of the pipeline runs - the code should also generate the data, data/raw, data/temp, and data/out directories for you.

We also may use data from the NOAA and the EIA as a part of the model - this data is not autodownloadable without an API key or a specific form to make a request. As such, we've made [this folder on Google Drive](https://drive.google.com/drive/folders/14_XtcM5IIhKrNBvBZEmIwExcd8EffnoE) available which contains the versions of the data we used.

While we don't use it in the final version of our model, we also have the EBU 3B Turtle file (ie. the building's representation in Brick) in our raw data. If you'd like to take a look at this, here's the [link](https://brickschema.org/ttl/ebu3b_brick.ttl). This will not auto-download for you.


## Running the Code
To run the model on the original data and compare to our paper or poster, use "python3 run.py data features model" or "python3 run.py all".

To use a smaller set of test data and test the output for the whole pipeline, use "python3 run.py test".

If you'd like to remove the auto-generated files created by running the code, please use "python3 run.py clean".

## Authors
We are four undergraduate Data Science students in our final year at UC San Diego.

If you're interested in our work, here's where you can find more about us:
| Name | GitHub | LinkedIn |
| ---- | ---- | ---- |
| **Jonah Bomwell** | [Click Here](https://github.com/Jbomwell) | | 
| **Alise Bruevich** | [Click Here](https://github.com/alisebruevich) | |
| **William Nathan** | [Click Here](https://github.com/Xenonition) | |
| **Esperanza Rozas** | [Click Here](https://github.com/ESR76) | [Click Here](https://www.linkedin.com/in/esperanza-r/) |


## Acknowledgments
This project was completed as a capstone project for the Data Science major at UC San Diego.
For more information on the course, please read about the class [here](https://dsc-capstone.github.io/).

Some of our supplemental data comes from the [NOAA (National Oceanic at Atmospheric Administration)](https://www.noaa.gov/) and the [EIA (U.S. Energy Information Administration)](https://www.eia.gov/).

We'd also like to thank: 
- Rajesh Gupta, our mentor and one of the creators of the Brick Schema.
- Xiaohan Fu and Hsin-Yu Liu, who provided us with the data we used and additional mentoring.
- Keaton Chia and the DERConnect team, who helped us with issues related to the Brick Server and supported us with several last minute meetings.
- Suraj Rampure, our instructor for the course.
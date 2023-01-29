# Capstone-Brick-Modeling

**Project description incoming.**

We also have a poster and website associated with this project (**links to be added soon**).


## Getting the Data
Ideally in our pipeline, data would be obtained by pairing sensor data with mappings from our [Brick Schema](https://brickschema.org/) in order to query our desired building and floor in a building for relevant sensors to perform our calculation, then using UCSD's Brick server.

However, we were not able to obtain access to the Brick server in the time we had for the project, so we used data from [a data pull from a previous project][(https://github.com/HYDesmondLiu/B2RL/tree/master/real_building_buffers). This represents data on floors 2, 3, and 4 of UC San Diego's EBU-3B (Computer Science)building with data from July 2017 to early January 2019. This data should download automatically when you run the model using "python3 run.py data" - the code should also generate the data, data/raw, data/temp, and data/out directories for you.

While we don't use it in the final version of our model, we also have the EBU 3B Turtle file (ie. the building's representation in Brick) in our raw data. If you'd like to take a look at this, here's the [link](https://brickschema.org/ttl/ebu3b_brick.ttl). This will not auto-download for you.


## Running the Code
To run the model on the original data and compare to our paper or poster, use "python3 run.py data features model" or "python3 run.py all".

To use a smaller set of test data and test the output, use "python3 run.py test features model".

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

We'd also like to thank: 
- Rajesh Gupta, our mentor and one of the creators of the Brick Schema.
- Xiaohan Fu and Hsin-Yu Liu, who provided us with the data we used and mentoring along the way.
- Keaton Chia and the DERConnect team, who helped us with issues related to the Brick Server.
- Suraj Rampure, our instructor for the course.
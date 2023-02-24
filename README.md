# Capstone-Brick-Modeling

In this project, we attempt to make predictions about future energy usage and cost in a building using energy data collected from UC San Diego's EBU-3B (the Computer Science & Engineering) building's HVAC system.

We also have a [poster](https://www.canva.com/design/DAFZKQlLOLo/2ALw0oHRO8qrPj--Q-8huw/view?utm_content=DAFZKQlLOLo&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink) and [website](https://xenonition.github.io/) associated with this project (**as of 2/23/22 these links are both visible, but do not have complete work yet.**).

### Getting the Data
Ideally in our pipeline, data would be obtained by pairing sensor data with mappings from our [Brick Schema](https://brickschema.org/) in order to query our desired building and floor in a building for relevant sensors to perform our calculation, then using UCSD's Brick server.

However, we were not able to obtain access to the Brick server in the time we had for the project, so we used data from [a data pull from a previous project](https://github.com/HYDesmondLiu/B2RL/tree/master/real_building_buffers). This represents 15 rooms worth of data on floors 2, 3, and 4 of UC San Diego's EBU-3B (Computer Science) building with data from July 2017 to early January 2019. This data should download automatically when the data part of the pipeline runs - the code should also generate the data, data/raw, data/temp, and data/out directories for you.

The features part of the data pipeline performs some data cleaning that is detailed more in our paper, but essentially:
1. We separate our data into training and testing sets based on dates in the original datasets, where approximately 70% of the data is before August 1, 2018 and the rest is August 1, 2018 onwards.
2. We floor timestamps in the dataset to the nearest hour and use medians to aggregate into buckets of that time, since the energy values range because of the 15 unidentified rooms in the dataset.
3. We impute the training datasets with data based on the median for the value at that hour - this was the most stable trend that we found in the original data. We do not impute the test dataset because we want to ensure that we are not predicting false values.


### Other Data and Goals
Along with predicting future energy usage using the energy values from the data pull above, we will be using data from UCSD's pricing plan to scale this for energy. We will only be scaling our data by a constant (derived from UCSD's cost of electricity in fiscal years 2017-2018/2018-2019), although we understand that with UCSD using both its own energy and energy from San Diego Gas and Electric, this likely leads to an underestimation.

While we don't use it in the final version of our model, we also have the EBU 3B Turtle file (ie. the building's representation in Brick) in our raw data to understand relationships between components of the HVAC system in the building. If you'd like to take a look at this, here's the [link](https://brickschema.org/ttl/ebu3b_brick.ttl). This will not auto-download for you.

We also initially pulled other temperature and climate information to use in this project from [NOAA (the National Oceanic and Atmospheric Administration)](https://www.noaa.gov/) and the [EIA (U.S. Energy Information Administration)](https://www.eia.gov/), but we did not end up having time to incorporate this data in our final model.


## Running the Code
To run the model on the original data and compare to our paper or poster, use "python3 run.py data features model" or "python3 run.py all".

To use a smaller set of test data and test the output for the whole pipeline, use "python3 run.py test".

If you'd like to remove the auto-generated files created by running the code, please use "python3 run.py clean".

## Authors
We are four undergraduate Data Science students in our final year at UC San Diego.

If you're interested in our work, here's where you can find more about us:
| Name | GitHub | LinkedIn |
| ---- | ---- | ---- |
| **Jonah Bomwell** | [Link](https://github.com/Jbomwell) | [Link](https://www.linkedin.com/in/jonah-bomwell-0756191b7/) | 
| **Alise Bruevich** | [Link](https://github.com/alisebruevich) | [Link](https://www.linkedin.com/in/alisebruevich/) |
| **William Nathan** | [Link](https://github.com/Xenonition) | [Link](https://www.linkedin.com/in/william-nathan-5019661b2/) |
| **Esperanza Rozas** | [Link](https://github.com/ESR76) | [Link](https://www.linkedin.com/in/esperanza-r/) |


## Acknowledgments
This project was completed as a capstone project for the Data Science major at UC San Diego.
For more information on the course, please read about the class [here](https://dsc-capstone.github.io/).

We'd also like to thank: 
- Rajesh Gupta, our mentor and one of the creators of the Brick Schema.
- Xiaohan Fu and Hsin-Yu Liu, who provided us with the data we used and additional mentoring.
- Keaton Chia and the DERConnect team, who helped us generate ideas for this project, provided us with UCSD's cost model, and discussed the possibilities of using Brick for future work in this area with us.
- Suraj Rampure, our instructor for the course.
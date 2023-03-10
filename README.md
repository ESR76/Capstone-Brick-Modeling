# Energy Cost and HVAC Optimization in Smart Buildings

In this project, we attempt to make predictions about future energy usage and cost in a building using energy data collected from UC San Diego's EBU-3B (the Computer Science & Engineering) building's HVAC system.

We also have a public-facing [poster](https://www.canva.com/design/DAFZKQlLOLo/2ALw0oHRO8qrPj--Q-8huw/view?utm_content=DAFZKQlLOLo&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink) from our poster session and [website](https://xenonition.github.io/) associated with this project.

## Running the Code
To test our code, please first clone the repository. 

If you're concerned that you won't have the proper packages to run something in our code, use the [Docker image](https://hub.docker.com/repository/docker/esr76/capstone-brick-modeling/general) associated with this project - we recommend using the **"final"** tag (the "latest" tag should also work). It is designed to be run on UCSD's JupyterHub service using the base UCSD notebook as a base, for which instructions to use it can be found [here](https://github.com/ucsd-ets/datahub-example-notebook).

Another option is running "pip install -r requirements.txt".

- To run the modeling pipeline on the original data and compare to our paper/poster run the line:
    - "python3 run.py all"
    **OR** 
    - "python3 run.py data features model optimize visualize"

If you try to run a later part of the pipeline (ie. model) before running the earlier parts (ie. data, features), this **will** raise an error.

- To use a smaller set of test data and test the output for the pipeline run:
    - "python3 run.py test"

Results from running this line will appear in the /test directory, with data in /test/testdata and visualizations in /test/testviz.

- To remove any files created by running the script, please run:
    - "python3 run.py clean"

After any call of run.py, the script will run through the steps called, creating files/file organization as necessary, and then will print which steps it took in the order it took them in.

***NOTE:***
Once you've run one of the stages in the main training pipeline or run whole the pipeline before, calling it again will SKIP regenerating the files and print what was skipped. Running the "clean" keyword is the only way to ensure that the files will regenerate. The test pipeline will rerun each time the "test" keyword is called.

## Data Notes

### Getting the Data
Ideally in our pipeline, data would be obtained by pairing sensor data with mappings from our building's [Brick Schema](https://brickschema.org/) in order to query the locations and floors for relevant sensors to perform our calculation, then using UCSD's Brick server.

However, we were not able to obtain access to the Brick server in the time we had for the project, so we used data from [a data pull from a previous project](https://github.com/HYDesmondLiu/B2RL/tree/master/real_building_buffers). This represents 15 rooms worth of data on floors 2, 3, and 4 of UC San Diego's EBU-3B (Computer Science) building with data from July 2017 to early January 2019. This data should download automatically when the data part of the pipeline runs - the code should also generate several directories for you, including: data and its subdirectories, test and its subdirectories, and visualizations.

The features part of the data pipeline is split into two parts that perform data cleaning steps that are detailed more on our website, but essentially:
1. We separate our data into training and testing sets based on dates in the original datasets, where approximately 70% of the data is before August 1, 2018 and the rest is August 1, 2018 and onwards.
2. We floor timestamps in the dataset to the nearest hour and use medians to aggregate into buckets of that time, since the energy values range because of the 15 unidentified rooms in the dataset.
3. We impute the training dataset with data based on the median for the value at that hour - this was the most stable trend that we found in the original data. We do not impute the testing dataset because we want to ensure that we are not evaluating the model on predictions of false values.

### Other Data and Goals
Along with predicting future energy usage using the energy values from the data pull above, we will be using data from UCSD's pricing plan to scale this for energy. We will only be scaling our data by a constant (derived from UCSD's cost of electricity in fiscal years 2017-2018/2018-2019), although we understand that with UCSD using both its own energy and energy from San Diego Gas and Electric, this likely leads to an underestimation.

While we don't use it in the final version of our model, we also have the EBU 3B Turtle file (ie. the building's representation in Brick) in our raw data to understand relationships between components of the HVAC system in the building. If you'd like to take a look at this, here's the [link](https://brickschema.org/ttl/ebu3b_brick.ttl). This will not auto-download for you.

We also initially pulled other temperature and climate information to use in this project from [NOAA (the National Oceanic and Atmospheric Administration)](https://www.noaa.gov/) and the [EIA (U.S. Energy Information Administration)](https://www.eia.gov/), but we did not end up having time to incorporate this data in our final model.

## Credits

### Authors
We are four undergraduate Data Science students in our final year at UC San Diego.

If you're interested in our work, here's where you can find more about us:
| Name | GitHub | LinkedIn |
| ---- | ---- | ---- |
| **Jonah Bomwell** | [Link](https://github.com/Jbomwell) | [Link](https://www.linkedin.com/in/jonah-bomwell-0756191b7/) | 
| **Alise Bruevich** | [Link](https://github.com/alisebruevich) | [Link](https://www.linkedin.com/in/alisebruevich/) |
| **William Nathan** | [Link](https://github.com/Xenonition) | [Link](https://www.linkedin.com/in/william-nathan-5019661b2/) |
| **Esperanza Rozas** | [Link](https://github.com/ESR76) | [Link](https://www.linkedin.com/in/esperanza-r/) |


### Acknowledgments
This project was completed as a capstone project for the Data Science major at UC San Diego in Winter of 2023.
For more information on the course, please read about the class [here](https://dsc-capstone.github.io/).

We'd also like to thank: 
- Rajesh Gupta, our mentor and one of the creators of the Brick Schema.
- Xiaohan Fu and Hsin-Yu Liu, who provided us with the data we used and additional mentoring.
- Keaton Chia and the DERConnect team, who helped us generate ideas for this project, provided us with UCSD's cost model, and discussed the possibilities of using Brick for future work in this area with us.
- Suraj Rampure, our instructor for the course.